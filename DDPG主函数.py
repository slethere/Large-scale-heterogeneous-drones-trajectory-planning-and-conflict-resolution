"""
综合无人机配送路径规划系统
==============================================
宏观分配  : ALNS（自适应大邻域搜索）
微观航迹  : DronePathPlanner（障碍物感知避障）
环境因素  : 风场影响——宏观等效速度 + 微观逐段耗电
==============================================
运行流程
  1. 读取配送订单 / 建筑物 / 风场 CSV
  2. 专家系统宏观分配——写回真实起飞时刻 + OD 相近任务时间错峰（distancing）
  3. 对每笔订单调用 DronePathPlanner——生成含避障航点的 3D 微观航迹
  4. 汇总输出：分配结果表 + 航迹 CSV + 2D 宏观路径图 + 3D 航迹样例图
"""

# ── 标准库 ────────────────────────────────────────────────────────────────────
import math
import time
import random
import sys
import os

# ── 第三方库 ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (注册 3D 投影)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Callable
from matplotlib.lines import Line2D


def _first_usable_cjk_sans_family() -> Optional[str]:
    """
    探测本机可用的中文无衬线字体；避免 matplotlib / mplot3d 退回 Arial 导致 CJK 缺字告警。
    """
    candidates = (
        'Microsoft YaHei',
        'Microsoft JhengHei',
        'SimHei',
        'PingFang SC',
        'Source Han Sans SC',
        'Noto Sans CJK SC',
        'Arial Unicode MS',
    )
    for name in candidates:
        path = fm.findfont(fm.FontProperties(family=name))
        if not path:
            continue
        base = os.path.basename(path).lower()
        if 'dejavu' in base:
            continue
        try:
            ret = fm.FontProperties(fname=path).get_name()
        except Exception:
            ret = name
        if ret:
            return ret
    return None


_CJK_SANS_FAMILY = _first_usable_cjk_sans_family()
if _CJK_SANS_FAMILY:
    _fallback = ('DejaVu Sans', 'Arial')
    matplotlib.rcParams['font.sans-serif'] = [_CJK_SANS_FAMILY] + list(_fallback)

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False


def _zh_font_props() -> Optional[fm.FontProperties]:
    """显式传给 set_title / legend / text；3D 子图否则易退回 Arial 报 Glyph missing。"""
    if not _CJK_SANS_FAMILY:
        return None
    return fm.FontProperties(family=_CJK_SANS_FAMILY)


def _apply_zh_fp_to_3d_axes(ax, fp: fm.FontProperties) -> None:
    """轴标题、刻度文字统一中文字体，便于 tight_layout / savefig 正确量字。"""
    ax.xaxis.label.set_fontproperties(fp)
    ax.yaxis.label.set_fontproperties(fp)
    ax.zaxis.label.set_fontproperties(fp)
    ax.title.set_fontproperties(fp)
    for t in ax.get_xticklabels():
        t.set_fontproperties(fp)
    for t in ax.get_yticklabels():
        t.set_fontproperties(fp)
    for t in ax.get_zticklabels():
        t.set_fontproperties(fp)

# ── 从 drone_obstacle_avoidance.py 导入微观航迹规划器 ────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drone_obstacle_avoidance import (
    Building,
    DeliveryTask,
    Drone,
    DronePathPlanner,
    load_wind_from_csv,
    load_wind_timeseries,
    WindTimeseries,
    BUILDING_THRESHOLD,
    SAFETY_MARGIN,
    CRUISE_HEIGHT,
    WIND_POWER_FACTOR,
    path_total_reward,
    FLEET_NUM_FIXED_WING,
    FLEET_NUM_MULTIROTOR,
    MR_BATTERY_CAPACITY_WH,
    FW_BATTERY_CAPACITY_WH,
    MACRO_MR_BATTERY_KM,
    MACRO_FW_BATTERY_KM,
    fleet_fixed_wing_count,
    macro_battery_km_for_craft,
    build_fleet_micro_drones,
)

# ══════════════════════════════════════════════════════════════════════════════
# §1  基础工具函数
# ══════════════════════════════════════════════════════════════════════════════

def haversine(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """两点 (lon, lat) Haversine 距离，单位 km。"""
    lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
    lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * math.asin(math.sqrt(a)) * 6371.0


def wind_effective_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    wind_speed_ms: float,
    wind_dir_deg: float,     # 风"吹向"方向（度，顺时针，北=0）
    base_speed_kmh: float,   # 无人机巡航空速 km/h
) -> float:
    """
    计算考虑风场后的等效飞行距离（km，以无风时为基准）。
    逆风 → 等效距离增大（耗时/耗电增加）；顺风 → 等效距离减小。
    """
    real_km = haversine(p1, p2)
    if real_km < 1e-6 or wind_speed_ms <= 0:
        return real_km

    # 航段朝向（度，北=0，顺时针）
    dlon = (p2[0] - p1[0]) * math.cos(math.radians((p1[1] + p2[1]) / 2))
    dlat = p2[1] - p1[1]
    heading_deg = math.degrees(math.atan2(dlon, dlat)) % 360

    # 风沿航向分量（m/s），正=顺风，负=逆风
    delta_rad = math.radians(wind_dir_deg - heading_deg)
    wind_along_ms = wind_speed_ms * math.cos(delta_rad)

    base_speed_ms = base_speed_kmh / 3.6
    ground_speed_ms = max(base_speed_ms + wind_along_ms, base_speed_ms * 0.2)

    # 等效距离 = 实际时间 × 基准空速  → km
    time_h = real_km / (ground_speed_ms * 3.6)
    return time_h * base_speed_kmh


def _wind_flight_info(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    wind_speed_ms: float,
    wind_dir_deg: float,
    base_speed_kmh: float,
    wind_power_factor: float = 0.12,
) -> Tuple[float, float]:
    """
    同时返回风影响下的飞行时间（小时）和电量消耗（km 等效）。

    时间与能量分开计算：
      - 飞行时间：由地速决定（顺风快、逆风慢）
      - 能量消耗：由空速功率决定（逆风时每 m/s 增加 wind_power_factor 的功率）
        能量消耗 ∝ 功率 × 时间 = base_power × (1 + factor×|headwind|/base_spd) × time

    返回：(flight_hours, battery_km)
      - flight_hours : 实际飞行时间（小时）
      - battery_km   : 等效电量消耗（km），与 AlnsUAV.max_battery 单位一致
    """
    real_km = haversine(p1, p2)
    if real_km < 1e-6:
        return 0.0, 0.0

    dlon = (p2[0] - p1[0]) * math.cos(math.radians((p1[1] + p2[1]) / 2))
    dlat = p2[1] - p1[1]
    heading_deg = math.degrees(math.atan2(dlon, dlat)) % 360

    delta_rad    = math.radians(wind_dir_deg - heading_deg)
    wind_along_ms = wind_speed_ms * math.cos(delta_rad)

    base_speed_ms   = base_speed_kmh / 3.6
    ground_speed_ms = max(base_speed_ms + wind_along_ms, base_speed_ms * 0.2)

    flight_hours = real_km / (ground_speed_ms * 3.6)

    # 逆风时额外功率（headwind > 0 表示逆风）
    headwind_ms  = max(0.0, -wind_along_ms)          # 仅逆风分量
    power_factor = 1.0 + wind_power_factor * headwind_ms / max(base_speed_ms, 1e-3)
    battery_km   = real_km * power_factor            # 等效电量消耗

    return flight_hours, battery_km


def _find_station_info(
    pos: Tuple[float, float],
    stations: List[Tuple[float, float]],
    wind_speed_ms: float,
    wind_dir_deg: float,
    speed_kmh: float,
) -> Tuple[Tuple[float, float], float, float]:
    """
    找最近充电站，返回 (station_pos, flight_hours_to_station, battery_km_to_station)。
    同时考虑风对时间和电量的不同影响。
    """
    best_station = stations[0]
    best_h, best_bat = _wind_flight_info(pos, stations[0], wind_speed_ms, wind_dir_deg, speed_kmh)
    for st in stations[1:]:
        h, bat = _wind_flight_info(pos, st, wind_speed_ms, wind_dir_deg, speed_kmh)
        if bat < best_bat:
            best_station, best_h, best_bat = st, h, bat
    return best_station, best_h, best_bat


# ══════════════════════════════════════════════════════════════════════════════
# §2  ALNS 任务 / 无人机 类（仅保留宏观调度所需字段）
# ══════════════════════════════════════════════════════════════════════════════

_DELIVERY_SLA_FACTOR = 2.0     # 允许飞行时间 = 直线时间 × SLA 系数（裕度）
_CRUISE_SPEED_MS     = 15.0    # 巡航速度（m/s），用于估算飞行时间


class AlnsTask:
    """宏观调度任务，与 drone_obstacle_avoidance.DeliveryTask 分离。"""
    def __init__(self, task_id, sender_X, sender_Y, receiver_X, receiver_Y,
                 otime: str, dronetime: str, order_cnt: int = 1):
        self.task_id    = int(task_id)
        self.sender     = (float(sender_X), float(sender_Y))   # (lon, lat)
        self.receiver   = (float(receiver_X), float(receiver_Y))
        self.otime      = datetime.strptime(otime.strip(),     "%Y/%m/%d %H:%M")
        self.dronetime  = datetime.strptime(dronetime.strip(), "%Y/%m/%d %H:%M")
        self.dronetime_ts = self.dronetime.timestamp()
        self.order_cnt  = int(order_cnt)
        # 估算截止时间：dronetime + 直线飞行时间 × SLA 系数
        # 直线距离（米）：Haversine 近似
        _dlat = (self.receiver[1] - self.sender[1]) * 111000.0
        _dlon = (self.receiver[0] - self.sender[0]) * 111000.0 * math.cos(
            math.radians(self.sender[1]))
        _dist_m = math.hypot(_dlat, _dlon)
        _flight_s = _dist_m / _CRUISE_SPEED_MS * _DELIVERY_SLA_FACTOR
        self.deadline_ts = self.dronetime_ts + max(_flight_s, 300.0)  # 最少 5 分钟

    def set_scheduled_start(self, ts: float, *, keep_deadline: bool = False) -> None:
        """写回调度起飞时刻；默认按原 SLA 重算 deadline，keep_deadline 则保留绝对截止。"""
        abs_deadline = self.deadline_ts
        self.dronetime_ts = float(ts)
        self.dronetime = datetime.fromtimestamp(self.dronetime_ts)
        if keep_deadline:
            self.deadline_ts = abs_deadline
            return
        _dlat = (self.receiver[1] - self.sender[1]) * 111000.0
        _dlon = (self.receiver[0] - self.sender[0]) * 111000.0 * math.cos(
            math.radians(self.sender[1]))
        _dist_m = math.hypot(_dlat, _dlon)
        _flight_s = _dist_m / _CRUISE_SPEED_MS * _DELIVERY_SLA_FACTOR
        self.deadline_ts = self.dronetime_ts + max(_flight_s, 300.0)


class AlnsUAV:
    """宏观调度无人机，电量单位 km（续航里程）。"""
    def __init__(self, uid: int, speed_kmh: float = 75.0,
                 craft_type: str = 'multirotor'):
        self.id             = uid
        self.craft_type     = craft_type
        self.base_speed     = speed_kmh
        self.current_speed  = speed_kmh
        self.max_battery    = macro_battery_km_for_craft(craft_type)
        self.current_battery = self.max_battery
        self.route: List[int] = []
        self.total_distance  = 0.0
        self.last_end_time   = 0.0
        self.charge_log: List    = []
        self.actual_charge_log: List = []
        self.charge_paths: List  = []
        self.infeasible_reason   = ""
        self.charge_time_per_km  = 60        # 秒/km
        self.current_load        = 0
        self.load_speed_penalty  = 0.02
        self.load_energy_penalty = 0.05
        self.flight_height       = 0.1       # km（100m）
        self.takeoff_speed       = 29.0
        self.landing_speed       = 21.5

    def update_load(self, order_cnt: int):
        self.current_load   = order_cnt
        reduction = min(0.5, order_cnt * self.load_speed_penalty)
        self.current_speed  = self.base_speed * (1 - reduction)

    def get_battery_consumption_rate(self) -> float:
        return 1 + min(1.0, self.current_load * self.load_energy_penalty)

    def get_vertical_distance(self) -> float:
        return self.flight_height * 2

    def get_total_vertical_time(self) -> float:
        return (self.flight_height / self.takeoff_speed +
                self.flight_height / self.landing_speed)


# ══════════════════════════════════════════════════════════════════════════════
# §3  充电站位置计算（KMeans 简化版，避免 community 依赖）
# ══════════════════════════════════════════════════════════════════════════════

def _numpy_kmeans(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 10) -> np.ndarray:
    """纯 numpy 实现的 KMeans，返回聚类中心 (k, dim)。"""
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(len(X), k, replace=False)]
    for _ in range(max_iter):
        dists   = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # (n, k)
        labels  = np.argmin(dists, axis=1)
        new_centers = np.array([
            X[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
            for j in range(k)
        ])
        if np.allclose(centers, new_centers, atol=1e-9):
            break
        centers = new_centers
    return centers


def calculate_charging_stations(
    tasks: List[AlnsTask], n_stations: int = 6
) -> List[Tuple[float, float]]:
    """用 KMeans 在所有任务点中选取 n_stations 个充电站（纯 numpy 实现）。"""
    random.seed(10);  np.random.seed(10)
    pts = []
    for t in tasks:
        pts.extend([t.sender, t.receiver])
    pts = list(set(pts))
    if len(pts) <= n_stations:
        return pts[:n_stations]

    X = np.array([[p[0], p[1]] for p in pts])
    # 标准化
    mu, sigma = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sigma
    centers_norm = _numpy_kmeans(Xs, n_stations)
    centers = centers_norm * sigma + mu

    # 将聚类中心对齐到最近实际任务点
    stations = []
    used_pts  = set()
    for c in centers:
        best_pt, best_d = None, float('inf')
        for p in pts:
            if p in used_pts:
                continue
            d = math.hypot(p[0] - c[0], p[1] - c[1])
            if d < best_d:
                best_d, best_pt = d, p
        if best_pt:
            stations.append(best_pt)
            used_pts.add(best_pt)
    while len(stations) < n_stations:
        for p in pts:
            if p not in used_pts:
                stations.append(p)
                used_pts.add(p)
                break
    return stations


def find_nearest_station(
    pos: Tuple[float, float],
    stations: List[Tuple[float, float]]
) -> Tuple[Tuple[float, float], float]:
    best_d, best_s = float('inf'), stations[0]
    for s in stations:
        d = haversine(pos, s)
        if d < best_d:
            best_d, best_s = d, s
    return best_s, best_d


# ══════════════════════════════════════════════════════════════════════════════
# §4  单机距离计算（含风场等效修正）
# ══════════════════════════════════════════════════════════════════════════════

def calc_uav_distance_wind(
    uav_tasks: List[AlnsTask],
    warehouse: Tuple[float, float],
    stations: List[Tuple[float, float]],
    uav: AlnsUAV,
    wind_speed_ms: float,
    wind_dir_deg: float,
) -> float:
    """计算单机任务路径的等效总飞行距离（风修正版）。"""
    def wd(a, b):
        return wind_effective_distance(a, b, wind_speed_ms, wind_dir_deg,
                                       uav.current_speed)

    uav.current_battery = uav.max_battery
    uav.infeasible_reason = ""
    uav.route = []
    uav.actual_charge_log.clear()
    uav.charge_paths.clear()

    total_dist   = 0.0
    current_pos  = warehouse
    current_time = 0.0

    if uav_tasks:
        vd = uav.get_vertical_distance()
        total_dist   += vd
        current_time += uav.get_total_vertical_time() * 3600
        uav.current_battery -= vd * uav.get_battery_consumption_rate()

    sorted_tasks = sorted(uav_tasks, key=lambda t: t.dronetime_ts)
    safety_margin = 0.25 * uav.max_battery

    for task in sorted_tasks:
        uav.update_load(task.order_cnt)
        cr = uav.get_battery_consumption_rate()
        to_sender = wd(current_pos, task.sender)
        to_recv   = wd(task.sender, task.receiver)
        task_total = to_sender + to_recv + uav.get_vertical_distance() * 2
        battery_needed = task_total * cr * (1 + 0.01 * max(0, to_sender - 10))

        if uav.current_battery < battery_needed + safety_margin:
            ns, ns_d = find_nearest_station(current_pos, stations)
            if uav.current_battery < ns_d * cr * 1.1:
                uav.infeasible_reason = f"任务{task.task_id}前电量不足"
                return float('inf')
            total_dist += ns_d + uav.get_vertical_distance() * 2
            uav.current_battery -= (ns_d + uav.get_vertical_distance() * 2) * cr
            charge_amt = uav.max_battery - max(0, uav.current_battery)
            charge_t   = charge_amt * uav.charge_time_per_km
            current_time += (ns_d / uav.current_speed) * 3600 + charge_t
            uav.current_battery = uav.max_battery
            total_dist += wd(ns, task.sender)
            current_time += (wd(ns, task.sender) / uav.current_speed) * 3600
            current_pos = ns

        if current_time < task.otime.timestamp():
            current_time = task.otime.timestamp()

        task_t  = ((to_recv) / uav.current_speed) * 3600 + uav.get_total_vertical_time() * 3600
        total_dist  += to_sender + to_recv + uav.get_vertical_distance() * 2
        uav.current_battery -= (to_recv + uav.get_vertical_distance() * 2) * cr
        current_time += task_t
        current_pos   = task.receiver
        uav.last_end_time = current_time
        uav.route.append(task.task_id)

    ret_d = wd(current_pos, warehouse)
    cr = uav.get_battery_consumption_rate()
    if uav.current_battery < ret_d * cr * 1.1:
        ns, nd = find_nearest_station(current_pos, stations)
        if uav.current_battery < nd * cr * 1.1:
            uav.infeasible_reason = "末任务后电量不足以返回"
            return float('inf')
        total_dist += nd + haversine(ns, warehouse) + uav.get_vertical_distance() * 2
    else:
        total_dist += ret_d + uav.flight_height

    if len(sorted_tasks) > 3:
        total_dist *= 0.98
    return total_dist


def calc_total_distance_wind(
    solution: List[List[AlnsTask]],
    warehouse: Tuple[float, float],
    uavs: List[AlnsUAV],
    stations: List[Tuple[float, float]],
    wind_speed_ms: float,
    wind_dir_deg: float,
) -> float:
    for uav in uavs:
        uav.charge_log.clear(); uav.charge_paths.clear(); uav.actual_charge_log.clear()
    return sum(
        calc_uav_distance_wind(uav_tasks, warehouse, stations, uavs[i],
                               wind_speed_ms, wind_dir_deg)
        for i, uav_tasks in enumerate(solution)
    )



# ══════════════════════════════════════════════════════════════════════════════
# §6  数据加载
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# §6  数据加载
# ══════════════════════════════════════════════════════════════════════════════

def load_delivery_tasks(csv_path: str) -> List[AlnsTask]:
    df = pd.read_csv(csv_path)
    tasks = []
    for _, row in df.iterrows():
        tasks.append(AlnsTask(
            task_id   = row['task_id'],
            sender_X  = row['sender_X'],
            sender_Y  = row['sender_Y'],
            receiver_X = row['receiver_X'],
            receiver_Y = row['receiver_Y'],
            otime     = row['otime'],
            dronetime = row['dronetime'],
            order_cnt = int(row.get('order_cnt', 1)),
        ))
    return tasks


def load_buildings(csv_path: str) -> List[Building]:
    """读取建筑物 CSV，返回 Building 列表。"""
    df = pd.read_csv(csv_path)
    buildings = []
    for _, row in df.iterrows():
        poly = None
        if 'polygon_vertices' in row and pd.notna(row['polygon_vertices']):
            try:
                raw = str(row['polygon_vertices']).strip()
                import ast
                poly = ast.literal_eval(raw)
            except Exception:
                poly = None
        buildings.append(Building(
            building_id = row.get('building_id', row.name),
            lon  = float(row.get('longitude', row.get('lon', 0))),
            lat  = float(row.get('latitude',  row.get('lat', 0))),
            height = float(row.get('height', 0)),
            polygon_vertices = poly,
        ))
    return buildings


# ══════════════════════════════════════════════════════════════════════════════
# §9  微观航迹规划（DronePathPlanner 集成）
# ══════════════════════════════════════════════════════════════════════════════

def build_micro_drones(num_uavs: int) -> List[Drone]:
    """创建微观耗电估算用 Drone 列表（5 固定翼 + 25 多旋翼 @30 架）。"""
    return build_fleet_micro_drones(num_uavs)


def build_alns_uav_fleet(num_uavs: int, speed_kmh: float = 75.0) -> List[AlnsUAV]:
    n_fw = fleet_fixed_wing_count(num_uavs)
    uavs: List[AlnsUAV] = []
    for i in range(num_uavs):
        ct = 'pure_fixed_wing' if i < n_fw else 'multirotor'
        uavs.append(AlnsUAV(i, speed_kmh=speed_kmh, craft_type=ct))
    return uavs


UAV_SAFE_INTERVAL_M   = 50.0   # 机间水平安全距离（米）
ALT_LAYER_STEP_M      = 30.0   # 高度层间距（米）
ALT_MAX_M             = 120.0  # 最大飞行高度（米），超过此高度的建筑需水平绕行
ALT_LAYERS            = [60.0, 90.0, 120.0]  # 三个合法高度层（60/90/120m）

# 时间维度冲突窗口：两机到达各自航点的时间差小于此值才算真实冲突。
# 对应 MultirotorUAV_4.py 的 12.5s 反应时间；路径规划中取 30s（15m/s×30s=450m 安全裕量）
TIME_CONFLICT_WINDOW_S = 30.0
# 估算航点到达时间用的巡航速度（m/s），与能量模型解耦，只用于时间戳计算
_TS_CRUISE_MS          = 15.0
_TS_VERT_MS            = 4.0   # 爬升/下降速度
# 仅「地面兜底」时域等待：首点 speed≤0.5 且无 `hold_duration_s` 时，段时长取此值（秒）。
# 空中机间时域避让优先用 `_air_timedomain_hover_duration_s`，并写入航点 `hold_duration_s`。
_SPEED_HOLD_WAIT_S     = 40.0
# 空中悬停至少占用的冲突检测步长个数（与 COLLISION2_TIME_STEP_S 相乘得到下限）
_AIR_HOVER_MIN_STEPS   = 3

# conflict——2 机间几何检测参数（对齐 path_uav_interval_reward：水平 50 m、垂直 30 m）
# time_step：规划阶段可从 2.0s 降到 3.0s 进一步提速，冲突边界会略粗。
COLLISION2_TIME_STEP_S = 5.0
COLLISION2_FUTURE_T_S  = 15.0
COLLISION2_VERT_TOL_M  = 30.0
_UAV_COLLISION2_DETECTOR_INSTANCE = None


def _get_uav_collision2_detector_cls():
    import importlib.util
    mod_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conflict——2.py')
    spec = importlib.util.spec_from_file_location('conflict_2_uav_mod', mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.UAVCollisionDetector


def _get_uav_collision2_detector():
    global _UAV_COLLISION2_DETECTOR_INSTANCE
    if _UAV_COLLISION2_DETECTOR_INSTANCE is None:
        cls = _get_uav_collision2_detector_cls()
        _UAV_COLLISION2_DETECTOR_INSTANCE = cls(
            time_step=COLLISION2_TIME_STEP_S,
            T=COLLISION2_FUTURE_T_S,
            height_tolerance_m=COLLISION2_VERT_TOL_M,
        )
    return _UAV_COLLISION2_DETECTOR_INSTANCE


def _air_timedomain_hover_duration_s(cruise_ms: float) -> float:
    """
    空中时域避让悬停时长（秒）：与机间安全间隔、巡航速度、冲突检测时间步长挂钩。

    含义：以巡航速度「沿航迹拉开约一个水平安全间隔」所需时间，向上取整为
    COLLISION2_TIME_STEP_S 的整数倍，且不少于 _AIR_HOVER_MIN_STEPS 个步长。
    """
    spd = max(float(cruise_ms or _TS_CRUISE_MS), 1e-3)
    step = max(float(COLLISION2_TIME_STEP_S), 1e-3)
    sep_t = float(UAV_SAFE_INTERVAL_M) / spd
    n_steps = max(_AIR_HOVER_MIN_STEPS, int(math.ceil(sep_t / step)))
    return n_steps * step


def _p1_score_from_cached_others(
    det,
    states_self,
    other_paths: List[List[Dict]],
    other_states_cached: Optional[List],
    spd: float,
    radius_sum: float,
    scene_lat_ref: float,
) -> float:
    """在已建好本机抽样 states 的前提下累加对他机的冲突惩罚（-2000/条）。"""
    if not states_self or not other_paths:
        return 0.0
    reward = 0.0
    for i, op in enumerate(other_paths):
        sop = None
        if other_states_cached is not None and i < len(other_states_cached):
            sop = other_states_cached[i]
        if sop is None:
            sop = det.build_sampled_states_geo(op, scene_lat_ref)
        if det.pairwise_conflict_from_states(
                states_self, sop, spd, spd, radius_sum):
            reward -= 2000.0
    return reward


def _p1_uav_conflict2_reward(
    path: List[Dict],
    other_paths: List[List[Dict]],
    safe_interval_m: float = 50.0,
    lat_ref: float = 32.0,
    time_window_s: float = None,
    drone: Optional[Drone] = None,
    detector=None,
    other_states_cached: Optional[List] = None,
    states_self_prebuilt=None,
) -> float:
    """
    P1 机间惩罚：每条他机路径若与当前路径几何冲突则为 -2000（与旧 P1 量级一致）。
    判定逻辑为 conflict——2.UAVCollisionDetector.pairwise_conflict_geo。
    time_window_s 仅保留给 _try_altitude_layer(..., time_window_s=...) 签名兼容；
    conflict——2 使用航点绝对时刻 _t 的重叠采样。

    other_states_cached 与 other_paths 等长且按索引对齐时，复用已抽样他机状态，避免
    每条任务 O(n²) 次重复 build_sampled_states_geo（微观规划主耗时之一）。
    states_self_prebuilt 若已传入则不再对本机构建抽样。
    """
    _ = time_window_s
    if not other_paths:
        return 0.0
    det = detector if detector is not None else _get_uav_collision2_detector()
    spd = float(getattr(drone, 'cruise_speed', _TS_CRUISE_MS) or _TS_CRUISE_MS)
    r_each = max(safe_interval_m / 2.0, 1e-3)
    radius_sum = 2.0 * r_each
    if states_self_prebuilt is not None:
        states_self = states_self_prebuilt
    else:
        states_self = det.build_sampled_states_geo(path, lat_ref)
    if not states_self:
        return 0.0
    if (
        other_states_cached is not None
        and len(other_states_cached) == len(other_paths)
    ):
        return _p1_score_from_cached_others(
            det, states_self, other_paths, other_states_cached,
            spd, radius_sum, lat_ref,
        )
    return _p1_score_from_cached_others(
        det, states_self, other_paths, None, spd, radius_sum, lat_ref,
    )


def _make_p1_conflict2_closure(drone: Drone, detector) -> Callable[..., float]:
    def _p1(path, other_paths, safe_interval_m=50.0, lat_ref=32.0, time_window_s=None,
            other_states_cached=None, states_self_prebuilt=None):
        return _p1_uav_conflict2_reward(
            path, other_paths, safe_interval_m, lat_ref, time_window_s,
            drone=drone, detector=detector,
            other_states_cached=other_states_cached,
            states_self_prebuilt=states_self_prebuilt,
        )
    return _p1


def _attach_timestamps(path: List[Dict], start_ts: float) -> List[Dict]:
    """为路径每个航点附加估算到达时间戳（'_t' 字段，unix 秒）。

    · path[0] 时刻 = start_ts（无人机起飞时刻）
    · 相邻航点间距 / 巡航速度 → 累计飞行时间
    · 起点航点含 hold_duration_s → 本段耗时取该秒数（空中时域避让）
    · 否则起点 speed≤0.5 → 本段耗时取 _SPEED_HOLD_WAIT_S（地面兜底等待）
    · 纯垂直段（水平位移 < 0.1m）按垂直速度估算
    对应 MultirotorUAV_4.py 的"12.5s 预判"：我们在路径规划阶段
    用绝对时刻做等价的时间维度过滤，防止时间不重叠的路径被误判冲突。
    """
    tagged: List[Dict] = []
    t = float(start_ts)
    for i, wp in enumerate(path):
        new_wp = dict(wp)
        new_wp['_t'] = t
        if i < len(path) - 1:
            nxt = path[i + 1]
            hd = wp.get('hold_duration_s', None)
            if hd is not None:
                try:
                    t += max(0.0, float(hd))
                except (TypeError, ValueError):
                    seg_spd = float(wp.get('speed', _TS_CRUISE_MS) or _TS_CRUISE_MS)
                    if seg_spd <= 0.5:
                        t += _SPEED_HOLD_WAIT_S
                    else:
                        dlat = (nxt['lat'] - wp['lat']) * 111000.0
                        dlon = (nxt['lon'] - wp['lon']) * 111000.0 * math.cos(math.radians(wp['lat']))
                        dh   = float(nxt.get('height', 0)) - float(wp.get('height', 0))
                        dist_h = math.sqrt(dlat ** 2 + dlon ** 2)
                        t_h  = dist_h / _TS_CRUISE_MS if dist_h > 0.1 else 0.0
                        t_v  = abs(dh) / _TS_VERT_MS  if (abs(dh) > 1.0 and dist_h < 0.1) else 0.0
                        t   += max(t_h, t_v)
            else:
                seg_spd = float(wp.get('speed', _TS_CRUISE_MS) or _TS_CRUISE_MS)
                if seg_spd <= 0.5:
                    t += _SPEED_HOLD_WAIT_S
                else:
                    dlat = (nxt['lat'] - wp['lat']) * 111000.0
                    dlon = (nxt['lon'] - wp['lon']) * 111000.0 * math.cos(math.radians(wp['lat']))
                    dh   = float(nxt.get('height', 0)) - float(wp.get('height', 0))
                    dist_h = math.sqrt(dlat ** 2 + dlon ** 2)
                    t_h  = dist_h / _TS_CRUISE_MS if dist_h > 0.1 else 0.0
                    # 纯垂直段（爬升/下降）
                    t_v  = abs(dh) / _TS_VERT_MS  if (abs(dh) > 1.0 and dist_h < 0.1) else 0.0
                    t   += max(t_h, t_v)
        tagged.append(new_wp)
    return tagged


def _path_min_cruise_height_m(path: List[Dict]) -> Optional[float]:
    """非起降点（height>0）中的最低巡航高度（m）；无巡航航点则为 None。"""
    hs = [float(wp['height']) for wp in path if float(wp.get('height', 0)) > 0]
    return min(hs) if hs else None


def _min_cruise_layer_bucket(path: List[Dict]) -> str:
    """
    将路径的最低巡航高度归入统计桶：合规层键 '60'/'90'/'120'，
    'no_cruise'、'other'（未对齐到 ALT_LAYERS）。
    """
    mn = _path_min_cruise_height_m(path or [])
    if mn is None:
        return 'no_cruise'
    for lay in ALT_LAYERS:
        if abs(mn - float(lay)) < 1.0:
            return str(int(lay))
    return 'other'


def _path_max_cruise_height(path: List[Dict]) -> float:
    """返回路径中非起降点（height > 0）的最大高度，无则返回 0。"""
    heights = [float(wp['height']) for wp in path if float(wp.get('height', 0)) > 0]
    return max(heights) if heights else 0.0


def _snap_path_to_layer(path: List[Dict]) -> List[Dict]:
    """
    将路径所有非起降点的高度对齐到最近的 30m 高度层（30/60/90/120m）。
    - height <= 0 → 保持（起降点）
    - height > 0  → 对齐到合法层中最低的、且 >= 当前高度的那一层（min legal layer ≥ h）
                    若 h > 120m，保持 120m（理论上规则规划器应水平绕行，不应出现此情况）
    这样路径高度严格落在 60/90/120m，高度分层冲突解决才能正确工作。
    """
    new_path = []
    for wp in path:
        new_wp = dict(wp)
        h = float(wp.get('height', 0))
        if h > 0:
            # 找最低的、满足 layer >= h 的合法层
            snapped = next((l for l in ALT_LAYERS if l >= h), ALT_MAX_M)
            new_wp['height'] = float(snapped)
        new_path.append(new_wp)
    return new_path


def _shift_path_to_layer(path: List[Dict], target_h: float) -> List[Dict]:
    """将路径所有非起降点统一设置到 target_h 高度层。"""
    new_path = []
    for wp in path:
        new_wp = dict(wp)
        if float(wp.get('height', 0)) > 0:
            new_wp['height'] = target_h
            orig_action = wp.get('action', '')
            if '高度分层' not in orig_action:
                new_wp['action'] = orig_action + f'  [高度分层→{target_h:.0f}m]'
        new_path.append(new_wp)
    return new_path


def _try_altitude_layer(
    path: List[Dict],
    other_paths: List[List[Dict]],
    det,
    drone: Optional[Drone],
    safe_m: float,
    scene_lat_ref: float,
    other_states_cached: Optional[List],
    time_window_s: float = None,
    ) -> Optional[Tuple[List[Dict], float]]:
    """
    机间冲突「先抬高」策略：在当前路径最低巡航高度之上，枚举所有更高的合规层
    ALT_LAYERS（60 / 90 / 120），用 _shift_path_to_layer 将整个巡航（height>0）抬到目标层，
    取 P1 得分严格优于抬高前的那条；若平局则优先选更低的爬升值（省能）。

    使用 other_states_cached 复用他机抽样，仅对本机候选路径反复 build_sampled_states_geo。
    """
    _ = time_window_s
    cruise_hs = [
        float(wp['height'])
        for wp in path if float(wp.get('height', 0)) > 0
    ]
    if not cruise_hs:
        return None
    min_h = min(cruise_hs)
    if min_h >= ALT_MAX_M - 1e-6:
        return None

    spd = float(getattr(drone, 'cruise_speed', _TS_CRUISE_MS) or _TS_CRUISE_MS)
    r_each = max(safe_m / 2.0, 1e-3)
    radius_sum = 2.0 * r_each

    states_self = det.build_sampled_states_geo(path, scene_lat_ref)
    score_before = _p1_score_from_cached_others(
        det, states_self, other_paths, other_states_cached,
        spd, radius_sum, scene_lat_ref,
    )

    best_path: Optional[List[Dict]] = None
    best_score = score_before
    best_target: Optional[float] = None

    for tgt in ALT_LAYERS:
        if tgt <= min_h + 1e-6:
            continue
        cand = _shift_path_to_layer(path, float(tgt))
        cand_states = det.build_sampled_states_geo(cand, scene_lat_ref)
        sc = _p1_score_from_cached_others(
            det, cand_states, other_paths, other_states_cached,
            spd, radius_sum, scene_lat_ref,
        )
        if sc > best_score + 1e-9:
            best_score = sc
            best_path = cand
            best_target = float(tgt)
        elif (
            best_path is not None
            and abs(sc - best_score) <= 1e-9
            and best_target is not None
            and tgt + 1e-9 < best_target
        ):
            # 同最优得分时优先较低目标层（少爬升）
            best_path = cand
            best_target = float(tgt)

    if best_path is None or best_score <= score_before + 1e-9:
        return None
    dh = (
        float(best_target) - float(min_h)
        if best_target is not None
        else ALT_LAYER_STEP_M
    )
    return best_path, max(dh, ALT_LAYER_STEP_M)


def _baseline_path(dt: DeliveryTask) -> List[Dict]:
    """
    生成起点→终点的无障碍直飞基线路径（4 个航点）。
    用于与实际避障路径做能耗差值，量化避撞额外耗电。
    """
    return [
        {'lon': dt.sender_lon,   'lat': dt.sender_lat,   'height': 0.0,
         'action': '起飞点'},
        {'lon': dt.sender_lon,   'lat': dt.sender_lat,   'height': float(CRUISE_HEIGHT),
         'action': '爬升'},
        {'lon': dt.receiver_lon, 'lat': dt.receiver_lat, 'height': float(CRUISE_HEIGHT),
         'action': '直飞'},
        {'lon': dt.receiver_lon, 'lat': dt.receiver_lat, 'height': 0.0,
         'action': '降落'},
    ]


def _path_flight_time_s(path: List[Dict], default_speed_ms: float = 15.0) -> float:
    """
    根据路径航点序列估算总飞行时间（秒）。
      · 起点含 hold_duration_s → 该段耗时取该秒数（空中时域避让）
      · 否则取起点 speed；缺失时用 default_speed_ms
      · speed<=0.5且无 hold_duration_s → _SPEED_HOLD_WAIT_S（地面兜底）
      · 其余按路程 / 起点速度
    """
    total_s = 0.0
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        hd = p1.get('hold_duration_s', None)
        if hd is not None:
            try:
                total_s += max(0.0, float(hd))
                continue
            except (TypeError, ValueError):
                pass
        dlat = (p2['lat'] - p1['lat']) * 111000.0
        dlon = (p2['lon'] - p1['lon']) * 111000.0 * math.cos(math.radians(p1['lat']))
        dh   = float(p2.get('height', 0)) - float(p1.get('height', 0))
        dist = math.sqrt(dlat ** 2 + dlon ** 2 + dh ** 2)
        spd  = float(p1.get('speed', default_speed_ms) or default_speed_ms)
        if spd <= 0.5:
            total_s += _SPEED_HOLD_WAIT_S
        elif dist > 0:
            total_s += dist / spd
    return total_s


def _path_landing_unix_ts(
    path: List[Dict],
    start_ts: float,
    drone: Optional[Drone] = None,
) -> float:
    """
    航迹末端到达时刻（unix 秒）。优先末航点 _t；否则段累计用时 + start_ts。
    """
    if path:
        tl = path[-1].get('_t')
        if tl is not None:
            try:
                return float(tl)
            except (TypeError, ValueError):
                pass
    spd = float(
        getattr(drone, 'cruise_speed', _TS_CRUISE_MS) or _TS_CRUISE_MS
    ) if drone is not None else _TS_CRUISE_MS
    return float(start_ts) + _path_flight_time_s(path, default_speed_ms=spd)


def _uav_avoidance_meets_deadline(
    path: List[Dict],
    start_ts: float,
    deadline_ts: float,
    drone: Optional[Drone],
) -> bool:
    """预计末端到达不晚于任务截止时间则 True；deadline_ts<=0 视为未配置、不据此判失败。"""
    dl = float(deadline_ts or 0.0)
    if dl <= 0:
        return True
    return _path_landing_unix_ts(path, start_ts, drone) <= dl + 1e-3


def _split_uav_resolve_counts_for_deadline(
    path: List[Dict],
    start_ts: float,
    deadline_ts: float,
    drone: Optional[Drone],
    applied_alt: bool,
    applied_timedomain: bool,
) -> Tuple[bool, bool, bool]:
    """
    已对机间冲突做了抬升或时域处置后：若预计仍晚于 deadline，整体不算「解决成功」，
    且不计入高度分层/时域分项成功计数。

    Returns:
        uav_resolved, count_alt_success, count_timedomain_success
    """
    if not applied_alt and not applied_timedomain:
        return False, False, False
    ok = _uav_avoidance_meets_deadline(path, start_ts, deadline_ts, drone)
    if not ok:
        return False, False, False
    return True, bool(applied_alt), bool(applied_timedomain)


def _deadline_feasible_with_one_ground_hold(
    path: List[Dict],
    start_ts: float,
    deadline_ts: float,
    drone: Optional[Drone],
) -> bool:
    """
    在路径上插入「一次」机间时域空中悬停避让后，预估是否仍能满足 deadline_ts。
    （始终为空中悬停，见 `_insert_timedomain_hold_waypoint`。）
    纯列表操作 + `_attach_timestamps`，用于消解阶段③选择谁等待。
    """
    dl = float(deadline_ts or 0.0)
    if dl <= 0:
        return True
    hyp = _attach_timestamps(
        _insert_timedomain_hold_waypoint([dict(w) for w in path], drone),
        start_ts,
    )
    return _uav_avoidance_meets_deadline(hyp, start_ts, dl, drone)


def _path_smooth_cost(path: List[Dict],
                      coeff1: float = 1.0,
                      coeff2: float = 0.5) -> float:
    """
    轨迹平滑成本 = Σ转弯角(°) × coeff1 + Σ|连续爬升角差|(°) × coeff2

      转弯角：相邻两段水平投影的夹角（0~180°），反映横向机动剧烈程度。
      爬升角差：相邻两段爬升角之差，反映纵向姿态变化剧烈程度。
    """
    n = len(path)
    if n < 3:
        return 0.0

    def seg_vec2d(a: Dict, b: Dict):
        dlat = (b['lat'] - a['lat']) * 111000.0
        dlon = (b['lon'] - a['lon']) * 111000.0 * math.cos(math.radians(a['lat']))
        return dlat, dlon

    def seg_climb_deg(a: Dict, b: Dict) -> float:
        dlat, dlon = seg_vec2d(a, b)
        dist_h = math.sqrt(dlat ** 2 + dlon ** 2)
        dh = float(b.get('height', 0)) - float(a.get('height', 0))
        return math.degrees(math.atan2(dh, dist_h)) if dist_h > 0.1 else 0.0

    # 水平转弯角
    turn_sum = 0.0
    for i in range(1, n - 1):
        v1 = seg_vec2d(path[i - 1], path[i])
        v2 = seg_vec2d(path[i],     path[i + 1])
        l1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        l2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if l1 > 0.1 and l2 > 0.1:
            cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
            cos_a = max(-1.0, min(1.0, cos_a))
            turn_sum += math.degrees(math.acos(cos_a))

    # 连续爬升角差之和
    climbs = [seg_climb_deg(path[i], path[i + 1]) for i in range(n - 1)]
    climb_diff_sum = sum(abs(climbs[j] - climbs[j - 1]) for j in range(1, len(climbs)))

    return turn_sum * coeff1 + climb_diff_sum * coeff2


def _insert_speed_hold_waypoint(path: List[Dict]) -> List[Dict]:
    """
    在路径起点之前插入地面等待航点（历史兜底；当前机间时域避让仅用空中悬停，见下面函数）。

    高度 0、`speed`=0：时段仍由 `_SPEED_HOLD_WAIT_S` 推导（无明示 hold_duration_s 时）。
    """
    if not path:
        return path
    hold_wp: Dict = {
        'lat':    path[0]['lat'],
        'lon':    path[0]['lon'],
        'height': 0.0,
        'speed':  0.0,
        'action': '[降速等待—机间时域避让]',
    }
    return [hold_wp] + list(path)


def _insert_timedomain_hold_waypoint(
    path: List[Dict],
    drone: Optional[Drone],
) -> List[Dict]:
    """
    机间冲突时域避让：仅在空中悬停错峰，不落回地面插队。

      · 有 height>1m 的航点：在该「第一个巡航点」经纬、同高度插入悬停；
      · 否则：在路径首点经纬、按 CRUISE_HEIGHT 对齐到合规巡航层 ALT_LAYERS 插入空中悬停。

    speed=0，段时长写入 hold_duration_s（由 `_air_timedomain_hover_duration_s`）。
    """
    if not path:
        return path
    cruise_ms = float(
        getattr(drone, 'cruise_speed', _TS_CRUISE_MS) or _TS_CRUISE_MS
    ) if drone is not None else float(_TS_CRUISE_MS)
    dur_air = _air_timedomain_hover_duration_s(cruise_ms)
    for i, wp in enumerate(path):
        if float(wp.get('height', 0)) > 1.0:
            hold_wp: Dict = {
                'lon':    wp['lon'],
                'lat':    wp['lat'],
                'height': float(wp['height']),
                'speed':  0.0,
                'hold_duration_s': dur_air,
                'action': '[空中悬停—机间时域避让]',
            }
            return list(path[:i]) + [hold_wp] + list(path[i:])
    # 全路径暂无巡航高度点：仍在首点经纬上空、合规巡航层悬停（不插入地面等待点）
    h0 = float(CRUISE_HEIGHT)
    hover_h = float(next((l for l in ALT_LAYERS if l >= h0), ALT_MAX_M))
    w0 = path[0]
    hold_wp_fallback: Dict = {
        'lon': float(w0['lon']),
        'lat': float(w0['lat']),
        'height': hover_h,
        'speed': 0.0,
        'hold_duration_s': dur_air,
        'action': '[空中悬停—机间时域避让]',
    }
    return [hold_wp_fallback] + list(path)


def _resolve_uav_conflicts_for_trajectory(
    path: List[Dict],
    start_ts: float,
    other_paths_at_plan_time: List[List[Dict]],
    all_paths: List[List[Dict]],
    all_paths_states: List,
    all_paths_info: List[Tuple],
    trajectory_dict: Dict[int, Dict],
    alns_t,
    scene_lat_ref: float,
    det,
    drone: Drone,
    drones: List[Drone],
) -> Tuple[List[Dict], bool, bool, Optional[str], float]:
    """
    机间冲突消解流水线（单一入口，顺序固定）。

    策略（与你方业务约定一致）：
      ① **先抬高**：在合规高度层上多轮抬升，用 P1（conflict——2）在「仍处于冲突」的
         他机子集上优选更高层，每次抬升后重打 `_t`。
      ② **再判定**：与 `all_paths` 活数据对齐后再算 P1；若已无冲突（score≥0）则直接返回，
         不再做时域避让。
      ③ **仍冲突**：在**订单时间更充裕**的一方航迹上插入**空中悬停**错峰（`_insert_timedomain_hold_waypoint`）：
         - 定义充裕度 slack = deadline_ts − dronetime_ts（与当前实现一致）；
         - 在「与本机仍几何冲突」的他机中取出**裕量最小**的一架作为比价对象；
         - 若本方 slack **大于**对方，则由**本方**空中悬停；否则由**对方**悬停（并对其
           已入库 path 回溯重写）；双方在「谁能悬停不伤 deadline」上再做可行性过滤。
       单次空中悬停的 deadline 预判见 `_deadline_feasible_with_one_ground_hold`（命名历史遗留，实为空中插入）。

    all_paths_states 在被改写对方 path 后须在相同下标 rebuild。

    Returns:
        path, applied_alt, applied_timedomain, resolution_method, alt_offset_m
    """
    applied_alt = False
    applied_timedomain = False
    resolution_method: Optional[str] = None
    path_work = list(path)
    initial_max_h = _path_max_cruise_height(path_work)

    spd = float(getattr(drone, 'cruise_speed', _TS_CRUISE_MS) or _TS_CRUISE_MS)
    r_each = max(UAV_SAFE_INTERVAL_M / 2.0, 1e-3)
    radius_sum = 2.0 * r_each
    n_snap = len(other_paths_at_plan_time)

    def _pair_conflict_states(st_a, st_b) -> bool:
        if not st_a or not st_b:
            return False
        return bool(
            det.pairwise_conflict_from_states(st_a, st_b, spd, spd, radius_sum))

    # ① 多轮抬升
    max_rounds = max(1, len(ALT_LAYERS))
    for _ in range(max_rounds):
        min_cr = min(
            (float(w['height']) for w in path_work if float(w.get('height', 0)) > 0),
            default=ALT_MAX_M,
        )
        if min_cr >= ALT_MAX_M - 1e-6:
            break
        sw_states = det.build_sampled_states_geo(path_work, scene_lat_ref)
        if not sw_states:
            break
        cfo: List[List[Dict]] = []
        oscfo: List = []
        for i in range(n_snap):
            if i >= len(all_paths_states):
                break
            op = other_paths_at_plan_time[i]
            sop = all_paths_states[i]
            if _pair_conflict_states(sw_states, sop):
                cfo.append(op)
                oscfo.append(sop)
        if not cfo:
            break
        res_layer = _try_altitude_layer(
            path_work, cfo, det, drone, UAV_SAFE_INTERVAL_M, scene_lat_ref,
            oscfo, time_window_s=TIME_CONFLICT_WINDOW_S,
        )
        if res_layer is None:
            break
        path_work, _ = res_layer
        path_work = _attach_timestamps(path_work, start_ts)
        applied_alt = True

    alt_offset_m = _path_max_cruise_height(path_work) - initial_max_h

    # ② 与他机轨迹「当前权威列表」对齐后再判。
    others_live = list(all_paths)
    sw_live = det.build_sampled_states_geo(path_work, scene_lat_ref)
    score_live = _p1_uav_conflict2_reward(
        path_work, others_live,
        safe_interval_m=UAV_SAFE_INTERVAL_M, lat_ref=scene_lat_ref,
        time_window_s=TIME_CONFLICT_WINDOW_S,
        drone=drone, detector=det,
        other_states_cached=all_paths_states,
        states_self_prebuilt=sw_live,
    )
    if score_live >= 0:
        resolution_method = 'alt_layer' if applied_alt else None
        return path_work, applied_alt, False, resolution_method, alt_offset_m

    # ③ 仍冲突：由订单时间裕量更大的一方空中悬停避让（见下方 cur_slack / conflict_slack）
    applied_timedomain = True
    cur_deadline_ts = float(
        getattr(alns_t, 'deadline_ts', getattr(alns_t, 'dronetime_ts', 0)) or 0)
    cur_dronetime_ts = float(getattr(alns_t, 'dronetime_ts', 0) or 0)
    cur_slack = cur_deadline_ts - cur_dronetime_ts

    sw_phase3 = det.build_sampled_states_geo(path_work, scene_lat_ref)
    idx_live = {id(p): i for i, p in enumerate(all_paths)}

    conflict_tid = None
    conflict_slack: Optional[float] = None
    for (opath, otid, odl_ts) in all_paths_info:
        j = idx_live.get(id(opath))
        if j is None or j >= len(all_paths_states):
            continue
        sop = all_paths_states[j]
        if not _pair_conflict_states(sw_phase3, sop):
            continue
        o_entry = trajectory_dict.get(otid, {})
        o_dronetime_ts = float(o_entry.get('dronetime_ts', 0) or 0)
        o_slack = float(odl_ts) - o_dronetime_ts
        if conflict_slack is None or o_slack < conflict_slack:
            conflict_slack = o_slack
            conflict_tid = otid

    hold_self = False
    hold_other = False

    peer_ix: Optional[int] = None
    if conflict_tid is not None:
        for idx2, (_op2, otid2, _) in enumerate(all_paths_info):
            if otid2 == conflict_tid:
                peer_ix = idx2
                break

    def _legacy_apply_peer_hold() -> None:
        """向对方路径插入一次时域避让（空中悬停优先）并重算抽样（无额外 P1 全量遍历）。"""
        nonlocal hold_other
        if peer_ix is None or conflict_tid is None:
            return
        ptid = int(conflict_tid)
        nd = trajectory_dict[ptid]
        nidx = int(nd.get('uav_idx', 0))
        n_drone = drones[nidx] if 0 <= nidx < len(drones) else None
        o_start_ts = float(
            trajectory_dict[ptid].get('dronetime_ts', 0) or 0)
        opath_now, otid_chk, odl_keep = all_paths_info[peer_ix]
        raw_hold = _insert_timedomain_hold_waypoint(opath_now, n_drone)
        new_opath = _attach_timestamps(raw_hold, o_start_ts)
        all_paths_info[peer_ix] = (new_opath, otid_chk, odl_keep)
        all_paths[peer_ix] = new_opath
        all_paths_states[peer_ix] = det.build_sampled_states_geo(
            new_opath, scene_lat_ref)
        trajectory_dict[ptid]['path'] = new_opath
        trajectory_dict[ptid]['uav_conflict'] = True
        om = trajectory_dict[ptid].get('resolution_method')
        trajectory_dict[ptid]['resolution_method'] = (
            'alt_layer+speed_hold' if om == 'alt_layer' else 'speed_hold')
        n_dl = float(nd.get('deadline_ts') or 0)
        nd['uav_resolved'] = _uav_avoidance_meets_deadline(
            new_opath, o_start_ts, n_dl, n_drone)
        hold_other = True

    ok_self_hold_feasible = (
        _deadline_feasible_with_one_ground_hold(
            path_work, start_ts, cur_deadline_ts, drone)
        if conflict_slack is not None
        else True
    )
    ok_peer_hold_feasible = False
    if conflict_tid is not None and peer_ix is not None:
        ndp = trajectory_dict[conflict_tid]
        pidx = int(ndp.get('uav_idx', 0))
        d_peer = drones[pidx] if 0 <= pidx < len(drones) else None
        pst = float(ndp.get('dronetime_ts', 0) or 0)
        p_dl = float(ndp.get('deadline_ts') or 0)
        ok_peer_hold_feasible = _deadline_feasible_with_one_ground_hold(
            all_paths_info[peer_ix][0], pst, p_dl, d_peer,
        )

    if conflict_tid is None:
        hold_self = True
    elif peer_ix is None:
        hold_self = True
    elif cur_slack > conflict_slack:
        # 本方 slack 大于「冲突他机中裕量最小者」→ 时间更充裕的本方空中悬停；不可行则让对方悬停或仍由本方承担
        if ok_self_hold_feasible:
            hold_self = True
        elif ok_peer_hold_feasible:
            _legacy_apply_peer_hold()
        else:
            hold_self = True
    else:
        # 对方更紧：优先让对方等；若对方加等待势必超时则改本方等
        if ok_peer_hold_feasible:
            _legacy_apply_peer_hold()
        elif ok_self_hold_feasible:
            hold_self = True
        else:
            _legacy_apply_peer_hold()

    if hold_self:
        path_work = _insert_timedomain_hold_waypoint(path_work, drone)

    path_work = _attach_timestamps(path_work, start_ts)

    if hold_self:
        resolution_method = 'alt_layer+speed_hold' if applied_alt else 'speed_hold'
    elif hold_other:
        resolution_method = 'alt_layer' if applied_alt else 'peer_hold'
    else:
        resolution_method = 'alt_layer' if applied_alt else 'speed_hold'

    return path_work, applied_alt, applied_timedomain, resolution_method, alt_offset_m


# ── RL 推理辅助（DDPG 优先，失败回退规则）────────────────────────────────────

_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DDPG_ACTOR_PATH = os.path.join(_BASE_DIR, 'ddpg_actor.pth')

_ddpg_agent_cache = None


def _load_ddpg_agent():
    """加载已训练的 DDPGAgent（惰性单例）。若模型文件不存在返回 None。"""
    global _ddpg_agent_cache
    if _ddpg_agent_cache is not None:
        return _ddpg_agent_cache
    if not os.path.exists(DDPG_ACTOR_PATH):
        return None
    try:
        from ddpg_train import DDPGAgent
        agent = DDPGAgent()
        agent.load(DDPG_ACTOR_PATH)
        _ddpg_agent_cache = agent
        print(f"  [DDPG] 已加载模型: {DDPG_ACTOR_PATH}")
        return agent
    except Exception as e:
        print(f"  [DDPG] 加载失败({e})，回退规则避障")
        return None


_RL_MAX_STEPS_INFER = 200   # 推理最大步数（150m×200=30km，覆盖绝大多数任务）
_RL_OTHER_PATHS_MAX = 4    # 推理时传入环境的 other_paths 数量上限（与 ddpg_train 对齐）


def _make_rl_path_env(buildings, lat_ref: float,
                      drone: 'Drone' = None,
                      wind_lookup: 'WindTimeseries' = None):
    """
    创建微观路径 RL 环境实例（类名为 SACPathEnv，与 ddpg_train / sac_train 共用）。
    含 KD-Tree；在主循环外调用一次，后续每条任务只需 set_task / set_other_paths。

    drone / wind_lookup 用于 P4 电量奖励及任务时刻时变风（与训练端对齐）。
    """
    from sac_path_env import SACPathEnv
    return SACPathEnv(
        buildings   = buildings,
        lat_ref     = lat_ref,
        drone       = drone,
        wind_lookup = wind_lookup,
    )


def _sample_nearby_paths(
    all_paths: List,
    dt: 'DeliveryTask',
    max_n: int = _RL_OTHER_PATHS_MAX,
) -> List:
    """
    从 all_paths 中挑选与当前任务路线空间最近的 max_n 条路径注入 RL 环境。

    观测里与他机相关的紧凑特征（最近机的 dist/cos/sin），传 4 条和传 699 条
    对模型看到的状态几乎相同——都只报告"最近那一架"。
    关键是传入地理上真正相邻的路径，而非随机采样到南京另一端的路径。
    """
    if not all_paths:
        return []
    if len(all_paths) <= max_n:
        return all_paths
    ref_lon, ref_lat = dt.sender_lon, dt.sender_lat
    def _path_dist(p):
        if not p:
            return float('inf')
        wp = p[0]
        dx = (wp['lon'] - ref_lon) * 111000.0 * np.cos(np.radians(ref_lat))
        dy = (wp['lat'] - ref_lat) * 111000.0
        return np.hypot(dx, dy)
    return sorted(all_paths, key=_path_dist)[:max_n]


def _ddpg_plan_path(
    agent,
    env,                  # 复用的 SACPathEnv 实例（KD-Tree 已构建）
    dt: 'DeliveryTask',
    other_paths: List,
    alns_t: 'AlnsTask' = None,
) -> Optional[List[Dict]]:
    """
    用 DDPG Actor（noise_scale=0 确定性）生成微观路径。
    成功到达目标 → 返回 path；超时未到达 → 返回 None（触发规则回退）。
    步数上限 200（150m×200=30km），other_paths 截断到 4 条与训练分布对齐。

    alns_t 的 dronetime 会被附加到 DeliveryTask，使 env 在 reset 时按
    任务时刻查时变风（与训练端完全对齐）。
    """
    if alns_t is not None:
        dt.dronetime = alns_t.dronetime   # env.reset() 据此查时变风
        dt.order_cnt = alns_t.order_cnt   # env.reset() 据此设 drone.payload (kg)
    sampled = _sample_nearby_paths(other_paths, dt)
    env.set_other_paths(sampled)
    env.set_task(dt)
    state = env.reset()
    for _ in range(_RL_MAX_STEPS_INFER):
        action = agent.select_action(state, noise_scale=0.0)
        state, _, done, info = env.step(action)
        if done:
            return env.get_path() if info.get('success', False) else None
    return None


def plan_micro_trajectories(
    solution: List[List[AlnsTask]],
    alns_tasks: List[AlnsTask],
    buildings_csv: str,
    wind_speed_ms: float,
    wind_dir_deg: float,
    num_uavs: int,
    wind_lookup: 'WindTimeseries' = None,
) -> Tuple[List[Drone], Dict[int, Dict]]:
    """
    对 ALNS 分配结果中的每笔订单生成微观航迹，并主动解决机间冲突。

    路径生成策略（按优先级）：
      1. DDPG Actor（若 ddpg_actor.pth 存在）—— 奖励驱动路径优化
      2. DronePathPlanner 规则避障（回退）

    机间冲突检测：conflict——2.UAVCollisionDetector（时间重叠内同步采样，
    水平间隔 R1+R2、垂直容差 COLLISION2_VERT_TOL_M、未来 T 秒航段相交）。

    机间冲突消解：`_resolve_uav_conflicts_for_trajectory`。
      · ① **先抬高**：多轮合规层抬升，每层后重打 `_t`；
      · ② 用活数据再判 P1，无冲突则结束；
      · ③ **仍冲突**则 **订单时间更充裕**的一方（slack = deadline_ts − dronetime_ts 更大）
        **空中悬停**错峰；另一方可先飞。（对方已入库时回溯其 path 并重打 `_t`。）

    trajectory_dict[task_id] = {
        'uav_idx'            : int,
        'path'               : List[Dict],   # 最终路径（含调整）
        'distance_m'         : float,
        'obstacles'          : int,          # 路径上建筑障碍数
        'adjusted'           : bool,         # 是否执行了建筑避障
        'r_obstacle'         : float,        # P2 建筑冲突奖励
        'uav_conflict'       : bool,         # 初始规划是否存在机间冲突
        'uav_resolved'       : bool,         # 已避让且预计末端到达不晚于 deadline_ts
        'resolution_method'  : str|None,     # alt_layer|speed_hold|alt_layer+speed_hold|peer_hold
        'alt_offset_m'       : float,        # 高度偏移量（仅 alt_layer 时有意义）
        'energy'             : Dict,
        'reward'             : float,
    }
    """
    print("\n" + "="*60)
    print("▶ 微观航迹规划（DDPG 优先 + DronePathPlanner 回退 + 机间冲突解决）")
    print("="*60)
    if wind_lookup is not None and len(wind_lookup) > 0:
        print(f"  风场模式         : 时变（按任务 dronetime 查询，样本 {len(wind_lookup)} 条）")
    else:
        print(f"  风场模式         : 均值 {wind_speed_ms:.2f} m/s / {wind_dir_deg:.1f}°")

    from drone_obstacle_avoidance import (
        path_obstacle_reward as _p2,
    )

    buildings_df = pd.read_csv(buildings_csv)
    planner  = DronePathPlanner(buildings_df)
    drones   = build_micro_drones(num_uavs)
    # RL 模型加载：DDPG（ddpg_actor.pth），不存在则纯规则
    ddpg_agent = _load_ddpg_agent()

    scene_lat_ref = float(np.mean([           # 全域参考纬度（P1/抽样缓存统一用此，避免每任务换一次投影）
        t.sender[1] for ts in solution for t in ts
    ])) if any(solution) else 32.0

    # SACPathEnv 复用实例（DDPG 训练共用）：KD-Tree 只构建一次，700 条任务共享
    # drone 会在主循环里按当前 UAV 实时 set_drone() 切换（机型/电池容量对齐观测）
    _init_drone = drones[0] if drones else None
    rl_env = _make_rl_path_env(planner.buildings, scene_lat_ref,
                               drone       = _init_drone,
                               wind_lookup = wind_lookup) \
             if ddpg_agent is not None else None
    ddpg_cnt = 0  # DDPG 成功生成路径的任务数
    rule_cnt = 0  # 回退规则避障的任务数

    trajectory_dict: Dict[int, Dict] = {}
    # all_paths_info: [(path, task_id, deadline_ts), ...]
    # 用于冲突解决时按截止时间决定谁悬停
    all_paths_info: List = []
    all_paths:      List = []   # 纯路径列表（与 all_paths_info 同步，供 RL 推理注入他机）
    # 与 all_paths 同长度的 build_sampled_states_geo 结果，避免每条任务对所有他机路径重复抽样
    all_paths_states: List = []

    total_tasks = sum(len(t) for t in solution)
    done = 0
    uav_conflict_cnt   = 0
    uav_resolved_cnt   = 0
    speed_hold_cnt     = 0
    _c2_det = _get_uav_collision2_detector()

    for uav_idx, uav_tasks in enumerate(solution):
        drone = drones[uav_idx]
        sorted_t = sorted(uav_tasks, key=lambda t: t.dronetime_ts)
        #  closure 仅需按 UAV 绑一次机型（巡航速度）；勿在单笔订单循环内重建
        _p1_drone = _make_p1_conflict2_closure(drone, _c2_det)
        for alns_t in sorted_t:
            # 动态设 payload = 订单数 (kg)，影响主程序 estimate_path_energy_wh
            drone.current_payload_kg = float(alns_t.order_cnt)
            # DDPG 推理：让 env 直接使用当前 UAV 的真实 drone（机型/电池容量对齐观测）
            if rl_env is not None:
                rl_env.set_drone(drone)

            dt = DeliveryTask(
                task_id      = alns_t.task_id,
                sender_lon   = alns_t.sender[0],
                sender_lat   = alns_t.sender[1],
                receiver_lon = alns_t.receiver[0],
                receiver_lat = alns_t.receiver[1],
            )
            dt.order_cnt = alns_t.order_cnt   # env.reset() 会自己再设一次，确保万无一失

            # ── 路径生成：DDPG 优先，失败回退规则避障 ────────────────────────────
            rl_path = None
            if ddpg_agent is not None and rl_env is not None:
                rl_path = _ddpg_plan_path(
                    ddpg_agent, rl_env, dt, all_paths,
                    alns_t = alns_t,   # 传入以便 env 查任务时刻的时变风
                )

            if rl_path is not None:
                path          = rl_path
                path_obs_list = planner.obstacles_on_straight_route(dt)
                n_build_obs   = len(path_obs_list)
                adjusted      = n_build_obs > 0
                ddpg_cnt     += 1
            else:
                result        = planner.plan_path(dt, verbose=False)
                path          = result['path']
                path_obs_list = result.get('obstacles_on_path_list', [])
                n_build_obs   = int(result['obstacles_on_path'])
                adjusted      = n_build_obs > 0
                rule_cnt     += 1

            # 所有巡航航点高度严格对齐到 60/90/120m 合规层（空域分层的前提）
            # RL Actor 输出的连续高度、规则避障的 "h+20m" 经此均落到 {60,90,120}
            path = _snap_path_to_layer(path)

            # 为每个航点附加估算到达时间戳（对应 MultirotorUAV_4.py 的时间预判维度）
            # 以 dronetime_ts 为起飞时刻，按巡航速度累加飞行时间
            start_ts = float(getattr(alns_t, 'dronetime_ts', 0) or 0)
            path = _attach_timestamps(path, start_ts)

            # ── P2 建筑障碍检验 ───────────────────────────────────────────
            r_obstacle = _p2(
                path, path_obs_list, dt,
                safe_r_m=SAFETY_MARGIN, lat_ref=alns_t.sender[1],
            )

            # ── P1 机间冲突检测 + 高度分层主动解决（conflict——2）───────────────
            lat_ref_task = float(alns_t.sender[1])
            other_paths_now = list(all_paths)   # 已确定的路径（均含 _t 时间戳）
            other_states_aligned = (
                all_paths_states
                if len(all_paths_states) == len(other_paths_now)
                else None
            )
            ss_plan = _c2_det.build_sampled_states_geo(path, scene_lat_ref)
            r_uav = _p1_drone(
                path, other_paths_now,
                safe_interval_m=UAV_SAFE_INTERVAL_M, lat_ref=scene_lat_ref,
                time_window_s=TIME_CONFLICT_WINDOW_S,
                other_states_cached=other_states_aligned,
                states_self_prebuilt=ss_plan,
            )

            uav_conflict      = (r_uav < 0)
            resolution_method = None
            alt_offset        = 0.0

            if uav_conflict:
                uav_conflict_cnt += 1
                path, applied_alt, applied_td, resolution_method, alt_offset = (
                    _resolve_uav_conflicts_for_trajectory(
                        path,
                        start_ts,
                        other_paths_now,
                        all_paths,
                        all_paths_states,
                        all_paths_info,
                        trajectory_dict,
                        alns_t,
                        scene_lat_ref,
                        _c2_det,
                        drone,
                        drones,
                    ))
                dl_for_resolve = float(
                    getattr(alns_t, 'deadline_ts',
                            getattr(alns_t, 'dronetime_ts', 0)) or 0)
                uav_resolved, ok_alt, ok_td = _split_uav_resolve_counts_for_deadline(
                    path, start_ts, dl_for_resolve, drone,
                    applied_alt, applied_td,
                )
                if ok_alt:
                    uav_resolved_cnt += 1
                if ok_td:
                    speed_hold_cnt += 1
            else:
                uav_resolved = False

            ss_to_register = (
                ss_plan if not uav_conflict
                else _c2_det.build_sampled_states_geo(path, scene_lat_ref)
            )

            # ── 确定本任务的风（时变优先，否则用标量均值）───────────────────
            if wind_lookup is not None and len(wind_lookup) > 0:
                ws_task, wd_task = wind_lookup.at(alns_t.dronetime)
            else:
                ws_task, wd_task = wind_speed_ms, wind_dir_deg

            # ── 耗电估算（含风，使用最终路径） ───────────────────────────────
            use_wind = ws_task > 0
            energy = drone.estimate_path_energy_wh(
                path,
                wind_speed_m_s     = ws_task if use_wind else None,
                wind_direction_deg = wd_task if use_wind else None,
            )

            # ── 避撞额外耗电 = 最终路径能耗 - 直飞基线能耗 ──────────────────
            # 仅对发生了建筑避障或机间冲突调整的任务计算
            had_avoidance = adjusted or uav_conflict
            if had_avoidance:
                baseline_energy = drone.estimate_path_energy_wh(
                    _baseline_path(dt),
                    wind_speed_m_s     = ws_task if use_wind else None,
                    wind_direction_deg = wd_task if use_wind else None,
                )
                energy_extra_wh = max(
                    0.0,
                    energy['total_wh'] - baseline_energy['total_wh']
                )
            else:
                energy_extra_wh = 0.0

            # ── 路径总奖励（使用最终路径） ────────────────────────────────────
            reward = path_total_reward(
                path, dt, drone,
                obstacles          = planner.buildings,
                other_paths        = other_paths_now,
                wind_speed_m_s     = ws_task if use_wind else None,
                wind_direction_deg = wd_task if use_wind else None,
                lat_ref            = lat_ref_task,
            )

            trajectory_dict[alns_t.task_id] = {
                'uav_idx':           uav_idx,
                'path':              path,
                'distance_m':        result['total_distance'] if rl_path is None else
                                     sum(
                                         float(np.hypot(
                                             (path[i+1]['lon']-path[i]['lon'])*111000*np.cos(np.radians(alns_t.sender[1])),
                                             (path[i+1]['lat']-path[i]['lat'])*111000
                                         ))
                                         for i in range(len(path)-1)
                                     ),
                'obstacles':         n_build_obs,
                'adjusted':          adjusted,
                'r_obstacle':        r_obstacle,
                'uav_conflict':      uav_conflict,
                'uav_resolved':      uav_resolved,
                'resolution_method': resolution_method,
                'alt_offset_m':      alt_offset,
                'energy':            energy,
                'energy_extra_wh':   energy_extra_wh,   # 避撞额外耗电（Wh）
                'reward':            reward,
                'wind_ms':           ws_task,           # 任务时刻风速 m/s
                'wind_deg':          wd_task,           # 任务时刻风向（吹向，度）
                # ── 成本类指标原始数据 ──────────────────────────────
                'flight_time_s':     _path_flight_time_s(
                                         path,
                                         default_speed_ms=getattr(drone, 'cruise_speed', 15.0),
                                     ),
                'straight_dist_m':   haversine(alns_t.sender, alns_t.receiver) * 1000.0,
                'dronetime_ts':      float(getattr(alns_t, 'dronetime_ts', 0) or 0),
                'deadline_ts':       float(getattr(alns_t, 'deadline_ts',
                                                   getattr(alns_t, 'dronetime_ts', 0)) or 0),
            }
            deadline_ts = float(getattr(alns_t, 'deadline_ts',
                                        getattr(alns_t, 'dronetime_ts', 0)) or 0)
            all_paths.append(path)
            all_paths_states.append(ss_to_register)
            all_paths_info.append((path, alns_t.task_id, deadline_ts))
            done += 1
            if done % 50 == 0 or done == total_tasks:
                print(f"  进度 {done}/{total_tasks}"
                      f"  机间冲突 {uav_conflict_cnt}"
                      f"  高度分层 {uav_resolved_cnt}"
                      f"  降速悬停 {speed_hold_cnt}")

    unresolved_final = uav_conflict_cnt - uav_resolved_cnt - speed_hold_cnt
    print(f"\n  全部 {total_tasks} 条航迹规划完成。")
    if ddpg_agent is not None:
        print(f"  路径来源：DDPG {ddpg_cnt} 条  |  规则回退 {rule_cnt} 条")
    print(f"  机间冲突：检测 {uav_conflict_cnt} 次"
          f"  → 高度分层解决 {uav_resolved_cnt}"
          f"  + 降速悬停解决 {speed_hold_cnt}"
          f"  + 未解决 {unresolved_final}")
    return drones, trajectory_dict


# ══════════════════════════════════════════════════════════════════════════════
# §9b  冲突指标计算
# ══════════════════════════════════════════════════════════════════════════════

def compute_conflict_metrics(
    trajectory_dict: Dict[int, Dict],
    safe_dist_m: float = 50.0,
) -> Dict:
    """
    计算冲突指标（建筑障碍与机间冲突分开统计，口径一致）。

    ┌───────────────────────────────────────────────────────────────┐
    │ 建筑障碍冲突（以"任务"为单位）                                  │
    │  · 检测到 ：obstacles_on_path > 0 的任务（路径与建筑相交）      │
    │  · 解决   ：执行了飞越/绕行 且 避障后 r_obstacle==0            │
    │  · 未解决 ：执行了飞越/绕行 但 r_obstacle<0（绕行后仍贴近建筑）  │
    │                                                               │
    │ 机间冲突                                                          │
    │  · plan_micro_trajectories 中：若存在机间冲突且经抬升/时域处置       │
    │    且航迹预估末端到达时刻 ≤ 任务 deadline_ts，则 uav_resolved=True │
    │  · compute_conflict_metrics 中「机间对」抽样仍可按 safe_dist_m     │
    │    复检几何接近（与本字段独立）                                    │
    │                                                                   │
    │ 综合成功率 = 建筑障碍解决任务数 / 建筑障碍检测任务数             │
    │             （仅统计建筑障碍，机间冲突单列）                     │
    └───────────────────────────────────────────────────────────────┘
    """
    task_ids = list(trajectory_dict.keys())
    n = len(task_ids)

    # ── 1. 建筑障碍冲突（以任务为单位，口径一致）────────────────────────────
    obs_tasks_detected   = 0   # 路径上有建筑障碍的任务数
    obs_tasks_resolved   = 0   # 避障后无残余冲突的任务数（r_obstacle == 0）
    obs_tasks_unresolved = 0   # 避障后仍有残余冲突的任务数（r_obstacle < 0）
    adjusted_traj        = 0   # 执行了飞越/绕行的任务数（同 obs_tasks_detected）

    for tid in task_ids:
        info = trajectory_dict[tid]
        if info['obstacles'] > 0:      # 路径上检测到建筑障碍
            obs_tasks_detected += 1
            adjusted_traj      += 1
            if info['r_obstacle'] < 0:
                obs_tasks_unresolved += 1   # 避障动作仍不足以清除冲突
            else:
                obs_tasks_resolved   += 1   # 避障成功

    # 建筑障碍成功率
    obs_rate = (obs_tasks_resolved / obs_tasks_detected * 100.0
                if obs_tasks_detected > 0 else 100.0)

    # ── 2. 机间冲突（直接读取 plan_micro_trajectories 写入的字段）────────────
    uav_detected    = sum(1 for tid in task_ids if trajectory_dict[tid].get('uav_conflict', False))
    uav_alt_layer   = sum(1 for tid in task_ids
                          if trajectory_dict[tid].get('resolution_method') in
                          ('alt_layer', 'alt_layer+speed_hold'))
    uav_speed_hold  = sum(1 for tid in task_ids
                          if trajectory_dict[tid].get('resolution_method') in
                          ('speed_hold', 'alt_layer+speed_hold'))
    uav_peer_hold = sum(
        1 for tid in task_ids
        if trajectory_dict[tid].get('resolution_method') == 'peer_hold')
    uav_resolved    = sum(1 for tid in task_ids if trajectory_dict[tid].get('uav_resolved', False))
    uav_unresolved  = uav_detected - uav_resolved
    uav_rate = uav_resolved / uav_detected * 100.0 if uav_detected > 0 else 100.0

    # ── 3. 汇总 ────────────────────────────────────────────────────────────────
    no_adjustment = sum(
        1 for tid in task_ids
        if not trajectory_dict[tid].get('adjusted', False)
        and not trajectory_dict[tid].get('uav_conflict', False)
    )

    total_detected   = obs_tasks_detected  + uav_detected
    total_resolved   = obs_tasks_resolved  + uav_resolved
    total_unresolved = obs_tasks_unresolved + uav_unresolved
    overall_rate = (total_resolved / total_detected * 100.0
                    if total_detected > 0 else 100.0)

    # ── 4. 能耗类指标（单次避撞额外消耗）────────────────────────────────────
    # 收集所有发生了建筑避障或机间冲突调整任务的额外能耗
    avoidance_extras = [
        trajectory_dict[tid].get('energy_extra_wh', 0.0)
        for tid in task_ids
        if trajectory_dict[tid].get('adjusted', False)
        or trajectory_dict[tid].get('uav_conflict', False)
    ]
    if avoidance_extras:
        max_avoidance_wh = float(max(avoidance_extras))
        avg_avoidance_wh = float(sum(avoidance_extras) / len(avoidance_extras))
    else:
        max_avoidance_wh = 0.0
        avg_avoidance_wh = 0.0

    # 分类统计：仅建筑避障、仅机间冲突
    obs_only_extras = [
        trajectory_dict[tid].get('energy_extra_wh', 0.0)
        for tid in task_ids
        if trajectory_dict[tid].get('adjusted', False)
        and not trajectory_dict[tid].get('uav_conflict', False)
    ]
    uav_only_extras = [
        trajectory_dict[tid].get('energy_extra_wh', 0.0)
        for tid in task_ids
        if trajectory_dict[tid].get('uav_conflict', False)
        and not trajectory_dict[tid].get('adjusted', False)
    ]
    obs_avg_extra_wh = float(sum(obs_only_extras) / len(obs_only_extras)) if obs_only_extras else 0.0
    uav_avg_extra_wh = float(sum(uav_only_extras) / len(uav_only_extras)) if uav_only_extras else 0.0

    # ── 5. 最终路径最低巡航高度层分布（每项任务取全体 height>0 航点中的 min，再归入 60/90/120）
    dist_min_layer = {'60': 0, '90': 0, '120': 0, 'other': 0, 'no_cruise': 0}
    for tid in task_ids:
        b = _min_cruise_layer_bucket(trajectory_dict[tid].get('path') or [])
        dist_min_layer[b] += 1

    return {
        # ── 调整轨迹 ──
        'adjusted_trajectories':       adjusted_traj,
        'total_trajectories':          n,
        'adjusted_ratio_pct':          adjusted_traj / n * 100 if n > 0 else 0.0,
        'no_adjustment_tasks':         no_adjustment,
        # ── 建筑障碍冲突（任务维度）──
        'obs_tasks_detected':          obs_tasks_detected,
        'obs_tasks_resolved':          obs_tasks_resolved,
        'obs_tasks_unresolved':        obs_tasks_unresolved,
        'obs_resolution_rate_pct':     obs_rate,
        # ── 机间冲突（高度分层 + 降速悬停双重解决）──
        'uav_detected':                uav_detected,
        'uav_alt_layer':               uav_alt_layer,    # 含 alt_layer+speed_hold
        'uav_speed_hold':              uav_speed_hold,   # 航迹上真有等待点：含 alt_layer+speed_hold
        'uav_peer_hold':               uav_peer_hold,   # 由对方承担等待，本机航迹无等待点
        'uav_alt_and_hold':            sum(1 for tid in task_ids
                                           if trajectory_dict[tid].get('resolution_method')
                                           == 'alt_layer+speed_hold'),
        'uav_resolved':                uav_resolved,
        'uav_unresolved':              uav_unresolved,
        'uav_resolution_rate_pct':     uav_rate,
        # ── 总计 ──
        'total_detected':              total_detected,
        'total_resolved':              total_resolved,
        'total_unresolved':            total_unresolved,
        'conflict_resolution_rate_pct': overall_rate,
        # ── 能耗类（避撞额外消耗）──
        'avoidance_count':             len(avoidance_extras),
        'max_avoidance_energy_wh':     max_avoidance_wh,
        'avg_avoidance_energy_wh':     avg_avoidance_wh,
        'obs_avg_extra_energy_wh':     obs_avg_extra_wh,
        'uav_avg_extra_energy_wh':     uav_avg_extra_wh,
        # ── 最低巡航层 ── （见 print_conflict_metrics）
        'min_cruise_layer_distribution': dict(dist_min_layer),
    }


def compute_cost_metrics(
    trajectory_dict: Dict[int, Dict],
    alns_speed_ms: float = 75.0 / 3.6,   # ALNS 宏观巡航速度，m/s（默认 75 km/h）
    smooth_coeff1: float = 1.0,           # 转弯角系数
    smooth_coeff2: float = 0.5,           # 爬升角差系数
) -> Dict:
    """
    计算四项成本类指标：

    ① 轨迹成本（轨迹长度）
       - 总轨迹长度：所有任务微观路径长度之和（m）
       - 平均轨迹长度：总长度 / 任务数（m）

    ② 轨迹平滑成本
       = Σ转弯角(°) × coeff1 + Σ|连续爬升角差|(°) × coeff2
       - 报告所有任务的均值（每条航迹的平滑成本均值）

    ③ 平均延迟时间
       延迟定义：实际飞行时间 - ALNS 基准飞行时间（直线距离 / ALNS 速度）
       仅统计延迟 > 0 的任务。报告：所有任务均值 & 仅延误任务均值。

    ④ 延误无人机数量
       至少有一个任务发生延迟的 UAV 数量。
    """
    task_ids = list(trajectory_dict.keys())
    n = len(task_ids)
    if n == 0:
        return {}

    # ── ① 轨迹长度 ──────────────────────────────────────────────────────────
    lengths_m = [trajectory_dict[tid]['distance_m'] for tid in task_ids]
    total_length_m   = float(sum(lengths_m))
    avg_length_m     = total_length_m / n

    # ── ② 轨迹平滑成本 ──────────────────────────────────────────────────────
    smooth_costs = [
        _path_smooth_cost(trajectory_dict[tid]['path'], smooth_coeff1, smooth_coeff2)
        for tid in task_ids
    ]
    avg_smooth_cost  = float(sum(smooth_costs) / n)
    total_smooth_cost = float(sum(smooth_costs))

    # ── ③ 延迟时间（实际飞行 vs ALNS 基准直线）──────────────────────────────
    delays_s: List[float] = []
    for tid in task_ids:
        info = trajectory_dict[tid]
        straight_m    = info.get('straight_dist_m', 0.0)
        alns_base_s   = straight_m / alns_speed_ms if alns_speed_ms > 0 else 0.0
        actual_s      = info.get('flight_time_s', 0.0)
        delay_s       = max(0.0, actual_s - alns_base_s)
        delays_s.append(delay_s)

    delayed_mask     = [d > 0 for d in delays_s]
    delayed_count    = sum(delayed_mask)
    avg_delay_all_s  = float(sum(delays_s) / n)
    avg_delay_late_s = (float(sum(d for d in delays_s if d > 0) / delayed_count)
                        if delayed_count > 0 else 0.0)

    # ── ④ 延误无人机数量 ─────────────────────────────────────────────────────
    delayed_uav_set: set = set()
    for tid, delay_s in zip(task_ids, delays_s):
        if delay_s > 0:
            delayed_uav_set.add(trajectory_dict[tid]['uav_idx'])
    delayed_uav_count = len(delayed_uav_set)

    return {
        # ① 轨迹长度
        'total_traj_length_m':    total_length_m,
        'avg_traj_length_m':      avg_length_m,
        # ② 平滑成本
        'smooth_coeff1':          smooth_coeff1,
        'smooth_coeff2':          smooth_coeff2,
        'total_smooth_cost':      total_smooth_cost,
        'avg_smooth_cost':        avg_smooth_cost,
        # ③ 延迟时间
        'delayed_task_count':     delayed_count,
        'avg_delay_all_s':        avg_delay_all_s,
        'avg_delay_late_s':       avg_delay_late_s,
        # ④ 延误无人机
        'delayed_uav_count':      delayed_uav_count,
    }


def print_cost_metrics(cost_metrics: Dict):
    """输出成本类指标（四项）。"""
    print("\n" + "="*65)
    print("  三、成本类指标")
    print("="*65)
    # ① 轨迹长度
    print(f"  ① 轨迹长度成本")
    print(f"     总轨迹长度   : {cost_metrics['total_traj_length_m']/1000:.2f} km")
    print(f"     平均轨迹长度 : {cost_metrics['avg_traj_length_m']:.1f} m")
    # ② 平滑成本
    c1, c2 = cost_metrics['smooth_coeff1'], cost_metrics['smooth_coeff2']
    print(f"  ② 轨迹平滑成本  (coeff1={c1}, coeff2={c2})")
    print(f"     平均平滑成本 : {cost_metrics['avg_smooth_cost']:.2f}")
    print(f"     总平滑成本   : {cost_metrics['total_smooth_cost']:.1f}")
    # ③ 延迟时间
    print(f"  ③ 平均延迟时间")
    print(f"     延误任务数   : {cost_metrics['delayed_task_count']} 条")
    print(f"     全任务均值   : {cost_metrics['avg_delay_all_s']:.1f} s")
    print(f"     仅延误任务均值: {cost_metrics['avg_delay_late_s']:.1f} s")
    # ④ 延误无人机
    print(f"  ④ 延误无人机数量 : {cost_metrics['delayed_uav_count']} 架")
    print("="*65)


def print_conflict_metrics(metrics: Dict):
    """输出冲突类指标（三项）+ 能耗类指标（两项）。"""
    adj        = metrics['adjusted_trajectories']
    uav_d      = metrics['uav_detected']
    alt_layer  = metrics['uav_alt_layer']
    speed_hold = metrics['uav_speed_hold']
    total_adjusted = adj + uav_d

    print("\n" + "="*65)
    print("  一、冲突类指标")
    print("="*65)
    print(f"  调整轨迹数量      : {total_adjusted} 条")
    print(f"    └ 建筑避障飞越/绕行 : {adj}")
    uav_both = metrics.get('uav_alt_and_hold', 0)
    peer_n = metrics.get('uav_peer_hold', 0)
    print(f"    └ 机间高度分层      : {alt_layer}  |  降速悬停避让 : {speed_hold}"
          + (f"  |  两者兼用 : {uav_both}" if uav_both > 0 else "")
          + (f"  |  对方等待(本机无等待点) : {peer_n}" if peer_n > 0 else ""))
    mcd = metrics.get('min_cruise_layer_distribution') or {}
    print("  （最终路径）最低巡航层分布 · 任务数 ："
          f" 60m {mcd.get('60', 0)}  | 90m {mcd.get('90', 0)}  |"
          f" 120m {mcd.get('120', 0)}  | 其它 {mcd.get('other', 0)}  |"
          f" 无巡航点 {mcd.get('no_cruise', 0)}")
    print(f"  未解决冲突数量    : {metrics['total_unresolved']} 次"
          f"  （建筑残余 {metrics['obs_tasks_unresolved']}"
          f" + 机间未解决 {metrics['uav_unresolved']}）")
    print(f"  冲突解决成功率    : {metrics['conflict_resolution_rate_pct']:.1f}%"
          f"  （解决 {metrics['total_resolved']} / 检测 {metrics['total_detected']}）")

    print("\n" + "="*65)
    print("  二、能耗类指标（避撞额外消耗 = 调整路径 - 直飞基线）")
    print("="*65)
    cnt = metrics['avoidance_count']
    print(f"  参与统计的避撞任务数 : {cnt} 条")
    print(f"  单次避撞最大消耗电量 : {metrics['max_avoidance_energy_wh']:.3f} Wh")
    print(f"  单次避撞平均消耗电量 : {metrics['avg_avoidance_energy_wh']:.3f} Wh")
    print(f"    └ 建筑避障任务均值  : {metrics['obs_avg_extra_energy_wh']:.3f} Wh")
    print(f"    └ 机间冲突任务均值  : {metrics['uav_avg_extra_energy_wh']:.3f} Wh")
    print("="*65)


def export_conflict_metrics_csv(
    metrics: Dict,
    cost_metrics: Dict = None,
    output_path: str = 'conflict_metrics.csv',
):
    """将冲突指标 + 成本指标保存为单行 CSV。"""
    # 控制台"调整轨迹数量"= 建筑避障 + 机间冲突，与此处对齐
    total_adjusted_csv = metrics['adjusted_trajectories'] + metrics['uav_detected']
    row = {
        # ── 冲突类 ────────────────────────────────────────────────────────────
        '总任务数':                    metrics['total_trajectories'],
        '调整轨迹数量':                 total_adjusted_csv,           # 建筑避障+机间冲突
        '建筑避障调整数':               metrics['adjusted_trajectories'],
        '机间冲突调整数':               metrics['uav_detected'],
        '调整轨迹比例_%':               round(total_adjusted_csv / metrics['total_trajectories'] * 100, 2)
                                        if metrics['total_trajectories'] > 0 else 0.0,
        '有障碍任务数（检测）':          metrics['obs_tasks_detected'],
        '建筑障碍成功避障任务数':         metrics['obs_tasks_resolved'],
        '建筑障碍残余冲突任务数':         metrics['obs_tasks_unresolved'],
        '建筑障碍冲突解决成功率_%':       round(metrics['obs_resolution_rate_pct'], 2),
        '机间冲突检测任务数':             metrics['uav_detected'],
        '机间冲突高度分层解决数':         metrics['uav_alt_layer'],     # 含 alt_layer+speed_hold
        '机间冲突降速悬停解决数':         metrics['uav_speed_hold'],    # 本机或对方航迹含等待点
        '机间冲突对方承担等待数':         metrics.get('uav_peer_hold', 0),  # peer_hold：本机航迹无等待点
        '机间冲突两者兼用数':             metrics.get('uav_alt_and_hold', 0),
        '机间冲突总解决数':              metrics['uav_resolved'],
        '机间冲突未解决数':              metrics['uav_unresolved'],
        '机间冲突解决成功率_%':          round(metrics['uav_resolution_rate_pct'], 2),
        '最低巡航层_60m任务数':           (metrics.get('min_cruise_layer_distribution') or {}).get('60', 0),
        '最低巡航层_90m任务数':           (metrics.get('min_cruise_layer_distribution') or {}).get('90', 0),
        '最低巡航层_120m任务数':          (metrics.get('min_cruise_layer_distribution') or {}).get('120', 0),
        '最低巡航层_其它任务数':           (metrics.get('min_cruise_layer_distribution') or {}).get('other', 0),
        '最低巡航层_无巡航点任务数':        (metrics.get('min_cruise_layer_distribution') or {}).get('no_cruise', 0),
        '总检测冲突数':                  metrics['total_detected'],
        '总解决冲突数':                  metrics['total_resolved'],
        '未解决冲突数量':                metrics['total_unresolved'],
        '综合冲突解决成功率_%':          round(metrics['conflict_resolution_rate_pct'], 2),
        # ── 能耗类（避撞额外消耗）──────────────────────────────────────────────
        '参与统计的避撞任务数':           metrics['avoidance_count'],
        '单次避撞最大消耗电量_Wh':        round(metrics['max_avoidance_energy_wh'], 4),
        '单次避撞平均消耗电量_Wh':        round(metrics['avg_avoidance_energy_wh'], 4),
        '建筑避障任务均值_Wh':            round(metrics['obs_avg_extra_energy_wh'], 4),
        '机间冲突任务均值_Wh':            round(metrics['uav_avg_extra_energy_wh'], 4),
    }
    # ── 成本类（可选）──────────────────────────────────────────────────────────
    if cost_metrics:
        row.update({
            '总轨迹长度_km':              round(cost_metrics['total_traj_length_m'] / 1000, 3),
            '平均轨迹长度_m':             round(cost_metrics['avg_traj_length_m'], 1),
            '轨迹平滑成本系数1':           cost_metrics['smooth_coeff1'],
            '轨迹平滑成本系数2':           cost_metrics['smooth_coeff2'],
            '平均轨迹平滑成本':            round(cost_metrics['avg_smooth_cost'], 3),
            '总轨迹平滑成本':              round(cost_metrics['total_smooth_cost'], 2),
            '延误任务数':                  cost_metrics['delayed_task_count'],
            '全任务平均延迟_s':            round(cost_metrics['avg_delay_all_s'], 1),
            '仅延误任务平均延迟_s':         round(cost_metrics['avg_delay_late_s'], 1),
            '延误无人机数量':              cost_metrics['delayed_uav_count'],
        })
    pd.DataFrame([row]).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  冲突+成本指标已保存：{output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# §10  结果输出
# ══════════════════════════════════════════════════════════════════════════════

def export_assignment_csv(
    solution: List[List[AlnsTask]],
    trajectory_dict: Dict[int, Dict],
    output_path: str = 'assignment_results.csv',
):
    rows = []
    for uav_idx, tasks in enumerate(solution):
        for rank, t in enumerate(sorted(tasks, key=lambda x: x.dronetime_ts), 1):
            traj = trajectory_dict.get(t.task_id, {})
            energy = traj.get('energy', {})
            r_obs = traj.get('r_obstacle', 0.0)
            rows.append({
                '无人机编号':       uav_idx + 1,
                '任务序号':         rank,
                '任务ID':           t.task_id,
                '发送点经度':       t.sender[0],
                '发送点纬度':       t.sender[1],
                '接收点经度':       t.receiver[0],
                '接收点纬度':       t.receiver[1],
                '下单时间':         t.otime.strftime('%Y/%m/%d %H:%M'),
                '截止时间':         t.dronetime.strftime('%Y/%m/%d %H:%M'),
                '订单数':           t.order_cnt,
                '微观航程_m':       round(traj.get('distance_m', 0), 1),
                '障碍物数量':       traj.get('obstacles', 0),
                '是否调整轨迹':     '是' if traj.get('adjusted', False) else '否',
                '障碍冲突已解决':   '是' if (traj.get('adjusted', False) and r_obs == 0.0) else
                                    ('否' if (traj.get('adjusted', False) and r_obs < 0.0) else '无需'),
                '障碍奖励':         round(r_obs, 1),
                '机间冲突检测':     '是' if traj.get('uav_conflict', False) else '否',
                '机间冲突解决':     '是' if traj.get('uav_resolved', False) else
                                    ('否' if traj.get('uav_conflict', False) else '无需'),
                '高度分层_m':       traj.get('alt_offset_m', 0.0),
                '估算耗电_Wh':      round(energy.get('total_wh', 0), 2),
                '路径奖励':         round(traj.get('reward', 0), 2),
            })
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  分配结果已保存：{output_path}")


def export_trajectory_csv(
    trajectory_dict: Dict[int, Dict],
    output_path: str = 'trajectory_results.csv',
):
    rows = []
    for task_id, info in trajectory_dict.items():
        for wp_idx, wp in enumerate(info['path']):
            rows.append({
                '任务ID':     task_id,
                '无人机编号': info['uav_idx'] + 1,
                '航点序号':   wp_idx,
                '经度':       round(wp['lon'], 7),
                '纬度':       round(wp['lat'], 7),
                '高度_m':     round(wp['height'], 1),
                '动作描述':   wp.get('action', ''),
            })
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  航迹结果已保存：{output_path}")


def print_assignment_summary(
    solution: List[List[AlnsTask]],
    uavs: List[AlnsUAV],
    trajectory_dict: Dict[int, Dict],
    warehouse: Tuple[float, float],
    stations: List[Tuple[float, float]],
    wind_speed_ms: float,
    wind_dir_deg: float,
    total_dist: float,
):
    print("\n" + "="*70)
    print("▶ ALNS 分配结果汇总")
    print("="*70)
    print(f"  仓库位置         : ({warehouse[0]:.5f}, {warehouse[1]:.5f})")
    print(f"  充电站数量       : {len(stations)}")
    print(f"  风速/风向        : {wind_speed_ms:.2f} m/s  →  {wind_dir_deg:.1f}°(吹向)")
    print(f"  无人机数量       : {len(solution)}")
    print(f"  总任务数         : {sum(len(t) for t in solution)}")
    print(f"  ALNS 等效总飞行距离: {total_dist:.2f} km")
    print()

    total_energy_wh  = 0.0
    total_micro_m    = 0.0
    total_obstacles  = 0
    assigned_cnt     = 0

    for uav_idx, tasks in enumerate(solution):
        n = len(tasks)
        assigned_cnt += n
        task_ids = sorted([t.task_id for t in tasks])
        dist_km  = calc_uav_distance_wind(
            tasks, warehouse, stations, uavs[uav_idx],
            wind_speed_ms, wind_dir_deg
        )
        e_wh = sum(
            trajectory_dict.get(t.task_id, {}).get('energy', {}).get('total_wh', 0)
            for t in tasks
        )
        micro_m = sum(
            trajectory_dict.get(t.task_id, {}).get('distance_m', 0)
            for t in tasks
        )
        obs = sum(
            trajectory_dict.get(t.task_id, {}).get('obstacles', 0)
            for t in tasks
        )
        total_energy_wh += e_wh
        total_micro_m   += micro_m
        total_obstacles += obs

        ids_str = str(task_ids[:5]) + ('...' if len(task_ids) > 5 else '')
        print(f"  UAV-{uav_idx+1:2d} | 任务 {n:3d} 笔 | 宏观 {dist_km:7.2f} km"
              f" | 微观 {micro_m/1000:6.2f} km"
              f" | 耗电 {e_wh:6.1f} Wh"
              f" | 避障 {obs} 次"
              f" | 任务ID(首5): {ids_str}")

    print()
    print(f"  汇总 — 总任务 {assigned_cnt} | 微观总航程 {total_micro_m/1000:.2f} km"
          f" | 总耗电 {total_energy_wh:.1f} Wh"
          f" | 总避障 {total_obstacles} 次")


# ══════════════════════════════════════════════════════════════════════════════
# §11  可视化
# ══════════════════════════════════════════════════════════════════════════════

UAV_COLORS = [
    '#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
    '#42d4f4','#f032e6','#bfef45','#808000','#469990',
    '#dcbeff','#9A6324','#800000','#aaffc3','#000075',
]


def visualize_2d_overview(
    solution: List[List[AlnsTask]],
    warehouse: Tuple[float, float],
    stations: List[Tuple[float, float]],
    wind_speed_ms: float,
    wind_dir_deg: float,
    output_path: str = 'overview_2d.png',
):
    """2D 宏观路径总览图（matplotlib）。"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_title('宏观路径分配总览（ALNS + 风场修正）', fontsize=16, fontweight='bold')
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')

    # 绘制每架 UAV 的路径
    for uav_idx, tasks in enumerate(solution):
        if not tasks:
            continue
        color = UAV_COLORS[uav_idx % len(UAV_COLORS)]
        sorted_t = sorted(tasks, key=lambda t: t.dronetime_ts)
        pts = [warehouse] + [p for t in sorted_t for p in (t.sender, t.receiver)] + [warehouse]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, '-', color=color, alpha=0.5, linewidth=0.8,
                label=f'UAV-{uav_idx+1}({len(tasks)}笔)')
        # 发送点
        for t in sorted_t:
            ax.plot(t.sender[0],   t.sender[1],   's', color=color, ms=3, alpha=0.6)
            ax.plot(t.receiver[0], t.receiver[1], '^', color=color, ms=3, alpha=0.6)

    # 仓库
    ax.plot(warehouse[0], warehouse[1], 'k*', ms=18, label='仓库')
    # 充电站
    for i, s in enumerate(stations):
        ax.plot(s[0], s[1], 'yo', ms=10, markeredgecolor='k', linewidth=0.8,
                label='充电站' if i == 0 else '')

    # 风向箭头（图内右上角）
    if wind_speed_ms > 0:
        wr = math.radians(wind_dir_deg)
        cx = ax.get_xlim()[1] if ax.get_xlim()[1] != 1.0 else (warehouse[0] + 0.05)
        cy = ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else (warehouse[1] + 0.05)
        ax.annotate(
            f'风 {wind_speed_ms:.1f} m/s',
            xy=(0.92, 0.95), xycoords='axes fraction',
            fontsize=10, ha='center',
        )

    ax.legend(loc='lower left', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  2D 总览图已保存：{output_path}")


def visualize_3d_sample(
    solution: List[List[AlnsTask]],
    trajectory_dict: Dict[int, Dict],
    planner: 'DronePathPlanner',
    n_sample_uavs: int = 3,
    max_tasks_legend_per_uav: int = 3,
    output_path: str = 'trajectory_3d_sample.png',
):
    """
    绘制前若干架 UAV 的微观三维航迹 + 区域内建筑物柱状体。
    风格对齐常见论文/报告图：经纬度平面、垂直高度轴、棕色建筑、分段线型图例。
    """
    ls_cycle = ('-', '--', ':')  # 与参考图：实线 / 虚线 / 点划线

    def _building_poly_faces(b) -> Tuple[list, dict]:
        bnd = b.get_bounds()
        x0, x1 = bnd['min_lon'], bnd['max_lon']
        y0, y1 = bnd['min_lat'], bnd['max_lat']
        h = float(b.height)
        verts = [
            [x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0],
            [x0, y0, h], [x1, y0, h], [x1, y1, h], [x0, y1, h],
        ]
        faces = [
            [verts[0], verts[1], verts[5], verts[4]],
            [verts[2], verts[3], verts[7], verts[6]],
            [verts[0], verts[3], verts[7], verts[4]],
            [verts[1], verts[2], verts[6], verts[5]],
            [verts[4], verts[5], verts[6], verts[7]],
        ]
        shade = np.clip(float(h) / 180.0, 0.0, 1.0)
        r = 0.75 + 0.12 * shade
        g = 0.65 + 0.10 * shade
        b_chan = 0.48 + 0.06 * shade
        # 半透明避免遮挡上层航迹（mplot3d 深度排序不完美，压低 alpha 更稳）
        props = {'facecolor': (r * 0.88, g * 0.86, b_chan),
                 'alpha': float(np.clip(0.16 + 0.00045 * float(h), 0.14, 0.28)),
                 'edgecolor': '#8B7355'}
        return faces, props

    # ── A. 采样范围内建筑物（去重写入） ───────────────────────────────────
    plotted_buildings = set()
    poly_specs: List[Tuple] = []
    all_xyz: List[Tuple[float, float, float]] = []

    for uav_idx in range(min(n_sample_uavs, len(solution))):
        sorted_t = sorted(solution[uav_idx], key=lambda t: t.dronetime_ts)
        for alns_t in sorted_t:
            info = trajectory_dict.get(alns_t.task_id)
            if not info:
                continue
            path = info['path']
            for p in path:
                all_xyz.append((
                    float(p['lon']), float(p['lat']), float(p.get('height', 0))))
            dt = DeliveryTask(
                task_id=alns_t.task_id,
                sender_lon=alns_t.sender[0],
                sender_lat=alns_t.sender[1],
                receiver_lon=alns_t.receiver[0],
                receiver_lat=alns_t.receiver[1],
            )
            for b in planner.get_obstacles_in_region(dt):
                bid = getattr(b, 'building_id', id(b))
                if bid in plotted_buildings:
                    continue
                plotted_buildings.add(bid)
                faces, pprops = _building_poly_faces(b)
                poly_specs.append((faces, pprops))

    # ── B. 画布与轴（显式中文字体，避免 mplot3d 量字时退回 Arial 报缺字）────────
    fp = _zh_font_props()
    fig = plt.figure(figsize=(13.6, 8.8))
    ax = fig.add_subplot(111, projection='3d')
    _kw = dict(fontproperties=fp) if fp is not None else {}
    ax.set_title(
        '微观三维航迹与建筑物示意（经纬度 × 离地高度）', fontsize=15, pad=12, **_kw)
    ax.set_xlabel('经度 (°)', **_kw)
    ax.set_ylabel('纬度 (°)', **_kw)
    ax.set_zlabel('高度 (m)', **_kw)

    for faces, pr in poly_specs:
        coll = Poly3DCollection(
            faces,
            facecolors=pr['facecolor'],
            linewidths=0.14,
            edgecolors=pr['edgecolor'],
            alpha=pr['alpha'],
        )
        ax.add_collection3d(coll)

    # ── C. 航迹线 + 起降点 ─────────────────────────────────────────────────
    for uav_idx in range(min(n_sample_uavs, len(solution))):
        color = UAV_COLORS[uav_idx % len(UAV_COLORS)]
        tasks_list = sorted(solution[uav_idx], key=lambda t: t.dronetime_ts)
        for t_rank, alns_t in enumerate(tasks_list):
            info = trajectory_dict.get(alns_t.task_id)
            if not info:
                continue
            path = info['path']
            xs = [float(p['lon']) for p in path]
            ys = [float(p['lat']) for p in path]
            zs = [float(p.get('height', 0)) for p in path]
            ls = ls_cycle[t_rank % len(ls_cycle)]
            show_lbl = t_rank < max_tasks_legend_per_uav
            lbl = (
                f'UAV{uav_idx + 1} 任务 {alns_t.task_id}'
                if show_lbl
                else None
            )
            ax.plot(
                xs, ys, zs,
                ls, color=color, linewidth=2.45, alpha=0.96,
                label=lbl,
                zorder=10,
            )
            ax.scatter(
                [xs[0]], [ys[0]], [zs[0]],
                color='#2ECC71', s=52, marker='o',
                edgecolors='#1f5f3a', linewidths=0.5, alpha=1.0, zorder=7)
            ax.scatter(
                [xs[-1]], [ys[-1]], [zs[-1]],
                color='#E62121', s=52, marker='o',
                edgecolors='#7a1515', linewidths=0.5, alpha=1.0, zorder=7)

    # ── D. 范围与视角（接近常见参考示意图） ───────────────────────────────
    if len(all_xyz) >= 2:
        arr = np.asarray(all_xyz, dtype=np.float64)
        lx = np.ptp(arr[:, 0]) or 5e-4
        ly = np.ptp(arr[:, 1]) or 5e-4
        pad_x = lx * 0.06 + 2e-4
        pad_y = ly * 0.06 + 2e-4
        ax.set_xlim(float(arr[:, 0].min() - pad_x), float(arr[:, 0].max() + pad_x))
        ax.set_ylim(float(arr[:, 1].min() - pad_y), float(arr[:, 1].max() + pad_y))
        zm = float(np.percentile(arr[:, 2], 99.5))
        zm = max(zm * 1.12, ALT_MAX_M * 2.9, float(np.max(arr[:, 2])) * 1.08)
        ax.set_zlim(0.0, float(min(520.0, max(340.0, zm))))
    else:
        ax.set_zlim(0.0, 350.0)

    ax.view_init(elev=23, azim=-58)
    if fp is not None:
        _apply_zh_fp_to_3d_axes(ax, fp)

    traj_h, traj_l = ax.get_legend_handles_labels()
    uniq_tl = OrderedDict()
    for h_item, lb in zip(traj_h, traj_l):
        if lb and lb not in uniq_tl:
            uniq_tl[lb] = h_item

    proxy_g = Line2D(
        [0], [0], linestyle='', marker='o', markersize=8,
        markerfacecolor='#2ECC71', markeredgecolor='#1f5f3a',
        markeredgewidth=0.6)
    proxy_r = Line2D(
        [0], [0], linestyle='', marker='o', markersize=8,
        markerfacecolor='#E62121', markeredgecolor='#7a1515',
        markeredgewidth=0.6)

    merged_h = list(uniq_tl.values()) + [proxy_g, proxy_r]
    merged_l = list(uniq_tl.keys()) + ['起点（绿●）', '终点（红●）']
    if merged_h:
        leg_kw = dict(
            loc='upper left',
            fontsize=9,
            framealpha=0.94,
            borderpad=0.55,
        )
        if fp is not None:
            leg_kw['prop'] = fp
        leg = ax.legend(merged_h, merged_l, **leg_kw)
        if fp is not None and leg is not None:
            for t in leg.get_texts():
                t.set_fontproperties(fp)

    _note = (
        '线型顺序：同一 UAV 多条任务依次为 实线 / 虚线 / 点划线；建筑物为半透明柱体'
    )
    if fp is not None:
        fig.text(0.02, 0.02, _note, transform=fig.transFigure,
                 fontsize=8, verticalalignment='bottom', alpha=0.85, fontproperties=fp)
    else:
        fig.text(0.02, 0.02, _note, transform=fig.transFigure,
                 fontsize=8, verticalalignment='bottom', alpha=0.85)

    plt.tight_layout()
    plt.savefig(output_path, dpi=165, bbox_inches='tight')
    plt.close()
    print(f"  3D 微观航迹图已保存：{output_path}")



# ══════════════════════════════════════════════════════════════════════════════
# §12  主入口
# ══════════════════════════════════════════════════════════════════════════════

def expert_system_assign(
    tasks: List[AlnsTask],
    num_uavs: int,
    warehouse: Tuple[float, float],
    uavs: List['AlnsUAV'],
    stations: List[Tuple[float, float]],
    wind_speed_ms: float,
    wind_dir_deg: float,
) -> List[List[AlnsTask]]:
    """
    专家系统宏观任务分配（规则驱动，替代 ALNS 迭代优化）。

    背景：数据时间窗口仅 6~30 分钟（均值约 20 分钟），10 架无人机处理 700 单时
    大多数任务在时序上无法满足严格截止时间约束（UAV 还在执行上一单）。
    因此采用"最优努力（Best-Effort）"策略：

    分配规则（优先级从高到低）：
      1. 按截止时间（dronetime）升序处理任务——越紧急越优先；
      2. 优先选能满足截止时间 AND 电量充足的无人机中"最早到达"的；
      3. 若无满足截止时间的候选，退而选"最早能到达"的（Best-Effort）；
      4. 若所有无人机电量连充电站都到不了，才分配给负载最少的无人机（兜底）。

    统计口径：
      - on_time   : 选到了满足截止时间的候选
      - best_effort: 无法满足截止时间，但选了最快能到达的
      - fallback  : 所有 UAV 电量耗尽，随机兜底

    返回值格式：List[List[AlnsTask]]
    """
    sorted_tasks = sorted(tasks, key=lambda x: x.dronetime_ts)
    uav_tasks    = [[] for _ in range(num_uavs)]

    # 每架 UAV 的状态：完成当前任务（含充电）后的 可用时刻、位置、剩余电量
    states = [{
        'avail_time':  0.0,               # UAV 下次可接单的时刻（秒，epoch）
        'pos':         warehouse,         # 届时所在位置
        'battery':     uavs[i].max_battery,
    } for i in range(num_uavs)]

    on_time     = 0
    best_effort = 0
    fallback    = 0

    for task in sorted_tasks:
        best_idx   = -1
        best_score = float('inf')   # 越小越好：按时完成=取件时刻；否则=完成时刻
        be_idx     = -1
        be_score   = float('inf')

        for idx in range(num_uavs):
            sp = uavs[idx].current_speed
            s  = states[idx]

            # ── 第一段：当前位置 → 取件点 ──────────────────────────────────
            h_to_s, bat_to_s = _wind_flight_info(
                s['pos'], task.sender, wind_speed_ms, wind_dir_deg, sp)

            # ── 第二段：取件点 → 收件点 ────────────────────────────────────
            h_s_to_r, bat_s_to_r = _wind_flight_info(
                task.sender, task.receiver, wind_speed_ms, wind_dir_deg, sp)

            # ── 第三段：收件点 → 最近充电站（交付后充电准备下一单）──────────
            ns, h_r_to_ns, bat_r_to_ns = _find_station_info(
                task.receiver, stations, wind_speed_ms, wind_dir_deg, sp)

            # 全链路电量需求：pos→sender→receiver→station
            bat_full_route = bat_to_s + bat_s_to_r + bat_r_to_ns

            # 电量不够全程：先去充电站充电
            if s['battery'] < bat_full_route:
                # 飞到充电站
                ns_pre, h_pre, bat_pre = _find_station_info(
                    s['pos'], stations, wind_speed_ms, wind_dir_deg, sp)
                if s['battery'] < bat_pre:
                    continue   # 连充电站都到不了，完全跳过
                charge_amt = uavs[idx].max_battery - (s['battery'] - bat_pre)
                charge_t   = charge_amt * uavs[idx].charge_time_per_km
                h_ns_to_s, _ = _wind_flight_info(
                    ns_pre, task.sender, wind_speed_ms, wind_dir_deg, sp)
                t_pickup = (s['avail_time']
                            + h_pre * 3600 + charge_t
                            + h_ns_to_s * 3600)
                bat_after_pickup = uavs[idx].max_battery - bat_s_to_r
            else:
                t_pickup     = s['avail_time'] + h_to_s * 3600
                bat_after_pickup = s['battery'] - bat_to_s - bat_s_to_r

            # 不早于下单时间
            t_pickup = max(t_pickup, task.otime.timestamp())

            # ── 主动充电：取件前有空闲且电量 < 80% 时去充电 ────────────────
            idle_s = task.otime.timestamp() - (s['avail_time'] + h_to_s * 3600)
            if (idle_s > 300                          # 至少 5 分钟空闲
                    and s['battery'] < 0.8 * uavs[idx].max_battery):
                _, h_idle_ns, bat_idle_ns = _find_station_info(
                    s['pos'], stations, wind_speed_ms, wind_dir_deg, sp)
                if s['battery'] >= bat_idle_ns:
                    fill_amt  = uavs[idx].max_battery - s['battery']
                    charge_t2 = fill_amt * uavs[idx].charge_time_per_km
                    if h_idle_ns * 3600 * 2 + charge_t2 <= idle_s:
                        # 来回充电不耽误取件，更新交付后电量
                        bat_after_pickup = uavs[idx].max_battery - bat_s_to_r
                        t_pickup = task.otime.timestamp()

            # ── 交付完成时刻（核心：UAV 下次可用时刻）────────────────────────
            t_deliver = t_pickup + h_s_to_r * 3600

            # 交付后充电时间（若电量不足下次任务）
            if bat_after_pickup < 0.3 * uavs[idx].max_battery:
                fill_after = uavs[idx].max_battery - bat_after_pickup
                t_deliver += h_r_to_ns * 3600 + fill_after * uavs[idx].charge_time_per_km

            # ── 评分：优先按时完成，其次完成时最早 ──────────────────────────
            if t_pickup <= task.dronetime_ts and t_pickup < best_score:
                best_score, best_idx = t_pickup, idx
            if t_deliver < be_score:
                be_score, be_idx = t_deliver, idx

        # 选定无人机
        if best_idx != -1:
            chosen_idx = best_idx
            on_time   += 1
        elif be_idx != -1:
            chosen_idx   = be_idx
            best_effort += 1
        else:
            chosen_idx = min(range(num_uavs), key=lambda i: len(uav_tasks[i]))
            fallback   += 1

        # ── 更新 chosen UAV 的状态 ────────────────────────────────────────
        sp = uavs[chosen_idx].current_speed
        s  = states[chosen_idx]

        h_to_s2, bat_to_s2 = _wind_flight_info(
            s['pos'], task.sender, wind_speed_ms, wind_dir_deg, sp)
        h_s_r2, bat_s_r2 = _wind_flight_info(
            task.sender, task.receiver, wind_speed_ms, wind_dir_deg, sp)
        ns2, h_r_ns2, bat_r_ns2 = _find_station_info(
            task.receiver, stations, wind_speed_ms, wind_dir_deg, sp)

        # 是否需要先充电
        bat_full2 = bat_to_s2 + bat_s_r2 + bat_r_ns2
        if s['battery'] < bat_full2:
            ns_pre2, h_pre2, bat_pre2 = _find_station_info(
                s['pos'], stations, wind_speed_ms, wind_dir_deg, sp)
            charge_amt2 = uavs[chosen_idx].max_battery - (s['battery'] - bat_pre2)
            charge_t2   = charge_amt2 * uavs[chosen_idx].charge_time_per_km
            h_ns_s2, _  = _wind_flight_info(
                ns_pre2, task.sender, wind_speed_ms, wind_dir_deg, sp)
            t_pickup2   = max(s['avail_time'] + h_pre2 * 3600 + charge_t2 + h_ns_s2 * 3600,
                              task.otime.timestamp())
            bat_at_recv = uavs[chosen_idx].max_battery - bat_s_r2
        else:
            t_pickup2   = max(s['avail_time'] + h_to_s2 * 3600, task.otime.timestamp())
            bat_at_recv = s['battery'] - bat_to_s2 - bat_s_r2

        t_deliver2 = t_pickup2 + h_s_r2 * 3600

        # 交付后充电：若剩余电量 < 30%，记录充电后的状态
        if bat_at_recv < 0.3 * uavs[chosen_idx].max_battery:
            fill3     = uavs[chosen_idx].max_battery - bat_at_recv
            t_avail   = t_deliver2 + h_r_ns2 * 3600 + fill3 * uavs[chosen_idx].charge_time_per_km
            bat_avail = uavs[chosen_idx].max_battery
            pos_avail = ns2
        else:
            t_avail   = t_deliver2
            bat_avail = bat_at_recv
            pos_avail = task.receiver

        # 写回真实起飞时刻，供微观航迹以调度结果为 start_ts（而非 CSV 原始 dronetime）
        task.set_scheduled_start(t_pickup2)

        uav_tasks[chosen_idx].append(task)
        states[chosen_idx] = {
            'avail_time': t_avail,
            'pos':        pos_avail,
            'battery':    bat_avail,
        }

    total_km = sum(haversine(t.sender, t.receiver) for ts in uav_tasks for t in ts)
    print(f"  [专家系统] 分配完成：共 {len(tasks)} 条任务")
    print(f"    ✓ 按时完成 {on_time} 条 | ～ 最优努力 {best_effort} 条 | ✗ 兜底 {fallback} 条")
    print(f"  [专家系统] 总配送距离（直线）: {total_km:.2f} km")
    return uav_tasks


# 宏观 OD 时空 distancing：起终点相近的任务拉开起飞间隔，降低微观机间冲突概率
_MACRO_OD_NEAR_KM = 0.8    # 取件点或收件点距离低于此值视为空间相近（km）
_MACRO_MIN_SEP_S  = 90.0   # 相近任务最小起飞间隔（秒）


def apply_macro_od_temporal_distancing(
    solution: List[List[AlnsTask]],
    near_km: float = _MACRO_OD_NEAR_KM,
    min_sep_s: float = _MACRO_MIN_SEP_S,
) -> int:
    """
    对已分配任务做跨 UAV 宏观时间错峰：若起终点空间相近且起飞过近，
    则推迟较晚一方（保留绝对 deadline；不可行则跳过该对）。

    返回实际调整起飞时刻的任务条数。
    """
    flat: List[AlnsTask] = [t for uav_ts in solution for t in uav_ts]
    if len(flat) < 2:
        return 0

    flat.sort(key=lambda t: t.dronetime_ts)
    shifted = 0
    total_delay_s = 0.0

    for i in range(1, len(flat)):
        ti = flat[i]
        need_ts = ti.dronetime_ts
        for j in range(i):
            tj = flat[j]
            near = (
                haversine(ti.sender, tj.sender) < near_km
                or haversine(ti.receiver, tj.receiver) < near_km
            )
            if not near:
                continue
            need_ts = max(need_ts, tj.dronetime_ts + min_sep_s)

        if need_ts <= ti.dronetime_ts + 1e-6:
            continue

        # 保留分配时的绝对截止：推迟起飞后仍须 need_ts ≤ deadline_ts
        if need_ts > ti.deadline_ts + 1e-6:
            continue

        delay = need_ts - ti.dronetime_ts
        ti.set_scheduled_start(need_ts, keep_deadline=True)
        shifted += 1
        total_delay_s += delay

    avg_delay = (total_delay_s / shifted) if shifted else 0.0
    print(f"  [宏观 distancing] 错峰调整 {shifted} 条"
          f"（近邻阈值 {near_km} km / 最小间隔 {min_sep_s:.0f} s"
          f" / 平均推迟 {avg_delay:.1f} s）")
    return shifted


def main():
    # ── 文件路径 ──────────────────────────────────────────────────────────────
    BASE = os.path.dirname(os.path.abspath(__file__))
    DELIVERY_CSV  = os.path.join(BASE, 'nanjing_delivery_700_2026_01.csv')
    BUILDINGS_CSV = os.path.join(BASE, 'nanjing_buildings.csv')
    WIND_CSV      = os.path.join(BASE, 'wind.csv')

    # ── 参数 ──────────────────────────────────────────────────────────────────
    NUM_UAVS       = 30
    N_STATIONS     = 6
    UAV_SPEED      = 75.0     # km/h（宏观调度无人机速度）

    # ── 1. 加载数据 ───────────────────────────────────────────────────────────
    tasks = load_delivery_tasks(DELIVERY_CSV)
    # 时变风（按订单 dronetime 查询）+ 均值（给日级调度用）
    wind_ts = load_wind_timeseries(WIND_CSV)
    if len(wind_ts) > 0:
        wind_speed_ms, wind_dir_deg = wind_ts.mean()
        print(f"[Wind] 时变样本 {len(wind_ts)} 条；日均 {wind_speed_ms:.2f} m/s → {wind_dir_deg:.1f}°")
        # 打印首末 3 个任务的风样本，便于核对时间戳是否对齐
        preview = sorted(tasks, key=lambda t: t.dronetime_ts)
        for t in preview[:3] + preview[-3:]:
            ws, wd = wind_ts.at(t.dronetime)
            print(f"       task {t.task_id:>4} @ {t.dronetime.strftime('%Y-%m-%d %H:%M')}  "
                  f"→ {ws:.2f} m/s, {wd:.1f}°")
    else:
        wind_speed_ms, wind_dir_deg = load_wind_from_csv(WIND_CSV)
        print(f"[Wind] 未加载到时变数据，退回均值 {wind_speed_ms:.2f} m/s / {wind_dir_deg:.1f}°")

    # ── 2. 仓库 + 充电站 ──────────────────────────────────────────────────────
    all_pts   = [p for t in tasks for p in (t.sender, t.receiver)]
    warehouse = (
        float(np.mean([p[0] for p in all_pts])),
        float(np.mean([p[1] for p in all_pts])),
    )
    stations = calculate_charging_stations(tasks, N_STATIONS)

    # ── 3. 专家系统宏观分配（用日均风做全天电量估算；调度为日级不宜每单查风）──
    uavs     = [AlnsUAV(i, speed_kmh=UAV_SPEED) for i in range(NUM_UAVS)]
    best_sol = expert_system_assign(
        tasks, NUM_UAVS, warehouse, uavs, stations, wind_speed_ms, wind_dir_deg)
    apply_macro_od_temporal_distancing(best_sol)

    # ── 4. 微观航迹规划（DronePathPlanner + 障碍物 + 按订单时刻的时变风）────
    _, trajectory_dict = plan_micro_trajectories(
        best_sol, tasks, BUILDINGS_CSV, wind_speed_ms, wind_dir_deg, NUM_UAVS,
        wind_lookup = wind_ts if len(wind_ts) > 0 else None)

    # 三维微观航迹参考示意（对齐常见论文插图风格）
    planner_3d = DronePathPlanner(pd.read_csv(BUILDINGS_CSV))
    visualize_3d_sample(
        best_sol,
        trajectory_dict,
        planner_3d,
        n_sample_uavs=3,
        max_tasks_legend_per_uav=3,
        output_path=os.path.join(BASE, '输出左边.png'),
    )

    # ── 5. 冲突类指标 ────────────────────────────────────────────────────────
    conflict_metrics = compute_conflict_metrics(trajectory_dict, safe_dist_m=50.0)
    print_conflict_metrics(conflict_metrics)

    # ── 6. 成本类指标 ────────────────────────────────────────────────────────
    cost_metrics = compute_cost_metrics(
        trajectory_dict,
        alns_speed_ms  = UAV_SPEED / 3.6,   # km/h → m/s
        smooth_coeff1  = 1.0,
        smooth_coeff2  = 0.5,
    )
    print_cost_metrics(cost_metrics)

    # ── 7. 保存指标 CSV ──────────────────────────────────────────────────────
    export_conflict_metrics_csv(
        conflict_metrics,
        cost_metrics   = cost_metrics,
        output_path    = os.path.join(BASE, 'conflict_metrics.csv'))


if __name__ == '__main__':
    main()
