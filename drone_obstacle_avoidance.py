"""
无人机配送避障算法

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, List, Optional, Tuple
from scipy.spatial import KDTree
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置参数
CRUISE_HEIGHT = 60  # 巡航高度（米）——最低合规飞行层
BUILDING_THRESHOLD = 120  # 建筑物高度阈值（米），超过则水平绕行
SAFETY_MARGIN = 20  # 安全距离（米）
DETECTION_RADIUS = 0.005  # 障碍物检测半径（度，约500米）
WIND_POWER_FACTOR = 0.12  # 风对功率影响系数：逆风每 m/s 增加约 12% 功率（经验值）

# ── 高度分层（所有合规飞行高度必须是下面三者之一） ─────────────────────────
# 规则：建筑 h → 选择满足 layer >= h 的最低层；若 h > 120m 则不能飞越，需水平绕行。
ALT_LAYERS = (60.0, 90.0, 120.0)
ALT_MAX_M  = 120.0

# ── 机队配置（SAC2 / 训练 / 宏观调度共用）────────────────────────────────────
FLEET_NUM_FIXED_WING = 5
FLEET_NUM_MULTIROTOR = 25
MR_BATTERY_CAPACITY_WH = 850.0
FW_BATTERY_CAPACITY_WH = 1500.0

# 宏观巡航功率（W）：与 Drone.BATTERY_DEFAULTS 的 cruise_power_w 一致，
# 供 AlnsUAV 等宏观层在不实例化 Drone 的情况下取用。
MACRO_MR_CRUISE_POWER_W = 900.0
MACRO_FW_CRUISE_POWER_W = 400.0

# 宏观充电功率（W）：假定统一充电桩功率，充电时长 = 缺电量(Wh) / 功率(W) * 3600s。
MACRO_CHARGE_POWER_W = 1500.0
MACRO_CHARGE_SECONDS_PER_WH = 3600.0 / MACRO_CHARGE_POWER_W  # = 2.4 s/Wh

# ── deprecated：等效 km 标定（宏观已迁移至 Wh，仅为兼容旧引用保留） ──────────
_LEGACY_MR_WH = 120.0
_LEGACY_FW_WH = 200.0
_LEGACY_MACRO_BATTERY_KM = 34.0
MACRO_MR_BATTERY_KM = _LEGACY_MACRO_BATTERY_KM * (MR_BATTERY_CAPACITY_WH / _LEGACY_MR_WH)
MACRO_FW_BATTERY_KM = _LEGACY_MACRO_BATTERY_KM * (FW_BATTERY_CAPACITY_WH / _LEGACY_FW_WH)

FLEET_FIXED_WING_RATIO = FLEET_NUM_FIXED_WING / (FLEET_NUM_FIXED_WING + FLEET_NUM_MULTIROTOR)


def is_fixed_wing_craft_type(craft_type: str) -> bool:
    return craft_type in (
        'pure_fixed_wing', 'vtol_tiltrotor', 'hybrid_fixed_rotor', 'fixed_wing',
    )


def fleet_fixed_wing_count(num_uavs: int) -> int:
    """按 5 固翼 + 25 多旋翼比例，计算 num_uavs 架中的固定翼数量。"""
    total = FLEET_NUM_FIXED_WING + FLEET_NUM_MULTIROTOR
    return min(num_uavs, max(0, round(num_uavs * FLEET_NUM_FIXED_WING / total)))


def macro_battery_wh_for_craft(craft_type: str) -> float:
    """宏观满电电量（Wh），与微观 Drone.battery_capacity_wh 口径一致。"""
    if is_fixed_wing_craft_type(craft_type):
        return FW_BATTERY_CAPACITY_WH
    return MR_BATTERY_CAPACITY_WH


def macro_cruise_power_w_for_craft(craft_type: str, payload_kg: float = 0.0) -> float:
    """宏观巡航功率（W），含载重附加功率（近似取 Drone.BATTERY_DEFAULTS 的 payload_power_factor）。"""
    if is_fixed_wing_craft_type(craft_type):
        base, factor = MACRO_FW_CRUISE_POWER_W, 0.15
    else:
        base, factor = MACRO_MR_CRUISE_POWER_W, 0.25
    return base + max(0.0, payload_kg) * factor


def macro_battery_km_for_craft(craft_type: str) -> float:
    """Deprecated：等效 km 标定，宏观层已迁移至 Wh。仅保留给未迁移的历史脚本使用。"""
    if is_fixed_wing_craft_type(craft_type):
        return MACRO_FW_BATTERY_KM
    return MACRO_MR_BATTERY_KM


def snap_to_layer(h: float) -> float:
    """把任意高度对齐到 {60, 90, 120}m 中最低的、且 >= h 的那一层。"""
    for layer in ALT_LAYERS:
        if layer >= h:
            return layer
    return ALT_MAX_M


def load_wind_from_csv(csv_path: str = 'wind.csv') -> Tuple[float, float]:
    """
    从 wind.csv 加载风场数据，返回 (平均风速 m/s, 平均风向_吹向 度)。
    csv 中 风向_度 为气象 convention：风从该方向吹来（from）；返回为吹向（to）= from+180°。
    列名：时间, 风速_m_s, 风向_度, ...
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception:
        return 0.0, 0.0
    if df.empty or '风速_m_s' not in df.columns or '风向_度' not in df.columns:
        return 0.0, 0.0
    wind_speed = float(df['风速_m_s'].mean())
    wind_from = df['风向_度'].values
    wind_to_rad = np.radians((wind_from + 180) % 360)
    wx = np.mean(np.cos(wind_to_rad))
    wy = np.mean(np.sin(wind_to_rad))
    wind_to_deg = (np.degrees(np.arctan2(wy, wx)) % 360)
    return wind_speed, wind_to_deg


# ---------------- 时变风场 ----------------
from datetime import datetime as _dt

_WIND_EPOCH = _dt(1970, 1, 1)


def _naive_dt_to_seconds(dt: _dt) -> float:
    """naive datetime → 相对 1970-01-01 的秒数（时区无关，用于 WindTimeseries 键对齐）。"""
    return (dt - _WIND_EPOCH).total_seconds()


class WindTimeseries:
    """
    时变风场查询器。

    - at(dt) / __call__(dt)：给定 naive datetime 返回 (wind_speed_ms, wind_dir_to_deg)，
      内部做最近邻查找（O(log N)）。
    - mean()：整体均值，接口与 load_wind_from_csv 一致。
    - 为了时区无关，一律使用 (naive_dt - 1970-01-01).total_seconds() 做索引键。
    """

    def __init__(self, keys_s: np.ndarray, speeds: np.ndarray, dir_to_deg: np.ndarray):
        self._keys = np.asarray(keys_s, dtype=np.float64)
        self._sp = np.asarray(speeds, dtype=np.float64)
        self._dr = np.asarray(dir_to_deg, dtype=np.float64)
        if len(self._keys):
            order = np.argsort(self._keys)
            self._keys = self._keys[order]
            self._sp = self._sp[order]
            self._dr = self._dr[order]

    def __len__(self) -> int:
        return int(len(self._keys))

    def at(self, dt) -> Tuple[float, float]:
        """按 datetime 最近邻查找。找不到数据时返回 (0.0, 0.0)。"""
        if len(self._keys) == 0 or dt is None:
            return 0.0, 0.0
        if isinstance(dt, (int, float)):
            key = float(dt)
        else:
            key = _naive_dt_to_seconds(dt)
        idx = int(np.clip(np.searchsorted(self._keys, key), 0, len(self._keys) - 1))
        if idx > 0 and abs(self._keys[idx - 1] - key) < abs(self._keys[idx] - key):
            idx -= 1
        return float(self._sp[idx]), float(self._dr[idx])

    def __call__(self, dt) -> Tuple[float, float]:
        return self.at(dt)

    def mean(self) -> Tuple[float, float]:
        """整体均值：风速算术平均，风向按向量平均（与 load_wind_from_csv 一致）。"""
        if len(self._keys) == 0:
            return 0.0, 0.0
        speed = float(self._sp.mean())
        rad = np.radians(self._dr)
        wx = float(np.mean(np.cos(rad)))
        wy = float(np.mean(np.sin(rad)))
        return speed, float(np.degrees(np.arctan2(wy, wx)) % 360)


def load_wind_timeseries(csv_path: str = 'wind.csv') -> WindTimeseries:
    """
    加载 wind.csv 并返回可按时间查询的 WindTimeseries 对象。

    列要求：时间, 风速_m_s, 风向_度
    风向从"气象 from"转换为"吹向 to"= (from + 180) % 360。
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception:
        return WindTimeseries(np.array([]), np.array([]), np.array([]))
    need = {'时间', '风速_m_s', '风向_度'}
    if df.empty or not need.issubset(df.columns):
        return WindTimeseries(np.array([]), np.array([]), np.array([]))
    ts = pd.to_datetime(df['时间'], errors='coerce')
    keys = np.array(
        [(t.to_pydatetime() - _WIND_EPOCH).total_seconds() if pd.notna(t) else np.nan
         for t in ts],
        dtype=np.float64,
    )
    mask = ~np.isnan(keys)
    keys = keys[mask]
    speeds = df['风速_m_s'].values.astype(float)[mask]
    dir_to = (df['风向_度'].values.astype(float)[mask] + 180.0) % 360.0
    return WindTimeseries(keys, speeds, dir_to)


# -------------------------- 奖励函数（来自 MultirotorUAV_4，供与多旋翼状态联用时使用） --------------------------
try:
    from MultirotorUAV_4 import (
        destination_reward,
        obstacle_reward,
        uav_interval_reward,
        battery_reward,
        total_reward as multirotor_total_reward,
    )
except ImportError:
    destination_reward = obstacle_reward = uav_interval_reward = battery_reward = multirotor_total_reward = None


# -------------------------- 路径奖励（基于航迹的优先级评估，用于本脚本规划结果） --------------------------
# 优先级：P1 机间避碰 > P2 避障 > P3 航迹意图(到达目的地) > P4 电量消耗


def path_uav_interval_reward(path: List[Dict], other_paths: List[List[Dict]],
                             safe_interval_m: float = 50.0, lat_ref: float = 32.0,
                             cruise_speed_ms: float = 15.0,
                             uav_radius_m: float = 1.0,
                             safe_vertical_m: float = 30.0,
                             time_window_s: float = None) -> float:
    """
    P1 机间避碰（最高优先级，惩罚 -2000）。

    与 MultirotorUAV_4.py 保持一致：
      · 反应时间：12.5 s，预测距离 = cruise_speed × 12.5（约等于航段末端）
      · 冲突判断：水平距离 AND 垂直高度差同时满足才算冲突
          - 水平净间隔 < safe_interval_m（50 m）
          - 垂直高度差 < safe_vertical_m（30 m，即高度分层阈值）
        两个条件同时满足才触发惩罚，高度已分层（|Δz| ≥ 30m）的路径不计冲突。
        原纯 3D 距离判断会把"水平很近但高度已分层"的合法编队误判为冲突，
        导致高度分层策略无法生效。

    time_window_s:
        若航点携带 '_t'（unix 秒）字段且此参数不为 None，则只有两机在
        |t_自 - t_他| < time_window_s 范围内才认定为真实冲突；
        时间差超出窗口（两机先后经过同一地点）直接跳过，不视为冲突。
        传 None（默认）时退化为纯空间检测（向后兼容）。
    """
    if not other_paths:
        return 0.0

    # 对齐 MultirotorUAV_4.py：反应时间 12.5s，预判距离 = 巡航速度 × 12.5s
    # 在离线路径规划中，这意味着检查每段航段末端附近的位置（比 0.5s 更保守）
    REACTION_S   = 12.5
    PREDICT_DIST = cruise_speed_ms * REACTION_S

    def _wp_xyz(wp: Dict):
        return (
            wp['lon'] * 111000.0 * np.cos(np.radians(lat_ref)),
            wp['lat'] * 111000.0,
            float(wp.get('height', 0.0)),
        )

    def _predict_pos(p1: Dict, p2: Dict) -> tuple:
        """沿 p1→p2 方向前推 PREDICT_DIST 米得到预测位置。"""
        x1, y1, z1 = _wp_xyz(p1)
        x2, y2, z2 = _wp_xyz(p2)
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        seg = np.sqrt(dx**2 + dy**2 + dz**2)
        if seg < 1e-6:
            return x1, y1, z1
        ratio = min(PREDICT_DIST / seg, 1.0)
        return x1 + dx * ratio, y1 + dy * ratio, z1 + dz * ratio

    reward = 0.0
    check_time = (time_window_s is not None)
    for i in range(len(path) - 1):
        p1_wp, p2_wp = path[i], path[i + 1]
        # 跳过起降段（起点或终点在地面），避免爬升穿层引发误判
        if float(p1_wp.get('height', 0)) <= 0 or float(p2_wp.get('height', 0)) <= 0:
            continue
        px, py, pz = _predict_pos(p1_wp, p2_wp)
        t_self = p1_wp.get('_t') if check_time else None
        for opath in other_paths:
            for q in opath:
                if float(q.get('height', 0)) <= 0:
                    continue   # 同样跳过他机的起降点
                # ── 时间维度过滤：两机到达各自位置的时间差超窗口则非真实冲突 ──
                if check_time and t_self is not None:
                    t_other = q.get('_t')
                    if t_other is not None and abs(t_self - t_other) > time_window_s:
                        continue
                qx, qy, qz = _wp_xyz(q)
                # 水平净间隔
                h_dist = np.sqrt((px - qx)**2 + (py - qy)**2) - 2 * uav_radius_m
                # 垂直高度差
                v_diff = abs(pz - qz)
                # 只有水平AND垂直同时不足时才是真冲突（高度已分层则安全）
                if h_dist < safe_interval_m and v_diff < safe_vertical_m:
                    reward -= 2000
                    break
    return reward


def path_obstacle_reward(path: List[Dict], obstacles: List, task: 'DeliveryTask',
                         safe_r_m: float = 20.0, lat_ref: float = 32.0,
                         cruise_speed_ms: float = 15.0) -> float:
    """
    P2 避障（第二优先级）。

    与 MultirotorUAV_4.py 保持一致：
      · 1 s 预判（next_1s）：沿当前段方向前推 cruise_speed_ms × 1 s 米，
                              若落入建筑范围 → 惩罚 -800（P2 第二优先级）。
      · 5 s 预判（next_5s）：沿当前段方向前推 cruise_speed_ms × 5 s 米，
                              若落入建筑范围 → 惩罚 -200（5s 预判，惩罚较轻）。

    排除逻辑（不变）：
      · 高度为 0 的航点（起飞 / 降落）不参与碰撞判定。
      · 起降区域内的航点（0.0005°≈55 m）跳过。
    """
    _EXCL_DEG   = 0.0005           # 起降点排除半径：约 55 m
    PREDICT_1S  = cruise_speed_ms * 1.0    # 1 s 预判距离（m）
    PREDICT_5S  = cruise_speed_ms * 5.0    # 5 s 预判距离（m）

    def _predict(p1: Dict, p2: Dict, dist_m: float) -> tuple:
        """沿 p1→p2 方向前推 dist_m 米，返回 (x_m, y_m, h_m)。"""
        x1 = p1['lon'] * 111000.0 * np.cos(np.radians(lat_ref))
        y1 = p1['lat'] * 111000.0
        z1 = float(p1.get('height', 0.0))
        x2 = p2['lon'] * 111000.0 * np.cos(np.radians(lat_ref))
        y2 = p2['lat'] * 111000.0
        z2 = float(p2.get('height', 0.0))
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        seg = np.sqrt(dx**2 + dy**2 + dz**2)
        ratio = min(dist_m / seg, 1.0) if seg > 1e-6 else 0.0
        return x1 + dx * ratio, y1 + dy * ratio, z1 + dz * ratio

    reward = 0.0
    for i in range(len(path) - 1):
        p = path[i]
        ph = float(p.get('height', 0.0))

        # 地面点直接跳过
        if ph <= 0:
            continue
        # 起降区域跳过
        near_sender   = (abs(p['lon'] - task.sender_lon)   < _EXCL_DEG and
                         abs(p['lat'] - task.sender_lat)   < _EXCL_DEG)
        near_receiver = (abs(p['lon'] - task.receiver_lon) < _EXCL_DEG and
                         abs(p['lat'] - task.receiver_lat) < _EXCL_DEG)
        if near_sender or near_receiver:
            continue

        nx1, ny1, nh1 = _predict(path[i], path[i + 1], PREDICT_1S)
        nx5, ny5, nh5 = _predict(path[i], path[i + 1], PREDICT_5S)

        for b in obstacles:
            bounds = b.get_bounds()
            cx  = (bounds['min_lon'] + bounds['max_lon']) / 2 * 111000.0 * np.cos(np.radians(lat_ref))
            cy  = (bounds['min_lat'] + bounds['max_lat']) / 2 * 111000.0
            ob_r = max(
                (bounds['max_lon'] - bounds['min_lon']) * 111000.0 * np.cos(np.radians(lat_ref)),
                (bounds['max_lat'] - bounds['min_lat']) * 111000.0
            ) / 2
            ob_hmax = float(b.height)
            ob_hmin = 0.0    # 建筑从地面起

            # 1 s 预判：-800（对齐参考代码 reward_1s）
            dist_1s = np.sqrt((nx1 - cx)**2 + (ny1 - cy)**2)
            if dist_1s <= ob_r + safe_r_m and ob_hmin < nh1 <= ob_hmax:
                reward -= 800

            # 5 s 预判：-200（对齐参考代码 reward_5s）
            dist_5s = np.sqrt((nx5 - cx)**2 + (ny5 - cy)**2)
            if dist_5s <= ob_r + safe_r_m and ob_hmin < nh5 <= ob_hmax:
                reward -= 200

    return reward


def path_destination_reward(path: List[Dict], task: 'DeliveryTask', lat_ref: float = 32.0) -> float:
    """
    P3 航迹意图（到达目的地）。

    对齐 MultirotorUAV_4.py destination_reward：对路径中每对相邻航点
    (current → next_1s)，计算靠近目标的进度奖励并累加：
        reward_i = 5 × (dist_current - dist_next) / (dist_current + 1e-10) - 5
    路径每个航点都有"当前位置"，逐段累加即与实时控制版本等价。
    """
    if len(path) < 2:
        return 0.0

    kx  = 111000.0 * np.cos(np.radians(lat_ref))
    ky  = 111000.0
    dest_x = task.receiver_lon * kx
    dest_y = task.receiver_lat * ky
    dest_h = 0.0

    def _dist(wp):
        return np.sqrt((wp['lon'] * kx - dest_x) ** 2 +
                       (wp['lat'] * ky - dest_y) ** 2 +
                       (float(wp.get('height', 0)) - dest_h) ** 2)

    reward = 0.0
    for i in range(len(path) - 1):
        d_cur  = _dist(path[i])
        d_next = _dist(path[i + 1])
        if d_cur < 1e-6:
            continue
        reward += 5.0 * (d_cur - d_next) / (d_cur + 1e-10) - 5.0
    return reward


def path_battery_reward(drone: 'Drone', path: List[Dict],
                       wind_speed_m_s: float = 0.0, wind_direction_deg: float = None) -> float:
    """
    P4 电量消耗：基于路径耗电估算，考虑风；耗电越多惩罚越重。
    """
    use_wind = (wind_speed_m_s is not None and wind_speed_m_s > 0 and wind_direction_deg is not None)
    energy = drone.estimate_path_energy_wh(
        path,
        wind_speed_m_s=wind_speed_m_s if use_wind else None,
        wind_direction_deg=wind_direction_deg if use_wind else None,
    )
    total_wh = energy['total_wh']
    capacity = getattr(drone, 'battery_capacity_wh', MR_BATTERY_CAPACITY_WH)
    power_proxy = total_wh / (capacity + 1e-10)
    return -2.0 * power_proxy


def path_total_reward(path: List[Dict], task: 'DeliveryTask', drone: 'Drone',
                     obstacles: List, other_paths: List[List[Dict]],
                     wind_speed_m_s: float = 0.0, wind_direction_deg: float = None,
                     lat_ref: float = 32.0) -> float:
    """
    路径总奖励 = P1机间避碰 + P2避障 + P3航迹意图 + P4电量消耗。
    优先级越高惩罚值越大，与 MultirotorUAV_4 的 total_reward 一致。
    """
    # 订单很多时对 other_paths 采样以加速机间间隔检查
    n_other = len(other_paths)
    op = other_paths if n_other <= 80 else [other_paths[i] for i in np.random.choice(n_other, size=80, replace=False)]
    r1 = path_uav_interval_reward(path, op, safe_interval_m=50.0, lat_ref=lat_ref)
    r2 = path_obstacle_reward(path, obstacles, task, safe_r_m=SAFETY_MARGIN, lat_ref=lat_ref)
    r3 = path_destination_reward(path, task, lat_ref=lat_ref)
    r4 = path_battery_reward(drone, path, wind_speed_m_s=wind_speed_m_s, wind_direction_deg=wind_direction_deg)
    return r1 + r2 + r3 + r4


class Building:
    """建筑物类"""
    def __init__(self, building_id, lon, lat, height, polygon_vertices=None):
        self.building_id = building_id
        self.name = building_id  # 使用building_id作为名称
        self.lon = lon
        self.lat = lat
        self.height = height
        self.polygon_vertices = polygon_vertices  # 多边形顶点列表
        
    def get_bounds(self):
        """获取建筑物边界"""
        if self.polygon_vertices and len(self.polygon_vertices) > 0:
            # 如果有多边形顶点，使用实际边界
            lons = [v[0] for v in self.polygon_vertices]
            lats = [v[1] for v in self.polygon_vertices]
            return {
                'min_lon': min(lons),
                'max_lon': max(lons),
                'min_lat': min(lats),
                'max_lat': max(lats)
            }
        else:
            # 否则使用中心点周围的小区域
            margin = 0.0001  # 约10米
            return {
                'min_lon': self.lon - margin,
                'max_lon': self.lon + margin,
                'min_lat': self.lat - margin,
                'max_lat': self.lat + margin
            }

class DeliveryTask:
    """配送任务类"""
    def __init__(self, task_id, sender_lon, sender_lat, receiver_lon, receiver_lat):
        self.task_id = task_id
        self.sender_lon = sender_lon
        self.sender_lat = sender_lat
        self.receiver_lon = receiver_lon
        self.receiver_lat = receiver_lat


# ---------------------------------------------------------------------------
# 无人机定义：参考 FixedWingUAV_4.py（固定翼）与 MultirotorUAV_4.py（多旋翼）
# 仅保留与配送路径规划相关的身份与性能参数，不包含状态转移、动作空间等动力学
# ---------------------------------------------------------------------------
class Drone:
    """
    无人机定义（参考 FixedWingUAV / MultirotorUAV）。
    - craft_type: 固定翼 'pure_fixed_wing' / 'vtol_tiltrotor' 或 多旋翼 'multirotor'（一种即可）
    - 固定翼参数与 FixedWingUAV_4.py 对齐；多旋翼参数与 MultirotorUAV_4.py 对齐
    """
    # 固定翼默认参数（参考 FixedWingUAV_4.py）
    FIXED_WING_DEFAULTS = {
        'weight_N': 300,
        'min_speed': 9.5,
        # 15.6 m/s 是 FixedWingUAV_4.py 中的“经济巡航速度”(最小功率/最大续航速度)。
        # 配送任务应取“最优航程速度” = 3^0.25 x 经济巡航 ≈ 20.5 m/s（阻力模型：
        # P(V)=aV^3+b/V，诱导阻力=3x寄生阻力处航程最大），比经济巡航快 32%、
        # 功率只增 14%、航程反而更长，故采用 20.5 而非 15.6。
        'cruise_speed': 20.5,
        'max_speed': 51.0,           # pure_fixed_wing
        'vertical_speed': 0.0,       # pure_fixed_wing 无垂直速度
        'min_turn_radius': None,     # 由 calc_min_turn_radius 计算
    }
    # 多旋翼默认参数（参考 MultirotorUAV_4.py，只保留一种）
    MULTIROTOR_DEFAULTS = {
        'max_speed': 30.0,
        'vert_speed_range': (0.5, 3.0),
        'max_hover_height': 1000,
        'payload': 2.5,
    }
    # 电量相关默认（固定翼续航长、多旋翼较短）
    # 功率标定：固定翼 400W 对应 20.5 m/s 处的阻力模型功率（见上方注释）；
    # 多旋翼 900W 使 1000Wh 满电续航约 1.11h（~72km @18m/s），在 2 小时任务窗口内
    # 仍需充电，保持电量为有效约束（旧标定 150W/120Wh 在容量放大后会导致续航
    # 虚高、电量约束失效，故随容量同步上调功率，climb/descent/hover 按比例缩放）。
    BATTERY_DEFAULTS = {
        'fixed_wing': {
            'capacity_wh': FW_BATTERY_CAPACITY_WH,
            'cruise_power_w': 400.0,     # 巡航功率（20.5 m/s 最优航程速度）
            # 额外功率（叠加在巡航之上）：weight_N=300 的机体以 3 m/s 爬升，
            # 额外功率 = W·Vz/η = 300×3/0.7 ≈ 1286W，取 1000W 偏保守。
            'climb_power_w': 1000.0,
            'descent_power_w': 200.0,    # 下降维持姿态开销，取爬升额外功率的 1/5
            'hover_power_w': 500.0,      # 悬停功率（起降点）
            'turn_power_multiplier': 1.3, # 转弯/机动动作功率倍数
            # 协调转弯角速度 ω=g·tan(φ)/V，取 30° 坡度、20.5 m/s：≈15.8 °/s
            'turn_rate_deg_s': 15.0,
            # 能耗记账用爬升率：固定翼靠前飞小角度爬升，与运动学 vertical_speed=0 解耦
            'climb_rate_m_s': 3.0,
            'payload_power_factor': 0.15, # 每kg载重增加功率的系数（W/kg）
        },
        'multirotor': {
            'capacity_wh': MR_BATTERY_CAPACITY_WH,
            'cruise_power_w': 900.0,
            # climb/descent 在 estimate_path_energy_wh 里是「叠加在巡航之上的额外
            # 功率」，不应随 cruise_power_w 同比例放大。按物理推导：hover 1080W
            # 对应 ~6kg(60N) 机体，3 m/s 爬升的额外功率 = W·Vz/η = 60×3/0.6 = 300W。
            'climb_power_w': 300.0,
            'descent_power_w': 100.0,    # 下降维持姿态开销，取爬升额外功率的 1/3
            'hover_power_w': 1080.0,    # 多旋翼悬停耗电高
            'turn_power_multiplier': 1.5, # 多旋翼转弯更耗电
            'turn_rate_deg_s': 45.0,     # 多旋翼可原地偏航，转弯远快于固定翼
            'climb_rate_m_s': 3.0,       # 与 vert_speed_range 上限一致
            'payload_power_factor': 0.25, # 多旋翼载重影响更大
        },
    }
    # 起降点默认悬停时长（秒）；航点显式给出 hold_duration_s 时优先用后者
    DEFAULT_HOVER_SECONDS = 5.0

    def __init__(self, drone_id: int, callsign: str = None, craft_type: str = 'multirotor',
                 color: str = None, **kwargs):
        """
        :param drone_id: 无人机编号
        :param callsign: 呼号/名称（参考 FixedWingUAV.callsign），用于显示与图例
        :param craft_type: 机型 - 固定翼 'pure_fixed_wing'/'vtol_tiltrotor' 或 多旋翼 'multirotor'
        :param color: 绘图颜色
        :param kwargs: 覆盖默认性能参数；电量相关: battery_capacity_wh, soc, cruise_power_w
        """
        self.drone_id = drone_id
        self.callsign = callsign or f"UAV-{drone_id}"
        self.craft_type = craft_type
        self.color = color or '#333333'
        self.assigned_task_indices: List[int] = []

        # 固定翼（参考 FixedWingUAV_4.py）
        if craft_type in ('pure_fixed_wing', 'vtol_tiltrotor', 'hybrid_fixed_rotor'):
            params = self.FIXED_WING_DEFAULTS.copy()
            if craft_type != 'pure_fixed_wing':
                params['max_speed'] = 40.379
                params['vertical_speed'] = 4.0
            params.update({k: v for k, v in kwargs.items() if k in params or k == 'weight_N'})
            self.weight_N = params.get('weight_N', self.FIXED_WING_DEFAULTS['weight_N'])
            self.min_speed = params.get('min_speed', self.FIXED_WING_DEFAULTS['min_speed'])
            self.cruise_speed = params.get('cruise_speed', self.FIXED_WING_DEFAULTS['cruise_speed'])
            self.max_speed = params.get('max_speed', params['max_speed'])
            self.vertical_speed = params.get('vertical_speed', params['vertical_speed'])
            self.min_turn_radius = kwargs.get('min_turn_radius') or self._calc_min_turn_radius()
            self.mg = None
            self.payload = None
        else:
            # 多旋翼（参考 MultirotorUAV_4.py，一种机型）
            params = self.MULTIROTOR_DEFAULTS.copy()
            params.update({k: v for k, v in kwargs.items() if k in params})
            self.mg = kwargs.get('mg', 20)  # 重力 N，默认约 2kg
            self.max_speed = params.get('max_speed')
            self.vert_speed_range = params.get('vert_speed_range')
            self.max_hover_height = params.get('max_hover_height')
            self.payload = params.get('payload')
            self.uav_radius = kwargs.get('uav_radius', 0.15)
            self.weight_N = None
            self.cruise_speed = kwargs.get('cruise_speed', self.max_speed * 0.6)
            self.min_turn_radius = None
            # 显式给出垂直速度，取 vert_speed_range 上限；此前该属性缺失，
            # 各处 getattr(drone,'vertical_speed',4.0) 会回退到 4.0，与声明的
            # (0.5, 3.0) 范围矛盾。
            self.vertical_speed = kwargs.get('vertical_speed', max(self.vert_speed_range))

        # 电量（所有机型统一，含详细功率参数）
        bat = self.BATTERY_DEFAULTS['fixed_wing' if craft_type in ('pure_fixed_wing', 'vtol_tiltrotor', 'hybrid_fixed_rotor') else 'multirotor'].copy()
        self.battery_capacity_wh = kwargs.get('battery_capacity_wh', bat['capacity_wh'])
        self.soc = kwargs.get('soc', 1.0)  # 当前电量比例 0~1，默认满电
        self.cruise_power_w = kwargs.get('cruise_power_w', bat['cruise_power_w'])
        self.climb_power_w = kwargs.get('climb_power_w', bat['climb_power_w'])
        self.descent_power_w = kwargs.get('descent_power_w', bat['descent_power_w'])
        self.hover_power_w = kwargs.get('hover_power_w', bat['hover_power_w'])
        self.turn_power_multiplier = kwargs.get('turn_power_multiplier', bat['turn_power_multiplier'])
        self.turn_rate_deg_s = kwargs.get('turn_rate_deg_s', bat['turn_rate_deg_s'])
        self.climb_rate_m_s = kwargs.get('climb_rate_m_s', bat['climb_rate_m_s'])
        self.payload_power_factor = kwargs.get('payload_power_factor', bat['payload_power_factor'])
        # 当前载重（kg），用于计算载重对功率的影响
        self.current_payload_kg = kwargs.get('current_payload_kg', 0.0)

    def _calc_min_turn_radius(self):
        """固定翼最小转弯半径（参考 FixedWingUAV_4.py）"""
        if self.craft_type != 'pure_fixed_wing':
            return 500.0
        turn_radius = (50**2) / (9.81 * np.tan(30 * np.pi / 180))
        return round(turn_radius, 1)

    @property
    def name(self) -> str:
        """显示名称（与图例、说明中的 callsign 一致）"""
        return self.callsign

    @property
    def remaining_wh(self) -> float:
        """当前剩余电量 (Wh)"""
        return self.battery_capacity_wh * self.soc

    def estimate_energy_wh(self, distance_m: float) -> float:
        """
        简化版：仅根据飞行距离估算耗电量 (Wh)。
        注意：更精确的估算请使用 estimate_path_energy_wh(path)
        """
        speed = getattr(self, 'cruise_speed', 15.0)
        if speed <= 0:
            speed = 15.0
        time_h = (distance_m / 1000.0) / (speed * 3.6)  # m -> km, m/s -> km/h
        base_power = self.cruise_power_w + self.current_payload_kg * self.payload_power_factor
        return base_power * time_h

    def estimate_path_energy_wh(self, path: List[Dict],
                                wind_speed_m_s: float = None,
                                wind_direction_deg: float = None) -> Dict:
        """
        根据完整路径估算耗电量 (Wh)，考虑：
        - 爬升/下降的额外耗电
        - 悬停耗电（起降点）
        - 载重对功率的影响
        - 转弯、避障等机动动作的额外耗电
        - 风对巡航时间与功率的影响（逆风增耗、顺风减耗）
        
        :param path: 路径列表，每个点包含 lon, lat, height, action
        :param wind_speed_m_s: 风速 (m/s)，来自 wind.csv，None 表示无风
        :param wind_direction_deg: 风向（度，风“吹向”的方向），None 表示无风
        :return: {'total_wh': 总耗电, 'breakdown': {'cruise': ..., 'climb': ..., 'wind': ..., ...}}
        """
        if len(path) < 2:
            return {'total_wh': 0.0, 'breakdown': {}}
        
        breakdown = {
            'cruise': 0.0,      # 巡航（无风基准）
            'climb': 0.0,       # 爬升
            'descent': 0.0,     # 下降
            'hover': 0.0,       # 悬停
            'turn': 0.0,        # 转弯/机动
            'payload': 0.0,     # 载重额外耗电
            'wind': 0.0,        # 风导致的额外耗电（正=逆风多耗，负=顺风节省）
        }
        
        speed = getattr(self, 'cruise_speed', 15.0)
        if speed <= 0:
            speed = 15.0
        
        use_wind = (wind_speed_m_s is not None and wind_speed_m_s > 0 and
                    wind_direction_deg is not None)
        
        # 基础功率（含载重影响）
        base_power = self.cruise_power_w + self.current_payload_kg * self.payload_power_factor

        # ── 悬停：单独遍历所有航点 ────────────────────────────────────────────
        # 起飞点位于 path[0]，在下面的「航段」循环里永远不会作为 p2 出现，
        # 旧写法因此漏计了起飞悬停；机间时域避让插入的空中悬停点也带有
        # hold_duration_s，应按该真实时长计费而非固定 5 秒。
        for wp in path:
            act_wp = wp.get('action', '') or ''
            if '起飞点' in act_wp or '降落' in act_wp or '悬停' in act_wp:
                hold_s = wp.get('hold_duration_s', None)
                hover_s = float(hold_s) if hold_s is not None else self.DEFAULT_HOVER_SECONDS
                breakdown['hover'] += self.hover_power_w * max(hover_s, 0.0) / 3600.0

        # ── 各航段水平朝向（度）：用于按真实转弯角度计费转弯能耗 ──────────────
        seg_heading: List[Optional[float]] = []
        for i in range(len(path) - 1):
            q1, q2 = path[i], path[i + 1]
            hx = (q2['lon'] - q1['lon']) * 111000 * np.cos(np.radians(q1['lat']))
            hy = (q2['lat'] - q1['lat']) * 111000
            seg_heading.append(
                float(np.degrees(np.arctan2(hx, hy)) % 360.0)
                if np.hypot(hx, hy) > 0.1 else None
            )

        turn_rate = float(getattr(self, 'turn_rate_deg_s', 45.0)) or 45.0

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            
            # 计算水平距离（米）
            dlon_deg = p2['lon'] - p1['lon']
            dlat_deg = p2['lat'] - p1['lat']
            dlon = dlon_deg * 111000 * np.cos(np.radians(p1['lat']))
            dlat = dlat_deg * 111000
            dist_h = np.sqrt(dlon**2 + dlat**2)
            
            # 高度变化（米）
            dh = p2['height'] - p1['height']
            
            # 判断动作类型
            action = p2.get('action', '')
            is_takeoff = '起飞' in action or '爬升' in action
            is_landing = '降落' in action
            is_climb = dh > 1.0  # 高度增加超过1米
            is_descent = dh < -1.0  # 高度减少超过1米

            # 爬升/下降耗电：用能耗记账专用爬升率。固定翼运动学 vertical_speed=0
            # 只表示「不能原地垂直移动」，并不意味着爬升不耗电，旧写法用
            # vertical_speed 做除数会把固定翼的爬升能耗静默算成 0。
            v_speed = float(getattr(self, 'climb_rate_m_s', 3.0)) or 3.0
            if is_climb and not is_takeoff and v_speed > 0:
                climb_time_h = abs(dh) / (v_speed * 3600)
                if climb_time_h > 0:
                    breakdown['climb'] += self.climb_power_w * climb_time_h

            # 下降耗电（固定翼可能回收能量，多旋翼仍需耗电）
            if is_descent and not is_landing and v_speed > 0:
                descent_time_h = abs(dh) / (v_speed * 3600)
                if descent_time_h > 0:
                    breakdown['descent'] += abs(self.descent_power_w) * descent_time_h
            
            # 水平飞行段：时间与风影响
            if dist_h > 0.1:  # 忽略极小移动
                # 航段朝向（度，从北顺时针，与风向一致约定）
                heading_deg = (np.degrees(np.arctan2(dlon_deg * np.cos(np.radians(p1['lat'])), dlat_deg)) % 360)
                heading_rad = np.radians(heading_deg)
                wind_to_rad = np.radians(wind_direction_deg) if use_wind else 0.0
                
                # 风沿航向分量 (m/s)：正=顺风，负=逆风
                if use_wind:
                    wind_along = wind_speed_m_s * np.cos(wind_to_rad - heading_rad)
                    ground_speed = max(speed + wind_along, speed * 0.2)  # 避免除零或倒退
                    time_h = dist_h / (ground_speed * 3600)
                    # 功率系数：逆风需加大推力，顺风可减
                    wind_factor = np.clip(
                        1.0 + (WIND_POWER_FACTOR / speed) * (-wind_along),
                        0.5, 2.0
                    )
                else:
                    time_h = dist_h / (speed * 3600)
                    wind_factor = 1.0
                
                # 无风基准巡航耗电
                base_cruise_energy = self.cruise_power_w * (dist_h / (speed * 3600))
                breakdown['cruise'] += base_cruise_energy
                
                # 载重额外耗电（单独记录）
                payload_extra = self.current_payload_kg * self.payload_power_factor * time_h
                breakdown['payload'] += payload_extra
                
                # 风导致的耗电变化：实际耗电 = 基准 × 时间比 × 功率系数，差值记入 wind
                if use_wind:
                    cruise_with_wind = base_power * wind_factor * time_h
                    cruise_no_wind = base_power * (dist_h / (speed * 3600))
                    breakdown['wind'] += (cruise_with_wind - cruise_no_wind)
                
                # 转弯/机动额外耗电：只对航点处真实的转弯机动时长计费。
                # 旧写法按 action 文本命中「绕行」就给整条航段乘功率倍数，
                # 一条 5 km 的巡航腿会被多收 ~35 Wh，导致「避撞额外能耗」虚高一个量级。
                # 现按转弯角度 / 转弯角速度求机动时长，并以航段时长为上限。
                if i > 0 and seg_heading[i] is not None and seg_heading[i - 1] is not None:
                    d_psi = abs((seg_heading[i] - seg_heading[i - 1] + 180.0) % 360.0 - 180.0)
                    if d_psi > 1.0:
                        turn_time_h = min(d_psi / turn_rate / 3600.0, time_h)
                        breakdown['turn'] += (
                            base_power * (self.turn_power_multiplier - 1.0) * turn_time_h)
        
        total_wh = sum(breakdown.values())
        return {'total_wh': total_wh, 'breakdown': breakdown}

    def assign_tasks(self, task_indices: List[int]):
        """为该无人机分配订单（订单在任务列表中的索引）"""
        self.assigned_task_indices = list(task_indices)


_FLEET_DRONE_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#000075',
]


def build_fleet_micro_drones(
    num_uavs: int,
    num_fixed_wing: int = None,
) -> List['Drone']:
    """
    创建微观机队：前 num_fixed_wing 架固定翼，其余多旋翼。
    默认按 FLEET_NUM_FIXED_WING:FLEET_NUM_MULTIROTOR 比例缩放。
    """
    if num_fixed_wing is None:
        num_fixed_wing = fleet_fixed_wing_count(num_uavs)
    drones: List[Drone] = []
    for i in range(num_uavs):
        if i < num_fixed_wing:
            d = Drone(
                i + 1, callsign=f'FixedWing-{i + 1:03d}',
                craft_type='pure_fixed_wing', color=_FLEET_DRONE_COLORS[i % len(_FLEET_DRONE_COLORS)],
            )
        else:
            d = Drone(
                i + 1, callsign=f'Multirotor-{i + 1:03d}',
                craft_type='multirotor', color=_FLEET_DRONE_COLORS[i % len(_FLEET_DRONE_COLORS)],
            )
        d.current_payload_kg = 1.0
        drones.append(d)
    return drones


class DronePathPlanner:
    """无人机路径规划器"""

    # 平坦地球近似缩放因子（经度→度，在南京纬度附近）
    _KD_LAT_REF = 32.0
    _KD_CX = 111000.0 * np.cos(np.radians(_KD_LAT_REF))
    _KD_CY = 111000.0

    def __init__(self, buildings_df):
        self.buildings_df = buildings_df
        self.buildings = self._load_buildings()
        # KD-Tree：以伪米坐标预构建，加速区域建筑查询（O(log n) 替代 O(n) 遍历）
        if self.buildings:
            bxy = np.array(
                [[b.lon * self._KD_CX, b.lat * self._KD_CY] for b in self.buildings],
                dtype=np.float64,
            )
            self._bkd: KDTree = KDTree(bxy)
        else:
            self._bkd = None

    def _load_buildings(self) -> List[Building]:
        """加载建筑物数据"""
        buildings = []
        for _, row in self.buildings_df.iterrows():
            # 解析多边形顶点
            polygon_vertices = None
            if 'polygon_vertices' in row and pd.notna(row['polygon_vertices']):
                try:
                    # polygon_vertices格式: "[(lon1,lat1),(lon2,lat2),...]"
                    vertices_str = row['polygon_vertices'].strip('[]')
                    vertices = []
                    for vertex in vertices_str.split('),('):
                        vertex = vertex.strip('()')
                        lon, lat = map(float, vertex.split(','))
                        vertices.append((lon, lat))
                    polygon_vertices = vertices
                except:
                    polygon_vertices = None

            building = Building(
                building_id=row['building_id'],
                lon=row['longitude'],
                lat=row['latitude'],
                height=row['height'],
                polygon_vertices=polygon_vertices
            )
            buildings.append(building)
        return buildings

    def get_obstacles_in_region(self, task: DeliveryTask) -> List[Building]:
        """获取起终点区域内的障碍物（KD-Tree 球形范围查询，O(log n)）"""
        if not self.buildings or self._bkd is None:
            return []

        # 计算任务区域中心和半径（度→米）
        center_lon = (task.sender_lon + task.receiver_lon) / 2.0
        center_lat = (task.sender_lat + task.receiver_lat) / 2.0
        half_dx = abs(task.receiver_lon - task.sender_lon) / 2.0 + DETECTION_RADIUS
        half_dy = abs(task.receiver_lat - task.sender_lat) / 2.0 + DETECTION_RADIUS
        # 将半径转换为米，取对角线长度作为球形半径（保守包含矩形区域）
        radius_m = float(np.hypot(half_dx * self._KD_CX, half_dy * self._KD_CY))

        center_pt = [center_lon * self._KD_CX, center_lat * self._KD_CY]
        idxs = self._bkd.query_ball_point(center_pt, r=radius_m)
        return [self.buildings[i] for i in idxs]
    
    def check_path_collision(self, start_lon, start_lat, end_lon, end_lat, 
                           height, building: Building) -> bool:
        """检查路径是否与建筑物碰撞"""
        # 如果飞行高度高于建筑物高度 + 安全距离，则不会碰撞
        if height > building.height + SAFETY_MARGIN:
            return False
        
        # 检查路径是否穿过建筑物边界
        bounds = building.get_bounds()
        
        # 简化的线段与矩形相交检测
        # 检查起点和终点是否在建筑物范围内
        start_in = (bounds['min_lon'] <= start_lon <= bounds['max_lon'] and
                   bounds['min_lat'] <= start_lat <= bounds['max_lat'])
        end_in = (bounds['min_lon'] <= end_lon <= bounds['max_lon'] and
                 bounds['min_lat'] <= end_lat <= bounds['max_lat'])
        
        if start_in or end_in:
            return True
        
        # 检查路径线段是否与建筑物矩形相交
        return self._line_rect_intersect(
            start_lon, start_lat, end_lon, end_lat,
            bounds['min_lon'], bounds['min_lat'],
            bounds['max_lon'], bounds['max_lat']
        )
    
    def _line_rect_intersect(self, x1, y1, x2, y2, rx1, ry1, rx2, ry2) -> bool:
        """线段与矩形相交检测"""
        # 使用参数方程检测
        def ccw(ax, ay, bx, by, cx, cy):
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
        
        # 检查线段是否与矩形的四条边相交
        edges = [
            (rx1, ry1, rx2, ry1),  # 底边
            (rx2, ry1, rx2, ry2),  # 右边
            (rx2, ry2, rx1, ry2),  # 顶边
            (rx1, ry2, rx1, ry1),  # 左边
        ]
        
        for ex1, ey1, ex2, ey2 in edges:
            if (ccw(x1, y1, ex1, ey1, ex2, ey2) != ccw(x2, y2, ex1, ey1, ex2, ey2) and
                ccw(x1, y1, x2, y2, ex1, ey1) != ccw(x1, y1, x2, y2, ex2, ey2)):
                return True
        
        return False

    def obstacles_on_straight_route(
        self, task: DeliveryTask, cruise_height: float = None,
    ) -> List[Building]:
        """
        与 plan_path 第 3 步相同：区域内除起降点建筑外，直飞线段在 cruise_height
        下与建筑物发生冲突的建筑列表（水平线段与楼块相交且高度未高出楼体+安全裕度）。

        RL（QDAC/SAC 等）生成航迹时亦用此列表统计「任务直飞是否需规避建筑」，
        与规则规划器口径一致；不因 rl_path 存在而强行记 obstacles=0。
        """
        h = float(CRUISE_HEIGHT if cruise_height is None else cruise_height)
        obstacles = self.get_obstacles_in_region(task)
        start_building_threshold = 0.0001
        out: List[Building] = []
        for building in obstacles:
            is_start = (
                abs(building.lon - task.sender_lon) < start_building_threshold
                and abs(building.lat - task.sender_lat) < start_building_threshold)
            is_end = (
                abs(building.lon - task.receiver_lon) < start_building_threshold
                and abs(building.lat - task.receiver_lat) < start_building_threshold)
            if is_start or is_end:
                continue
            if self.check_path_collision(
                task.sender_lon, task.sender_lat,
                task.receiver_lon, task.receiver_lat,
                h, building,
            ):
                out.append(building)
        return out
    
    def plan_path(self, task: DeliveryTask, verbose: bool = True) -> Dict:
        """规划配送路径。verbose=False 时减少打印，适用于大批量订单。"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"开始规划任务 {task.task_id}")
            print(f"起点: ({task.sender_lon:.6f}, {task.sender_lat:.6f})")
            print(f"终点: ({task.receiver_lon:.6f}, {task.receiver_lat:.6f})")
        
        # 1. 获取区域内的障碍物
        obstacles = self.get_obstacles_in_region(task)
        if verbose:
            print(f"\n检测到 {len(obstacles)} 个区域内建筑物")
        
        # 识别起点和终点的建筑物
        start_building = None
        end_building = None
        threshold = 0.0001  # 约10米
        for building in obstacles:
            if (abs(building.lon - task.sender_lon) < threshold and 
                abs(building.lat - task.sender_lat) < threshold):
                start_building = building
            if (abs(building.lon - task.receiver_lon) < threshold and 
                abs(building.lat - task.receiver_lat) < threshold):
                end_building = building
        
        if verbose:
            if start_building:
                print(f"起点建筑: {start_building.name} (高度 {start_building.height:.1f}米)")
            if end_building:
                print(f"终点建筑: {end_building.name} (高度 {end_building.height:.1f}米)")
        
        # 2. 初始化路径（起飞 -> 巡航 -> 降落）
        path = []
        current_height = CRUISE_HEIGHT
        current_lon = task.sender_lon
        current_lat = task.sender_lat
        
        # 起飞阶段
        path.append({
            'lon': current_lon,
            'lat': current_lat,
            'height': 0,
            'action': '起飞点'
        })
        path.append({
            'lon': current_lon,
            'lat': current_lat,
            'height': CRUISE_HEIGHT,
            'action': f'爬升到{CRUISE_HEIGHT}米开始巡航'
        })
        
        # 3. 检查直线路径是否有障碍（与 obstacles_on_straight_route 逻辑一致）
        obstacles_on_path = self.obstacles_on_straight_route(
            task, cruise_height=current_height)
        
        if verbose:
            print(f"\n路径上需要避开的障碍物: {len(obstacles_on_path)} 个（已排除起降点建筑）")
        
        # 4. 避障策略 - 沿路径分段处理
        if len(obstacles_on_path) == 0:
            # 无障碍，直飞
            if verbose:
                print("[OK] 无障碍，直飞")
            path.append({
                'lon': task.receiver_lon,
                'lat': task.receiver_lat,
                'height': CRUISE_HEIGHT,
                'action': '直飞到目的地'
            })
        else:
            # 有障碍，需要避障
            # 按照障碍物到起点的距离排序，依次处理
            obstacles_sorted = sorted(obstacles_on_path, 
                                     key=lambda b: np.sqrt((b.lon - task.sender_lon)**2 + 
                                                          (b.lat - task.sender_lat)**2))
            
            for i, building in enumerate(obstacles_sorted):
                if verbose:
                    print(f"\n障碍物 {i+1}: {building.name}")
                    print(f"  位置: ({building.lon:.6f}, {building.lat:.6f})")
                    print(f"  高度: {building.height:.1f}米")
                
                if building.height < BUILDING_THRESHOLD:
                    # 低于120米，改变高度飞越（严格对齐到 60/90/120 层）
                    # 规则：建筑 ≤60m→60m, (60,90]→90m, (90,120)→120m
                    new_height = snap_to_layer(building.height)
                    new_height = min(new_height, ALT_MAX_M)   # 强制 ≤120m
                    height_diff = max(0.0, new_height - current_height)
                    if verbose:
                        print(f"  策略: 边飞边升到 {new_height:.1f}m 飞越"
                              f"（建筑 {building.height:.1f}m → 合规层 {new_height:.0f}m）")

                    # ── 边飞边升：在建筑前方计算"开始爬升点"，使爬升段为斜线 ──
                    # 爬升率 1:5（垂直:水平），爬升角约 11°，保证轨迹平滑
                    if height_diff > 0:
                        dir_lon = building.lon - current_lon
                        dir_lat = building.lat - current_lat
                        dist_m = np.sqrt(
                            (dir_lon * 111000.0 * np.cos(np.radians(current_lat))) ** 2
                            + (dir_lat * 111000.0) ** 2
                        )
                        if dist_m > 1.0:
                            climb_h_dist_m = height_diff * 5.0   # 1:5 比例
                            if dist_m > climb_h_dist_m:
                                # 离建筑足够远：先水平飞到"爬升起点"，再斜飞到飞越点
                                ratio = 1.0 - climb_h_dist_m / dist_m
                                pre_lon = current_lon + dir_lon * ratio
                                pre_lat = current_lat + dir_lat * ratio
                                path.append({
                                    'lon': pre_lon,
                                    'lat': pre_lat,
                                    'height': current_height,
                                    'action': f'飞向{building.name}（准备爬升）'
                                })
                                current_lon = pre_lon
                                current_lat = pre_lat
                            # else: 离建筑太近，直接斜飞（爬升角较陡，仍保证安全）

                    # 飞越障碍物（高度从上一点线性爬升到 new_height ——无原地垂直段）
                    path.append({
                        'lon': building.lon,
                        'lat': building.lat,
                        'height': new_height,
                        'action': f'飞越{building.name}'
                    })
                    current_lon = building.lon
                    current_lat = building.lat
                    current_height = new_height
                    
                else:
                    # 超过120米，改变轨迹绕行
                    if verbose:
                        print(f"  策略: 改变轨迹，绕行")
                    
                    # 计算绕行路径（向路径的垂直方向偏移）
                    # 计算起点到终点的方向向量
                    dx = task.receiver_lon - task.sender_lon
                    dy = task.receiver_lat - task.sender_lat
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    # 垂直向量（逆时针旋转90度）
                    perp_x = -dy / dist
                    perp_y = dx / dist
                    
                    # 绕行偏移量（根据建筑物边界尺寸）
                    bounds = building.get_bounds()
                    w_deg = bounds['max_lon'] - bounds['min_lon']
                    h_deg = bounds['max_lat'] - bounds['min_lat']
                    offset = max(w_deg, h_deg) * 2  # 度，加倍以保证安全
                    
                    # 绕行航点（建筑物侧面）
                    detour_lon = building.lon + perp_x * offset
                    detour_lat = building.lat + perp_y * offset
                    
                    path.append({
                        'lon': detour_lon,
                        'lat': detour_lat,
                        'height': current_height,
                        'action': f'绕行避开{building.name}'
                    })
                    current_lon = detour_lon
                    current_lat = detour_lat
            
            # 飞向目的地
            path.append({
                'lon': task.receiver_lon,
                'lat': task.receiver_lat,
                'height': current_height,
                'action': '到达目的地上空'
            })
        
        # 降落阶段
        path.append({
            'lon': task.receiver_lon,
            'lat': task.receiver_lat,
            'height': 0,
            'action': '降落'
        })
        
        # 5. 计算路径信息
        total_distance = self._calculate_path_distance(path)
        
        result = {
            'task_id': task.task_id,
            'path': path,
            'obstacles_detected': len(obstacles),
            'obstacles_on_path': len(obstacles_on_path),
            'obstacles_on_path_list': obstacles_on_path,   # 供外部做冲突验证
            'total_distance': total_distance
        }
        
        if verbose:
            print(f"\n路径规划完成:")
            print(f"  总航程: {total_distance:.2f}米")
            print(f"  航点数: {len(path)}")
            print(f"{'='*60}\n")
        
        return result
    
    def _calculate_path_distance(self, path: List[Dict]) -> float:
        """计算路径总距离（米）"""
        total_distance = 0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            
            # 水平距离（简化：1度≈111km）
            dlon = (p2['lon'] - p1['lon']) * 111000 * np.cos(np.radians(p1['lat']))
            dlat = (p2['lat'] - p1['lat']) * 111000
            dheight = p2['height'] - p1['height']
            
            distance = np.sqrt(dlon**2 + dlat**2 + dheight**2)
            total_distance += distance
        
        return total_distance
    
    def visualize_path(self, task: DeliveryTask, result: Dict):
        """可视化路径 - 三维视图"""
        # 创建3D图形，增加尺寸以容纳所有内容
        fig = plt.figure(figsize=(20, 9))
        
        # 左图：真正的3D视图
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title(f'任务 {task.task_id} - 三维飞行路径', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('经度 (Longitude)', fontsize=11, labelpad=10)
        ax1.set_ylabel('纬度 (Latitude)', fontsize=11, labelpad=10)
        ax1.set_zlabel('高度 (米)', fontsize=11, labelpad=10)
        
        # 获取任务区域内的建筑物
        obstacles = self.get_obstacles_in_region(task)
        
        # 绘制建筑物为3D立方体
        for building in obstacles:
            bounds = building.get_bounds()
            
            # 建筑物底面四个角
            x_min, x_max = bounds['min_lon'], bounds['max_lon']
            y_min, y_max = bounds['min_lat'], bounds['max_lat']
            z_min, z_max = 0, building.height
            
            # 定义立方体的8个顶点
            vertices = [
                [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],  # 底面
                [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max],  # 顶面
            ]
            
            # 定义立方体的6个面
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
            ]
            
            # 根据建筑高度选择颜色
            color = 'red' if building.height >= BUILDING_THRESHOLD else 'orange'
            alpha = 0.5 if building.height >= BUILDING_THRESHOLD else 0.3
            
            # 添加3D多边形集合
            poly3d = Poly3DCollection(faces, facecolors=color, linewidths=0.5, 
                                     edgecolors='black', alpha=alpha)
            ax1.add_collection3d(poly3d)
            
            # 标注高度（只标注高建筑）
            if building.height >= 100:
                ax1.text(building.lon, building.lat, building.height + 10, 
                        f'{building.height:.0f}m',
                        ha='center', va='bottom', fontsize=8, 
                        color='darkred', fontweight='bold')
        
        # 绘制飞行路径
        path = result['path']
        lons = [p['lon'] for p in path]
        lats = [p['lat'] for p in path]
        heights = [p['height'] for p in path]
        
        # 绘制路径线
        ax1.plot(lons, lats, heights, 'b-', linewidth=3, label='飞行路径', zorder=100)
        
        # 绘制航点
        ax1.scatter(lons, lats, heights, c='blue', s=100, marker='o', 
                   edgecolors='white', linewidths=2, zorder=101)
        
        # 标注起点和终点（特殊显示）
        ax1.scatter([task.sender_lon], [task.sender_lat], [0], 
                   c='green', s=300, marker='o', edgecolors='white', 
                   linewidths=2, label='起点', zorder=102)
        ax1.scatter([task.sender_lon], [task.sender_lat], [CRUISE_HEIGHT], 
                   c='green', s=200, marker='^', edgecolors='white', 
                   linewidths=2, zorder=102)
        
        ax1.scatter([task.receiver_lon], [task.receiver_lat], [heights[-2]], 
                   c='red', s=200, marker='v', edgecolors='white', 
                   linewidths=2, zorder=102)
        ax1.scatter([task.receiver_lon], [task.receiver_lat], [0], 
                   c='red', s=300, marker='*', edgecolors='white', 
                   linewidths=2, label='终点', zorder=102)
        
        # 标注航点编号
        for i, (lon, lat, h) in enumerate(zip(lons, lats, heights)):
            ax1.text(lon, lat, h + 15, f'P{i}', 
                    ha='center', va='bottom', fontsize=10, 
                    fontweight='bold', color='blue',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                             alpha=0.7, edgecolor='blue', linewidth=1.5))
        
        # 添加参考平面
        # 巡航高度参考面
        xx, yy = np.meshgrid(
            np.linspace(min(lons), max(lons), 2),
            np.linspace(min(lats), max(lats), 2)
        )
        zz = np.ones_like(xx) * CRUISE_HEIGHT
        ax1.plot_surface(xx, yy, zz, alpha=0.1, color='green', label=f'巡航高度 {CRUISE_HEIGHT}m')
        
        # 地面参考面
        zz_ground = np.zeros_like(xx)
        ax1.plot_surface(xx, yy, zz_ground, alpha=0.05, color='gray')
        
        # 调整图例位置和样式，避免被裁剪
        legend = ax1.legend(loc='upper left', fontsize=9, framealpha=0.9, 
                           edgecolor='black', fancybox=True, shadow=True,
                           bbox_to_anchor=(0.02, 0.98))
        legend.get_frame().set_linewidth(1.5)
        
        ax1.view_init(elev=25, azim=45)  # 设置视角
        
        # 右图：多视角组合
        ax2 = fig.add_subplot(122)
        ax2.set_title(f'任务 {task.task_id} - 多视角投影', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # 创建子图网格
        gs = ax2.get_subplotspec().subgridspec(2, 2)
        
        # 俯视图（XY平面）
        ax_top = fig.add_subplot(gs[0, 0])
        ax_top.set_title('俯视图 (XY平面)', fontsize=10, fontweight='bold')
        ax_top.set_xlabel('经度', fontsize=9)
        ax_top.set_ylabel('纬度', fontsize=9)
        ax_top.grid(True, alpha=0.3)
        
        # 绘制建筑物俯视图
        for building in obstacles:
            bounds = building.get_bounds()
            width = bounds['max_lon'] - bounds['min_lon']
            height = bounds['max_lat'] - bounds['min_lat']
            color = 'red' if building.height >= BUILDING_THRESHOLD else 'orange'
            alpha = 0.6 if building.height >= BUILDING_THRESHOLD else 0.4
            rect = Rectangle((bounds['min_lon'], bounds['min_lat']), width, height,
                           facecolor=color, edgecolor='black', alpha=alpha, linewidth=0.5)
            ax_top.add_patch(rect)
        
        ax_top.plot(lons, lats, 'b-', linewidth=2, label='路径')
        ax_top.scatter([task.sender_lon], [task.sender_lat], c='green', s=100, 
                      marker='o', edgecolors='white', linewidths=1.5, zorder=10, label='起点')
        ax_top.scatter([task.receiver_lon], [task.receiver_lat], c='red', s=100, 
                      marker='*', edgecolors='white', linewidths=1.5, zorder=10, label='终点')
        ax_top.legend(fontsize=7, loc='best', framealpha=0.9)
        
        # 侧视图1（XZ平面）
        ax_side1 = fig.add_subplot(gs[0, 1])
        ax_side1.set_title('侧视图 (经度-高度)', fontsize=10, fontweight='bold')
        ax_side1.set_xlabel('经度', fontsize=9)
        ax_side1.set_ylabel('高度 (米)', fontsize=9)
        ax_side1.grid(True, alpha=0.3)
        
        # 绘制建筑物侧视图
        for building in obstacles:
            bounds = building.get_bounds()
            width = bounds['max_lon'] - bounds['min_lon']
            color = 'red' if building.height >= BUILDING_THRESHOLD else 'orange'
            alpha = 0.6 if building.height >= BUILDING_THRESHOLD else 0.4
            rect = Rectangle((bounds['min_lon'], 0), width, building.height,
                           facecolor=color, edgecolor='black', alpha=alpha, linewidth=0.5)
            ax_side1.add_patch(rect)
        
        ax_side1.plot(lons, heights, 'b-', linewidth=2, marker='o', markersize=5, label='飞行路径')
        ax_side1.axhline(y=CRUISE_HEIGHT, color='green', linestyle='--', 
                        label=f'巡航 {CRUISE_HEIGHT}m', alpha=0.5, linewidth=1.5)
        ax_side1.axhline(y=BUILDING_THRESHOLD, color='red', linestyle='--', 
                        label=f'阈值 {BUILDING_THRESHOLD}m', alpha=0.5, linewidth=1.5)
        ax_side1.legend(fontsize=6, loc='best', framealpha=0.9)
        
        # 侧视图2（YZ平面）
        ax_side2 = fig.add_subplot(gs[1, 0])
        ax_side2.set_title('侧视图 (纬度-高度)', fontsize=10, fontweight='bold')
        ax_side2.set_xlabel('纬度', fontsize=9)
        ax_side2.set_ylabel('高度 (米)', fontsize=9)
        ax_side2.grid(True, alpha=0.3)
        
        # 绘制建筑物侧视图
        for building in obstacles:
            bounds = building.get_bounds()
            width = bounds['max_lat'] - bounds['min_lat']
            color = 'red' if building.height >= BUILDING_THRESHOLD else 'orange'
            alpha = 0.6 if building.height >= BUILDING_THRESHOLD else 0.4
            rect = Rectangle((bounds['min_lat'], 0), width, building.height,
                           facecolor=color, edgecolor='black', alpha=alpha, linewidth=0.5)
            ax_side2.add_patch(rect)
        
        ax_side2.plot(lats, heights, 'b-', linewidth=2, marker='o', markersize=5, label='飞行路径')
        ax_side2.axhline(y=CRUISE_HEIGHT, color='green', linestyle='--', alpha=0.5, 
                        linewidth=1.5, label=f'巡航 {CRUISE_HEIGHT}m')
        ax_side2.axhline(y=BUILDING_THRESHOLD, color='red', linestyle='--', alpha=0.5, 
                        linewidth=1.5, label=f'阈值 {BUILDING_THRESHOLD}m')
        ax_side2.legend(fontsize=6, loc='best', framealpha=0.9)
        
        # 高度剖面图
        ax_profile = fig.add_subplot(gs[1, 1])
        ax_profile.set_title('高度变化剖面', fontsize=10, fontweight='bold')
        ax_profile.set_xlabel('航点序号', fontsize=9)
        ax_profile.set_ylabel('高度 (米)', fontsize=9)
        ax_profile.grid(True, alpha=0.3)
        
        waypoint_indices = range(len(heights))
        ax_profile.plot(waypoint_indices, heights, 'b-', linewidth=2.5, marker='o', 
                       markersize=8, label='高度变化')
        ax_profile.axhline(y=CRUISE_HEIGHT, color='green', linestyle='--', 
                          label=f'巡航{CRUISE_HEIGHT}m', alpha=0.6, linewidth=1.5)
        ax_profile.axhline(y=BUILDING_THRESHOLD, color='red', linestyle='--', 
                          label=f'阈值{BUILDING_THRESHOLD}m', alpha=0.6, linewidth=1.5)
        
        # 标注每个航点的高度
        for i, h in enumerate(heights):
            ax_profile.annotate(f'{h:.0f}m', (i, h), 
                              xytext=(0, 10), textcoords='offset points',
                              ha='center', fontsize=8, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
                                       alpha=0.8, edgecolor='blue'))
        
        ax_profile.set_xticks(waypoint_indices)
        ax_profile.set_xticklabels([f'P{i}' for i in waypoint_indices])
        ax_profile.legend(fontsize=7, loc='best', framealpha=0.9)
        
        # 使用tight_layout并增加边距，确保所有内容（包括图例）都能完整显示
        plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.98])
        
        # 保存时使用bbox_inches='tight'和pad_inches来确保不裁剪
        plt.savefig(f'task_{task.task_id}_path_3d.png', dpi=300, 
                   bbox_inches='tight', pad_inches=0.3)
        print(f"\n[OK] 三维路径图已保存: task_{task.task_id}_path_3d.png")
        plt.show()
        
        # 打印详细路径信息
        print(f"\n{'='*80}")
        print(f"详细路径信息 - 任务 {task.task_id}")
        print(f"{'='*80}")
        print(f"{'航点':<8}{'经度':<14}{'纬度':<14}{'高度(米)':<12}{'动作说明':<40}")
        print("-" * 80)
        for i, p in enumerate(path):
            print(f"P{i:<7}{p['lon']:<14.6f}{p['lat']:<14.6f}{p['height']:<12.1f}{p['action']:<40}")
        print(f"{'='*80}\n")

    def visualize_all_paths_3d(self, tasks: List[DeliveryTask], results: List[Dict],
                               drones: List[Drone]):
        """
        在一张三维图中绘制所有订单的飞行路径，并明确标出路线与无人机的对应关系。
        使用无人机定义（Drone）：名称、颜色由各 Drone 实例提供。
        """
        line_styles = ['-', '--']  # 同机第1条实线、第2条虚线
        # 由无人机定义得到：任务索引 -> (无人机, 该机内第几条)
        task_to_drone = {}
        for drone in drones:
            for local_idx, task_idx in enumerate(drone.assigned_task_indices):
                task_to_drone[task_idx] = (drone, local_idx)

        # 合并所有任务区域内的建筑物（去重）
        seen_ids = set()
        all_obstacles = []
        for task in tasks:
            for b in self.get_obstacles_in_region(task):
                if b.building_id not in seen_ids:
                    seen_ids.add(b.building_id)
                    all_obstacles.append(b)

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('三架无人机完成五个订单 - 三维飞行路径总览', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('经度 (Longitude)', fontsize=11, labelpad=10)
        ax.set_ylabel('纬度 (Latitude)', fontsize=11, labelpad=10)
        ax.set_zlabel('高度 (米)', fontsize=11, labelpad=10)

        # 绘制建筑物
        for building in all_obstacles:
            bounds = building.get_bounds()
            x_min, x_max = bounds['min_lon'], bounds['max_lon']
            y_min, y_max = bounds['min_lat'], bounds['max_lat']
            z_min, z_max = 0, building.height
            vertices = [
                [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max],
            ]
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[0], vertices[3], vertices[7], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
            ]
            color = 'red' if building.height >= BUILDING_THRESHOLD else 'orange'
            alpha = 0.5 if building.height >= BUILDING_THRESHOLD else 0.3
            poly3d = Poly3DCollection(faces, facecolors=color, linewidths=0.5,
                                      edgecolors='black', alpha=alpha)
            ax.add_collection3d(poly3d)

        # 绘制五条路径，按无人机定义（名称、颜色）标注对应关系
        all_lons, all_lats = [], []
        for idx, (task, result) in enumerate(zip(tasks, results)):
            drone, local_idx = task_to_drone[idx]
            c = drone.color or '#333333'
            ls = line_styles[local_idx % len(line_styles)]
            path = result['path']
            lons = [p['lon'] for p in path]
            lats = [p['lat'] for p in path]
            heights = [p['height'] for p in path]
            all_lons.extend(lons)
            all_lats.extend(lats)
            label = f'{drone.name}-订单{task.task_id}'
            ax.plot(lons, lats, heights, linestyle=ls, color=c, linewidth=2.5, label=label, zorder=100)
            ax.scatter(lons, lats, heights, c=c, s=40, marker='o', edgecolors='white', linewidths=0.5, zorder=101)
            ax.scatter([task.sender_lon], [task.sender_lat], [0], c=c, s=120, marker='o',
                      edgecolors='white', linewidths=1.5, zorder=102)
            ax.scatter([task.receiver_lon], [task.receiver_lat], [0], c=c, s=120, marker='*',
                      edgecolors='white', linewidths=1.5, zorder=102)

        # 巡航高度参考面（范围覆盖所有路径）
        if all_lons and all_lats:
            xx, yy = np.meshgrid(
                np.linspace(min(all_lons), max(all_lons), 2),
                np.linspace(min(all_lats), max(all_lats), 2)
            )
            zz = np.ones_like(xx) * CRUISE_HEIGHT
            ax.plot_surface(xx, yy, zz, alpha=0.08, color='green')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9, bbox_to_anchor=(0.02, 0.98))
        # 在图上加文字说明：路线与无人机对应关系（来自无人机定义）
        summary_lines = []
        for drone in drones:
            task_ids = [tasks[i].task_id for i in drone.assigned_task_indices]
            summary_lines.append(f"{drone.name} → 订单 {task_ids}")
        summary_text = '\n'.join(summary_lines)
        ax.text2D(0.02, 0.02, summary_text, transform=ax.transAxes, fontsize=10,
                  verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        ax.view_init(elev=25, azim=45)
        plt.tight_layout()
        plt.savefig('three_drones_five_orders_3d.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
        print("\n[OK] 三维总览图已保存: three_drones_five_orders_3d.png")
        plt.show()


def select_orders_in_one_region(delivery_df, n_orders=5, region_radius_deg=0.015):
    """
    从配送数据中选出 n_orders 个订单，使这些订单的起终点尽量集中在一块区域。
    策略：以全表起点的中心为参考，选取起终点都在该中心附近区域内的订单。
    """
    cx = delivery_df['sender_X'].median()
    cy = delivery_df['sender_Y'].median()
    # 计算每个订单的“最远点”到中心的距离（起终点谁离中心更远取谁）
    def max_dist_to_center(row):
        d1 = np.sqrt((row['sender_X'] - cx)**2 + (row['sender_Y'] - cy)**2)
        d2 = np.sqrt((row['receiver_X'] - cx)**2 + (row['receiver_Y'] - cy)**2)
        return max(d1, d2)
    delivery_df = delivery_df.copy()
    delivery_df['_dist'] = delivery_df.apply(max_dist_to_center, axis=1)
    # 取距离中心最近的 n_orders 个订单，保证集中在一块区域
    nearest = delivery_df.nsmallest(n_orders, '_dist')
    return nearest.drop(columns=['_dist'])


def main():
    """主函数：十架无人机完成全部订单"""
    NUM_DRONES = 10

    # 1. 加载数据
    print("正在加载数据...")
    buildings_df = pd.read_csv('nanjing_buildings.csv')
    delivery_df = pd.read_csv('nanjing_delivery_700_2026_01.csv')
    print(f"[OK] 已加载 {len(buildings_df)} 个建筑物")
    print(f"[OK] 已加载 {len(delivery_df)} 个配送任务")

    # 2. 使用全部订单
    orders_subset = delivery_df
    NUM_ORDERS = len(orders_subset)
    print(f"\n[OK] 使用全部 {NUM_ORDERS} 个订单")

    # 3. 创建路径规划器
    planner = DronePathPlanner(buildings_df)

    # 4. 十架无人机定义，10 种颜色
    DRONE_COLORS = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    drones = build_fleet_micro_drones(NUM_DRONES)
    # 轮询分配：订单索引 i 分配给无人机 i % 10
    for d in range(NUM_DRONES):
        indices = [i for i in range(NUM_ORDERS) if i % NUM_DRONES == d]
        drones[d].assign_tasks(indices)
    for drone in drones:
        drone.current_payload_kg = 1.0   # 每次飞行只携带 1 单包裹（1kg）
    print("\n[无人机定义] 十架无人机，轮询分配全部订单")
    for d in drones:
        n = len(d.assigned_task_indices)
        print(f"  {d.callsign}: {n} 单, 电量={d.battery_capacity_wh:.0f}Wh, 单次载重=1.0kg")

    all_tasks = []
    all_results = []
    for _, row in orders_subset.iterrows():
        t = DeliveryTask(
            task_id=row['task_id'],
            sender_lon=row['sender_X'],
            sender_lat=row['sender_Y'],
            receiver_lon=row['receiver_X'],
            receiver_lat=row['receiver_Y']
        )
        all_tasks.append(t)
        all_results.append(None)

    # 大批量时减少单任务打印
    verbose_plan = NUM_ORDERS <= 20

    # 5. 按无人机依次规划路径
    for di, drone in enumerate(drones):
        n_tasks = len(drone.assigned_task_indices)
        print(f"\n{drone.callsign}: 规划 {n_tasks} 个订单...")
        for k, idx in enumerate(drone.assigned_task_indices):
            if verbose_plan or k == 0 or k == n_tasks - 1:
                result = planner.plan_path(all_tasks[idx], verbose=verbose_plan)
            else:
                result = planner.plan_path(all_tasks[idx], verbose=False)
            all_results[idx] = result
        print(f"  {drone.callsign} 完成 {n_tasks} 条路径")

    # 5.0 加载风场（用于电量估算）
    wind_speed_m_s, wind_direction_deg = load_wind_from_csv('wind.csv')
    if wind_speed_m_s > 0:
        print(f"\n[风场] 已加载 wind.csv: 风速 {wind_speed_m_s:.2f} m/s, 吹向 {wind_direction_deg:.1f}° (影响电量估算)")
    else:
        print("\n[风场] 未使用风场数据，按无风估算电量")

    # 5.1 电量检查：使用详细版路径分析（含风）
    print("\n" + "="*60)
    print("[电量检查 - 详细版路径分析]")
    print("="*60)
    show_breakdown = NUM_ORDERS <= 15  # 订单过多时只打印汇总
    for drone in drones:
        n_tasks    = len(drone.assigned_task_indices)
        total_dist = sum(all_results[i]['total_distance'] for i in drone.assigned_task_indices)

        # 逐单计算耗电，找最大单次与均值（无人机每单送完后充电，应按单次比较）
        per_task_wh: list = []
        for idx in drone.assigned_task_indices:
            e = drone.estimate_path_energy_wh(
                all_results[idx]['path'],
                wind_speed_m_s=wind_speed_m_s if wind_speed_m_s > 0 else None,
                wind_direction_deg=wind_direction_deg if wind_speed_m_s > 0 else None,
            )
            per_task_wh.append(e['total_wh'])

        total_wh   = sum(per_task_wh)
        max_wh     = max(per_task_wh) if per_task_wh else 0.0
        avg_wh     = total_wh / n_tasks if n_tasks else 0.0
        capacity   = drone.battery_capacity_wh
        remain_min = capacity - max_wh   # 最危险单次的剩余电量
        ok         = remain_min >= 0

        print(f"\n{drone.callsign} ({n_tasks} 单): 总航程 {total_dist:.0f}m"
              f" | 累计耗电 {total_wh:.1f}Wh"
              f" | 单次均值 {avg_wh:.1f}Wh  最大 {max_wh:.1f}Wh"
              f" | 电池 {capacity:.0f}Wh"
              f" | 最危险单次剩余 {remain_min:.1f}Wh → {'✓ 充足' if ok else '✗ 不足'}")

    # 5.2 路径奖励评估（P1机间避碰 > P2避障 > P3航迹意图 > P4电量）
    print("\n" + "="*60)
    print("[路径奖励评估]")
    print("="*60)
    lat_ref = float(np.mean([t.sender_lat for t in all_tasks])) if all_tasks else 32.0
    rewards_per_order = []
    for idx in range(len(all_tasks)):
        task = all_tasks[idx]
        path = all_results[idx]['path']
        drone = next((d for d in drones if idx in d.assigned_task_indices), drones[0])
        obstacles = planner.get_obstacles_in_region(task)
        other_paths = [all_results[j]['path'] for j in range(len(all_tasks)) if j != idx]
        r = path_total_reward(
            path, task, drone, obstacles, other_paths,
            wind_speed_m_s=wind_speed_m_s, wind_direction_deg=wind_direction_deg,
            lat_ref=lat_ref,
        )
        rewards_per_order.append(r)
        if NUM_ORDERS <= 25:
            print(f"  订单 {task.task_id} ({drone.callsign}): 路径总奖励 = {r:.2f}")
    if NUM_ORDERS > 25:
        print(f"  共 {NUM_ORDERS} 单, 路径总奖励 均值 = {np.mean(rewards_per_order):.2f}, 最小 = {np.min(rewards_per_order):.2f}, 最大 = {np.max(rewards_per_order):.2f}")

    # 6. 可视化：订单数较少时输出 3D 总览图，否则仅保存 2D 俯视图
    if NUM_ORDERS <= 30:
        planner.visualize_all_paths_3d(all_tasks, all_results, drones)
    else:
        # 订单过多时只画 2D 俯视图并保存，不弹窗
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_title(f'十架无人机完成全部 {NUM_ORDERS} 单 - 俯视图', fontsize=14, fontweight='bold')
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        ax.grid(True, alpha=0.3)
        DRONE_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for idx, (task, result) in enumerate(zip(all_tasks, all_results)):
            path = result['path']
            lons = [p['lon'] for p in path]
            lats = [p['lat'] for p in path]
            drone = next((d for d in drones if idx in d.assigned_task_indices), drones[0])
            c = drone.color
            ax.plot(lons, lats, '-', color=c, linewidth=0.4, alpha=0.7)
        ax.legend([d.callsign for d in drones], loc='upper left', fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig('ten_drones_all_orders_overview.png', dpi=150, bbox_inches='tight', pad_inches=0.3)
        print(f"\n[OK] 俯视图已保存: ten_drones_all_orders_overview.png (共 {NUM_ORDERS} 条路径)")
        plt.close()


if __name__ == "__main__":
    main()
