"""
sac_path_env.py  —  SAC 路径规划 Gym 环境
==========================================
单架无人机从起点飞往终点，动作驱动航向与高度调整。

【高度层规则（与主流程对齐）】
  飞行高度严格落在 {60, 90, 120}m 三个合规层。step() 内通过 snap_to_layer() 实现：
  · action[1] > 0（任意正高度变化）→ 抬升到上一层（60→90 / 90→120）
  · action[1] ≤ -0.5（delta ≤ -30m）→ 降到下一层（120→90 / 90→60）
  · 其他 → 保持当前层不变
  P1 判定：水平净距 < 50m 且 垂直差 < 30m ⟺ 同层 + 水平不足，训练/推理口径统一。

奖励函数（与 drone_obstacle_avoidance.py 的 path_total_reward 对齐 P1~P4，
增量版 —— 每一步只对"新增段"评分，避免重复计分）：
  · P1 机间避碰：当前段 0.5s 预判命中他机 → -2000 × SCALE
  · P2 建筑避障：当前段 1s 预判命中 -800 × SCALE；5s 预判命中 -200 × SCALE
  · P3 航迹意图：增量 = 5 × (prev_dist - curr_dist) / init_dist
  · P4 电量消耗：段 ΔWh → -2 × ΔWh / capacity（与 path_battery_reward 累加等价）
                 使用 task.dronetime 从 wind_lookup 查任务时刻的时变风
  · 时间惩罚   ：-0.1 / step
  · 到达奖励   ：+50；超时惩罚：-20 × 剩余进度

SCALE = 0.01  使 P1/P2 的量级压入 [-20, -2]，与 SAC Q 值动态范围匹配。
P3/P4 原本就在合适量级，不做缩放。
去掉了旧版 "alignment × 0.5" 的航向对齐 shaping —— 该项在兜圈时也能
刷出正奖励，是旧模型策略坍缩到常数"持续右转 45°"的根因。

状态维度  : STATE_DIM  = 16  （SAC/DDPG/QDAC 统一）
动作维度  : ACTION_DIM = 2   (连续, [-1, 1])

状态 16 维（归一化到 [-1,1]）
  目标相对位置 (3)        : [0] dx [1] dy [2] dz
  最近建筑信息 (4)        : [3] nb_dist [4] nb_cos [5] nb_sin [6] nb_hdiff
  最近他机信息 (3)        : [7] nu_dist [8] nu_cos [9] nu_sin
  机型标识 (1)           : [10] is_fixed_wing  (0=多旋翼, 1=固定翼/VTOL)
  风速向量 (2)           : [11] wind_x  [12] wind_y   (ws × cos/sin(wind_to_deg) / MAX_WIND)
  当前航向 (2)           : [13] heading_cos [14] heading_sin
  剩余电量 (1)           : [15] soc     (1 − 累计Wh / capacity)

动作向量（2 维连续，[-1, 1]）：
  [0]  heading_change: 映射到 [-MAX_HEADING_CHANGE, +MAX_HEADING_CHANGE] 度
  [1]  height_change : 映射到 [-MAX_HEIGHT_CHANGE,  +MAX_HEIGHT_CHANGE]  米
                       经 snap_to_layer 离散化 → 实际效果为 ±30m 层跳变
"""

import numpy as np
import os, sys
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drone_obstacle_avoidance import (
    Building, DeliveryTask, Drone, WindTimeseries,
    CRUISE_HEIGHT, SAFETY_MARGIN, WIND_POWER_FACTOR,
    snap_to_layer, ALT_LAYERS,
    MR_BATTERY_CAPACITY_WH,
)

# ══════════════════════════════════════════════════════════════════════════════
# 环境超参数
# ══════════════════════════════════════════════════════════════════════════════
STEP_DIST_M    = 150.0   # 每步飞行水平距离（米）——加大步长，长任务可在 MAX_STEPS 内到达
MAX_STEPS      = 400     # 每 episode 最大步数（150m×400=60km，覆盖所有南京任务）
MAX_DIST_M     = 25000.0 # 归一化参考距离（米）
MAX_ALT_M      = 120.0   # 最大飞行高度（米）= 最高合规层
MIN_ALT_M      = 60.0    # 最低飞行高度（米）= 最低合规层
GOAL_THRESH_M  = 200.0   # 到达目标判定半径（米，放宽使学习更容易启动）
OBS_SAFE_M     = SAFETY_MARGIN          # 建筑安全距离（米）
UAV_SAFE_M     = 50.0                   # 机间水平安全距离（米）
UAV_VERT_SAFE_M = 30.0                  # 机间垂直安全间距（米）= 最小高度分层量
MAX_HEADING_CHANGE = 45.0               # 每步最大航向变化（度，限制剧烈转向）
MAX_HEIGHT_CHANGE  = 60.0               # 每步最大高度变化（米）

# ── 对齐 drone_obstacle_avoidance 的预判参数 ────────────────────────────────
CRUISE_SPEED_MS = 15.0                   # 巡航速度（与主程序一致）
PREDICT_1S_M    = CRUISE_SPEED_MS * 1.0  # P2 建筑 1s 预判距离 = 15 m
PREDICT_5S_M    = CRUISE_SPEED_MS * 5.0  # P2 建筑 5s 预判距离 = 75 m
REACTION_S      = 12.5                   # P1 机间反应时间（对齐 MultirotorUAV_4.py 的 12.5s）
PREDICT_UAV_M   = CRUISE_SPEED_MS * REACTION_S  # = 187.5 m（检查到航段末端附近）
UAV_RADIUS_M    = 1.0                    # 双机半径
EXCL_DEG        = 0.0005                 # 起降排除经纬度 ≈ 55 m
REWARD_SCALE    = 0.01                   # P1/P2 量级缩放：-2000→-20 / -800→-8 / -200→-2

STATE_DIM     = 16   # 统一：目标3 + 建筑4 + 他机3 + 机型1 + 风2 + 航向2 + soc1
ACTION_DIM    = 2
MAX_WIND_MS   = 20.0  # 风速归一化参考（m/s，南京极端风速上限）


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _m2deg_lon(dx_m: float, lat_ref: float) -> float:
    return dx_m / (111000.0 * np.cos(np.radians(lat_ref)))

def _m2deg_lat(dy_m: float) -> float:
    return dy_m / 111000.0

def _deg2m_x(dlon: float, lat_ref: float) -> float:
    return dlon * 111000.0 * np.cos(np.radians(lat_ref))

def _deg2m_y(dlat: float) -> float:
    return dlat * 111000.0


# ══════════════════════════════════════════════════════════════════════════════
# 环境类
# ══════════════════════════════════════════════════════════════════════════════

class SACPathEnv:
    """
    单架无人机路径规划环境（Gym 风格，不依赖 gym 包）。

    状态向量（16 维，归一化到 [-1, 1]）：
      [0]  dx_goal        目标相对 x / MAX_DIST_M
      [1]  dy_goal        目标相对 y / MAX_DIST_M
      [2]  dz_goal        目标相对 z / MAX_ALT_M
      [3]  nb_dist        最近建筑水平距离 / MAX_DIST_M
      [4]  nb_cos         最近建筑方向余弦
      [5]  nb_sin         最近建筑方向正弦
      [6]  nb_hdiff       建筑高度 - 当前高度 / MAX_ALT_M
      [7]  nu_dist        最近他机距离 / MAX_DIST_M
      [8]  nu_cos         最近他机方向余弦
      [9]  nu_sin         最近他机方向正弦
      [10] is_fixed_wing  机型（0=多旋翼 / 1=固定翼/VTOL）
      [11] wind_x         风北分量 = ws·cos(wind_to_deg) / MAX_WIND_MS
      [12] wind_y         风东分量 = ws·sin(wind_to_deg) / MAX_WIND_MS
      [13] heading_cos    当前航向 cos
      [14] heading_sin    当前航向 sin
      [15] soc            剩余电量比例 = 1 - 累计ΔWh / capacity

    动作向量（2 维连续，[-1, 1]）：
      [0]  heading_change: 映射到 [-MAX_HEADING_CHANGE, +MAX_HEADING_CHANGE] 度
      [1]  height_change : 映射到 [-MAX_HEIGHT_CHANGE,  +MAX_HEIGHT_CHANGE]  米
    """

    def __init__(
        self,
        buildings: List[Building],
        lat_ref: float = 32.0,
        other_paths: Optional[List[List[Dict]]] = None,
        drone: Optional[Drone] = None,
        wind_lookup: Optional[WindTimeseries] = None,
    ):
        self.buildings    = buildings
        self.lat_ref      = lat_ref
        self.other_paths: List[List[Dict]] = other_paths or []
        # P4 电量/风需要的 drone + 时变风。drone=None 时跳过 P4
        self.drone        = drone
        self.wind_lookup  = wind_lookup
        # 当前 episode 使用的风（在 reset() 根据 task.dronetime 查得）
        self._wind_ms     = 0.0
        self._wind_deg    = 0.0
        self._wind_nx     = 0.0   # 归一化的风向量（北/东分量）
        self._wind_ny     = 0.0
        # 当前 episode 的机型标识（reset 里根据 drone.craft_type 计算）
        self._is_fixed_wing = 0.0
        # 累计能耗（Wh），用于 soc 观测
        self._used_wh     = 0.0

        # 经度/纬度 → 米 的缩放因子（以 lat_ref 为基准平面展开），
        # 与 drone_obstacle_avoidance.py 里 path_obstacle_reward 所用一致。
        self._kd_cx = 111000.0 * np.cos(np.radians(lat_ref))
        self._kd_cy = 111000.0

        # 预缓存每栋建筑的米坐标中心 / 半径 / 最大高度，避免每步 step 重复换算
        if buildings:
            self._b_cx_m    = np.empty(len(buildings), dtype=np.float64)
            self._b_cy_m    = np.empty(len(buildings), dtype=np.float64)
            self._b_r_m     = np.empty(len(buildings), dtype=np.float64)
            self._b_hmax_m  = np.empty(len(buildings), dtype=np.float64)
            for i, b in enumerate(buildings):
                bd = b.get_bounds()
                self._b_cx_m[i]   = 0.5 * (bd['min_lon'] + bd['max_lon']) * self._kd_cx
                self._b_cy_m[i]   = 0.5 * (bd['min_lat'] + bd['max_lat']) * self._kd_cy
                self._b_r_m[i]    = 0.5 * max(
                    (bd['max_lon'] - bd['min_lon']) * self._kd_cx,
                    (bd['max_lat'] - bd['min_lat']) * self._kd_cy,
                )
                self._b_hmax_m[i] = float(b.height)
        else:
            self._b_cx_m = self._b_cy_m = self._b_r_m = self._b_hmax_m = np.empty(0)

        # 走廊建筑子集（reset 时按任务起终点重新计算，初始化为空）
        _empty = np.empty(0, dtype=np.float64)
        self._sub_b_cx     = _empty
        self._sub_b_cy     = _empty
        self._sub_b_r      = _empty
        self._sub_b_hmax   = _empty
        self._sub_b_cx_abs = _empty
        self._sub_b_cy_abs = _empty

        # 当前 episode 状态
        self.task: Optional[DeliveryTask] = None
        self._origin_lon = 0.0
        self._origin_lat = 0.0
        self.pos_m       = np.zeros(3, dtype=np.float64)   # [x_m, y_m, z_m]
        self.goal_m      = np.zeros(3, dtype=np.float64)
        self.heading_deg = 0.0
        self.step_count  = 0
        self.path: List[Dict] = []

    # ── 任务注入 ──────────────────────────────────────────────────────────────

    def set_task(self, task: DeliveryTask):
        self.task = task

    def set_other_paths(self, other_paths: List[List[Dict]]):
        self.other_paths = other_paths

    def set_drone(self, drone: Optional[Drone]):
        """切换 SAC 环境内部的 drone（影响 P4 电量奖励）。"""
        self.drone = drone

    def set_wind_lookup(self, wind_lookup: Optional[WindTimeseries]):
        """切换时变风查询器（下个 reset 生效）。"""
        self.wind_lookup = wind_lookup

    # ── 重置 ──────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        assert self.task is not None, "请先调用 set_task()"
        self._origin_lon = self.task.sender_lon
        self._origin_lat = self.task.sender_lat

        # 起点在原点
        self.pos_m = np.array([0.0, 0.0, float(CRUISE_HEIGHT)], dtype=np.float64)

        # 目标（接收方，巡航高度）
        gx = _deg2m_x(self.task.receiver_lon - self._origin_lon, self.lat_ref)
        gy = _deg2m_y(self.task.receiver_lat - self._origin_lat)
        self.goal_m = np.array([gx, gy, float(CRUISE_HEIGHT)], dtype=np.float64)

        # 初始距离（用于归一化进度奖励，保证不同距离的任务奖励量纲一致）
        self._init_dist = float(np.linalg.norm(self.goal_m - self.pos_m))
        self._init_dist = max(self._init_dist, 1.0)   # 防止除零

        # 初始航向朝向目标
        self.heading_deg = float(np.degrees(np.arctan2(gx, gy)) % 360)
        self.step_count  = 0

        # ── 查当前任务时刻的风（时变）────────────────────────────────────────
        if self.wind_lookup is not None and len(self.wind_lookup) > 0:
            dt = getattr(self.task, 'dronetime', None)
            ws, wd = self.wind_lookup.at(dt) if dt is not None else self.wind_lookup.mean()
            self._wind_ms, self._wind_deg = float(ws), float(wd)
        else:
            self._wind_ms, self._wind_deg = 0.0, 0.0
        # 归一化风向量：北=x、东=y 分量（吹向角度，北=0° 顺时针）
        wr = np.radians(self._wind_deg)
        self._wind_nx = float(self._wind_ms * np.cos(wr) / MAX_WIND_MS)
        self._wind_ny = float(self._wind_ms * np.sin(wr) / MAX_WIND_MS)

        # ── 按订单规模动态设 drone payload + 机型标识 + soc 清零 ─────────────
        self._used_wh = 0.0
        if self.drone is not None:
            oc = float(getattr(self.task, 'order_cnt', 1.0) or 1.0)
            self.drone.current_payload_kg = oc
            ct = getattr(self.drone, 'craft_type', 'multirotor')
            self._is_fixed_wing = 1.0 if ('fixed_wing' in ct or 'tiltrotor' in ct) else 0.0
        else:
            self._is_fixed_wing = 0.0

        # ── 任务走廊建筑子集（大幅减少每步遍历量）────────────────────────────
        # 以起点(0,0)→终点(gx,gy)连线的中点为圆心，半径 = 直线距离/2 + 缓冲，
        # 只保留圆内建筑。UAV 最多偏离直线 MAX_STEPS×STEP_DIST_M 米，但实际
        # 有效偏离远小于此；用 1.5 倍半径作为保守缓冲，覆盖绕行场景。
        cx_m = gx * 0.5
        cy_m = gy * 0.5
        corridor_r = self._init_dist * 0.5 * 1.5 + 500.0   # 额外 500m 安全边距
        # 建筑中心米坐标需换算到与 pos_m 同参考系（以起点为原点）
        # _b_cx_m/_b_cy_m 存的是绝对米坐标，需减去起点绝对坐标
        orig_ax = self._origin_lon * self._kd_cx
        orig_ay = self._origin_lat * self._kd_cy
        bx_rel  = self._b_cx_m - orig_ax
        by_rel  = self._b_cy_m - orig_ay
        mask = np.hypot(bx_rel - cx_m, by_rel - cy_m) <= corridor_r
        idx_sub = np.where(mask)[0]
        # 保存子集（若无建筑在走廊内则保持空，避免除零）
        self._sub_b_cx   = bx_rel[idx_sub]    # 相对起点的米坐标
        self._sub_b_cy   = by_rel[idx_sub]
        self._sub_b_r    = self._b_r_m[idx_sub]
        self._sub_b_hmax = self._b_hmax_m[idx_sub]
        # 绝对米坐标版（供 _segment_obstacle_reward 用）
        self._sub_b_cx_abs = self._b_cx_m[idx_sub]
        self._sub_b_cy_abs = self._b_cy_m[idx_sub]

        # 路径初始化
        self.path = [
            {'lon': self._origin_lon, 'lat': self._origin_lat,
             'height': 0.0, 'action': '起飞点'},
            {'lon': self._origin_lon, 'lat': self._origin_lat,
             'height': float(CRUISE_HEIGHT), 'action': f'爬升到{CRUISE_HEIGHT}米'},
        ]
        return self._get_obs()

    # ── 步进 ──────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, -1.0, 1.0)
        delta_h_deg = float(action[0]) * MAX_HEADING_CHANGE
        delta_z_m   = float(action[1]) * MAX_HEIGHT_CHANGE

        self.heading_deg = (self.heading_deg + delta_h_deg) % 360
        # 高度对齐到合规层 {60, 90, 120}m：
        #   action[1] > 0 → 抬高到当前层+1层（60→90 / 90→120）
        #   action[1] < 0 且 delta <= -30m → 降低到当前层-1层（120→90 / 90→60）
        #   其他 → 保持当前层不变
        raw_z = float(np.clip(self.pos_m[2] + delta_z_m, MIN_ALT_M, MAX_ALT_M))
        new_z = snap_to_layer(raw_z)

        rad = np.radians(self.heading_deg)
        dx  = STEP_DIST_M * np.sin(rad)
        dy  = STEP_DIST_M * np.cos(rad)

        prev_pos = self.pos_m.copy()
        self.pos_m = np.array([
            self.pos_m[0] + dx,
            self.pos_m[1] + dy,
            new_z,
        ], dtype=np.float64)
        self.step_count += 1

        cur_lon = self._origin_lon + _m2deg_lon(self.pos_m[0], self.lat_ref)
        cur_lat = self._origin_lat + _m2deg_lat(self.pos_m[1])
        self.path.append({
            'lon':    cur_lon,
            'lat':    cur_lat,
            'height': float(self.pos_m[2]),
            'action': f'SAC-step{self.step_count}',
        })

        # ── 奖励（增量版 P1~P3，对齐 drone_obstacle_avoidance.path_total_reward）──
        reward = 0.0

        prev_dist = float(np.linalg.norm(self.goal_m - prev_pos))
        curr_dist = float(np.linalg.norm(self.goal_m - self.pos_m))

        # P3 航迹意图（对齐 MultirotorUAV_4.py destination_reward）
        #   reward_i = 5 × (dist_current - dist_next) / (dist_current + 1e-10) - 5
        #   · 分母用 prev_dist（当前距离，动态），靠近目标时增益自然放大
        #   · -5 为时间惩罚，乘 REWARD_SCALE 与 P1/P2 量级对齐
        if prev_dist > 1.0:
            reward += 5.0 * (prev_dist - curr_dist) / (prev_dist + 1e-10)
            reward += -5.0 * REWARD_SCALE   # 时间惩罚：每步 -0.05

        # 构造当前新增段 (seg_from → seg_to) 用于 P1/P2 预判评分
        seg_from_lon = self._origin_lon + _m2deg_lon(prev_pos[0], self.lat_ref)
        seg_from_lat = self._origin_lat + _m2deg_lat(prev_pos[1])
        seg_from = {
            'lon':    seg_from_lon,
            'lat':    seg_from_lat,
            'height': float(prev_pos[2]),
        }
        seg_to = {
            'lon':    cur_lon,
            'lat':    cur_lat,
            'height': float(self.pos_m[2]),
        }

        # P2 建筑避障（1s 预判 -800，5s 预判 -200；仅当前段）
        # 排除：起降区域 / 地面段（与 path_obstacle_reward 排除逻辑一致）
        near_sender = (abs(seg_from['lon'] - self.task.sender_lon) < EXCL_DEG and
                       abs(seg_from['lat'] - self.task.sender_lat) < EXCL_DEG)
        near_recv   = (abs(seg_from['lon'] - self.task.receiver_lon) < EXCL_DEG and
                       abs(seg_from['lat'] - self.task.receiver_lat) < EXCL_DEG)
        if seg_from['height'] > 0 and not near_sender and not near_recv:
            r_p2 = self._segment_obstacle_reward(seg_from, seg_to)
            reward += r_p2 * REWARD_SCALE   # -800 → -8  /  -200 → -2

        # P1 机间避碰（12.5s 预判 -2000；仅当前段）
        r_p1 = self._segment_uav_interval_reward(seg_from, seg_to)
        reward += r_p1 * REWARD_SCALE       # -2000 → -20

        # P4 电量消耗（增量：段 ΔWh → -2 × ΔWh / capacity，与 path_battery_reward 累加等价）
        # 用 reset() 时从 wind_lookup 查到的任务时刻风；ΔWh 累加到 _used_wh 供 soc 观测
        seg_wh = self._segment_energy_wh(seg_from, seg_to)
        self._used_wh += seg_wh
        if self.drone is not None:
            capacity = float(getattr(self.drone, 'battery_capacity_wh', MR_BATTERY_CAPACITY_WH)) + 1e-10
            reward += -2.0 * (seg_wh / capacity)

        # 注：时间惩罚已包含在 P3 的 -5×REWARD_SCALE=-0.05/步 中，不再单独叠加

        # ── 终止判定 ─────────────────────────────────────────────────────────
        done = False
        info = {'success': False}

        if curr_dist < GOAL_THRESH_M:
            reward += 50.0
            done = True
            info['success'] = True
            land_wp = {
                'lon':    self.task.receiver_lon,
                'lat':    self.task.receiver_lat,
                'height': 0.0,
                'action': '降落',
            }
            # 降落段也纳入 P4，保证 ΣP4 == path_battery_reward(完整路径)
            land_wh = self._segment_energy_wh(seg_to, land_wp)
            self._used_wh += land_wh
            if self.drone is not None:
                capacity = float(getattr(self.drone, 'battery_capacity_wh', MR_BATTERY_CAPACITY_WH)) + 1e-10
                reward += -2.0 * (land_wh / capacity)
            self.path.append(land_wp)
        elif self.step_count >= MAX_STEPS:
            # 超时惩罚与剩余进度成比例
            reward -= 20.0 * (curr_dist / self._init_dist)
            done = True
            info['success'] = False
            self.path.append({
                'lon': cur_lon, 'lat': cur_lat,
                'height': 0.0, 'action': '超时降落',
            })

        # 单步 reward clipping，让 critic 学习更稳（tail 惩罚过大会放大目标偏差）
        reward = float(np.clip(reward, -30.0, 60.0))
        return self._get_obs(), reward, done, info

    # ── 获取观测 ──────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        dx = (self.goal_m[0] - self.pos_m[0]) / MAX_DIST_M
        dy = (self.goal_m[1] - self.pos_m[1]) / MAX_DIST_M
        dz = (self.goal_m[2] - self.pos_m[2]) / MAX_ALT_M

        nb_dist, nb_cos, nb_sin, nb_hdiff = self._nearest_building_info()
        nu_dist, nu_cos, nu_sin           = self._nearest_uav_info()

        # 当前航向 cos/sin（heading_deg: 北=0°，顺时针；sin→东，cos→北）
        hr = np.radians(self.heading_deg)
        hcos = float(np.cos(hr))
        hsin = float(np.sin(hr))

        # 剩余电量
        if self.drone is not None:
            capacity = float(getattr(self.drone, 'battery_capacity_wh', MR_BATTERY_CAPACITY_WH)) + 1e-10
            soc = float(np.clip(1.0 - self._used_wh / capacity, 0.0, 1.0))
        else:
            soc = 1.0
        soc01 = 2.0 * soc - 1.0   # 归一化到 [-1, 1]

        obs = np.array([
            dx, dy, dz,
            nb_dist / MAX_DIST_M, nb_cos, nb_sin, nb_hdiff / MAX_ALT_M,
            nu_dist / MAX_DIST_M, nu_cos, nu_sin,
            float(self._is_fixed_wing),
            float(self._wind_nx), float(self._wind_ny),
            hcos, hsin,
            soc01,
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    # ── 最近建筑信息 ──────────────────────────────────────────────────────────

    def _nearest_building_info(self) -> Tuple[float, float, float, float]:
        """
        在任务走廊子集内查最近建筑（reset 时预筛，平均只扫全量的 5-15%）。
        返回：距离(m), 方向 cos, 方向 sin, 建筑高度 - UAV 高度
        """
        px, py, pz = self.pos_m

        if len(self._sub_b_cx) == 0:
            return MAX_DIST_M, 1.0, 0.0, -MAX_ALT_M

        # UAV 当前位置相对起点的米坐标（与 _sub_b_cx/cy 同参考系）
        dists = np.hypot(self._sub_b_cx - px, self._sub_b_cy - py)
        idx   = int(np.argmin(dists))
        best_dist = float(dists[idx])

        # 方向向量（UAV → 建筑中心）
        bx, by = float(self._sub_b_cx[idx]), float(self._sub_b_cy[idx])
        angle  = np.arctan2(bx - px, by - py)
        best_cos   = float(np.cos(angle))
        best_sin   = float(np.sin(angle))
        best_hdiff = float(self._sub_b_hmax[idx] - pz)
        return best_dist, best_cos, best_sin, best_hdiff

    # ── 最近他机信息 ──────────────────────────────────────────────────────────

    def _nearest_uav_info(self) -> Tuple[float, float, float]:
        px, py, pz = self.pos_m
        best_dist  = MAX_DIST_M
        best_cos, best_sin = 1.0, 0.0
        for opath in self.other_paths:
            for wp in opath:
                ox = _deg2m_x(wp['lon'] - self._origin_lon, self.lat_ref)
                oy = _deg2m_y(wp['lat'] - self._origin_lat)
                oz = wp['height']
                d  = float(np.sqrt((px - ox)**2 + (py - oy)**2 + (pz - oz)**2))
                if 1e-6 < d < best_dist:
                    best_dist = d
                    angle     = np.arctan2(ox - px, oy - py)
                    best_cos  = float(np.cos(angle))
                    best_sin  = float(np.sin(angle))
        return best_dist, best_cos, best_sin

    def get_path(self) -> List[Dict]:
        return self.path

    # ── 新增段建筑 1s/5s 预判评分（对齐 path_obstacle_reward）──────────────────

    def _segment_obstacle_reward(self, p1: Dict, p2: Dict) -> float:
        """
        对单段 (p1, p2) 做 1s 和 5s 预判，命中建筑时返回对应的负分：
          · 1s 预判命中 → -800 （与 path_obstacle_reward 一致）
          · 5s 预判命中 → -200

        实现与 drone_obstacle_avoidance.path_obstacle_reward 完全对齐：
        · 起降排除：p1.height≤0 / near_sender / near_receiver → 返回 0
        · 建筑扫描：全量遍历 self.buildings（numpy 向量化，O(n)）
        """
        if not self.buildings:
            return 0.0

        # ── 起降段排除（与 path_obstacle_reward 排除规则一致）──────────────
        if float(p1['height']) <= 0.0:
            return 0.0
        if self.task is not None:
            if (abs(p1['lon'] - self.task.sender_lon)   < EXCL_DEG and
                abs(p1['lat'] - self.task.sender_lat)   < EXCL_DEG):
                return 0.0
            if (abs(p1['lon'] - self.task.receiver_lon) < EXCL_DEG and
                abs(p1['lat'] - self.task.receiver_lat) < EXCL_DEG):
                return 0.0

        x1 = p1['lon'] * self._kd_cx
        y1 = p1['lat'] * self._kd_cy
        z1 = float(p1['height'])
        x2 = p2['lon'] * self._kd_cx
        y2 = p2['lat'] * self._kd_cy
        z2 = float(p2['height'])
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        seg = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if seg < 1e-6:
            return 0.0

        r1 = min(PREDICT_1S_M / seg, 1.0)
        r5 = min(PREDICT_5S_M / seg, 1.0)
        n1x, n1y, n1h = x1 + dx * r1, y1 + dy * r1, z1 + dz * r1
        n5x, n5y, n5h = x1 + dx * r5, y1 + dy * r5, z1 + dz * r5

        # 走廊子集向量化扫描（reset 时已按任务起终点范围预筛，平均只扫全量 5-15%）
        bx = self._sub_b_cx_abs
        by = self._sub_b_cy_abs
        br = self._sub_b_r
        bh = self._sub_b_hmax

        thresh = br + OBS_SAFE_M
        d1 = np.hypot(n1x - bx, n1y - by)
        d5 = np.hypot(n5x - bx, n5y - by)
        hit1 = (d1 <= thresh) & (n1h > 0.0) & (n1h <= bh)
        hit5 = (d5 <= thresh) & (n5h > 0.0) & (n5h <= bh)
        return float(-800.0 * hit1.sum() - 200.0 * hit5.sum())

    # ── 段机间 12.5s 预判评分（对齐 MultirotorUAV_4.py uav_interval_reward）────

    def _segment_uav_interval_reward(self, p1: Dict, p2: Dict) -> float:
        """
        对单段 (p1, p2) 做 12.5s 预判（PREDICT_UAV_M = 187.5m，对齐参考实现）。
        冲突判断：水平净间隔 < UAV_SAFE_M 且 垂直高度差 < UAV_VERT_SAFE_M，
        两个条件同时满足才算冲突（h+v 分离判断以支持高度分层策略）。
        """
        if not self.other_paths:
            return 0.0

        x1, y1, z1 = p1['lon'] * self._kd_cx, p1['lat'] * self._kd_cy, float(p1['height'])
        x2, y2, z2 = p2['lon'] * self._kd_cx, p2['lat'] * self._kd_cy, float(p2['height'])
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        seg = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        ratio = min(PREDICT_UAV_M / seg, 1.0) if seg > 1e-6 else 0.0
        px = x1 + dx * ratio
        py = y1 + dy * ratio
        pz = z1 + dz * ratio

        for opath in self.other_paths:
            for q in opath:
                qz = float(q.get('height', 0.0))
                if qz <= 0:
                    continue   # 跳过他机起降点
                qx = q['lon'] * self._kd_cx
                qy = q['lat'] * self._kd_cy
                h_dist = float(np.hypot(px - qx, py - qy)) - 2.0 * UAV_RADIUS_M
                v_diff = abs(pz - qz)
                if h_dist < UAV_SAFE_M and v_diff < UAV_VERT_SAFE_M:
                    return -2000.0
        return 0.0

    # ── 新增段电量奖励（对齐 path_battery_reward 的增量版）────────────────────

    def _segment_energy_wh(self, p1: Dict, p2: Dict) -> float:
        """
        对单段 (p1, p2) 估算 ΔWh（对齐 Drone.estimate_path_energy_wh 的水平 + 爬降分支）。

        · 水平段
          - 风沿航向分量 wind_along = wind_ms * cos(wind_deg - heading)
          - ground_speed = max(speed + wind_along, 0.2*speed)
          - wind_factor  = clip(1 + (WIND_POWER_FACTOR/speed)*(-wind_along), 0.5, 2.0)
          - energy = base_power * wind_factor * (dist_h / (ground_speed * 3600))
        · 爬降：|dh| > 1m 时额外 climb/descent_power_w * (|dh|/(v_speed*3600))
        · 若 self.drone 未提供，返回 0（P4 跳过）
        """
        if self.drone is None:
            return 0.0
        drone = self.drone
        speed = max(float(getattr(drone, 'cruise_speed', 15.0)), 1.0)

        lat0  = float(p1['lat'])
        dlon  = (p2['lon'] - p1['lon']) * 111000.0 * np.cos(np.radians(lat0))
        dlat  = (p2['lat'] - p1['lat']) * 111000.0
        dist_h = float(np.hypot(dlon, dlat))
        dh = float(p2['height'] - p1['height'])

        base_power = (float(drone.cruise_power_w)
                      + float(drone.current_payload_kg) * float(drone.payload_power_factor))
        wh = 0.0

        # 水平巡航（含风，对齐 estimate_path_energy_wh）
        if dist_h > 0.1:
            heading = float(np.degrees(np.arctan2(dlon, dlat)) % 360.0)
            if self._wind_ms > 0:
                delta = np.radians(self._wind_deg - heading)
                wind_along = self._wind_ms * float(np.cos(delta))
                ground = max(speed + wind_along, speed * 0.2)
                time_h = dist_h / (ground * 3600.0)
                wind_factor = float(np.clip(
                    1.0 + (WIND_POWER_FACTOR / speed) * (-wind_along),
                    0.5, 2.0,
                ))
            else:
                time_h = dist_h / (speed * 3600.0)
                wind_factor = 1.0
            wh += base_power * wind_factor * time_h

        # 爬升/下降（跳过 '起飞'/'降落' action，与原函数一致）
        # 用能耗记账专用爬升率：固定翼运动学 vertical_speed=0，若沿用它会让
        # 固定翼的爬升能耗被静默跳过，与 estimate_path_energy_wh 口径不一致。
        v_speed = float(getattr(drone, 'climb_rate_m_s', 3.0))
        if v_speed > 0 and abs(dh) > 1.0:
            action = p2.get('action', '')
            is_takeoff = ('起飞' in action) or ('爬升' in action)
            is_landing = ('降落' in action)
            if dh > 0 and not is_takeoff:
                wh += float(drone.climb_power_w) * (abs(dh) / (v_speed * 3600.0))
            elif dh < 0 and not is_landing:
                wh += abs(float(drone.descent_power_w)) * (abs(dh) / (v_speed * 3600.0))

        return float(wh)

    def _segment_battery_reward(self, p1: Dict, p2: Dict) -> float:
        """兼容接口：段电量 → P4 奖励（-2 × ΔWh / capacity）。"""
        if self.drone is None:
            return 0.0
        wh = self._segment_energy_wh(p1, p2)
        capacity = float(getattr(self.drone, 'battery_capacity_wh', MR_BATTERY_CAPACITY_WH)) + 1e-10
        return -2.0 * (wh / capacity)
