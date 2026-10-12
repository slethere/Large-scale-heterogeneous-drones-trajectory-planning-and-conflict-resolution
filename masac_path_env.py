#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
masac_path_env.py — K=2 CTDE 双机同步路径规划环境。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from drone_obstacle_avoidance import (
    Building,
    CRUISE_HEIGHT,
    DeliveryTask,
    Drone,
    WindTimeseries,
    WIND_POWER_FACTOR,
    snap_to_layer,
    MR_BATTERY_CAPACITY_WH,
)
from sac_path_env import (
    ACTION_DIM,
    EXCL_DEG,
    GOAL_THRESH_M,
    MAX_ALT_M,
    MAX_DIST_M,
    MAX_HEADING_CHANGE,
    MAX_HEIGHT_CHANGE,
    MAX_STEPS,
    MAX_WIND_MS,
    MIN_ALT_M,
    OBS_SAFE_M,
    PREDICT_1S_M,
    PREDICT_5S_M,
    PREDICT_UAV_M,
    REWARD_SCALE,
    STATE_DIM,
    STEP_DIST_M,
    UAV_RADIUS_M,
    UAV_SAFE_M,
    UAV_VERT_SAFE_M,
    _deg2m_x,
    _deg2m_y,
    _m2deg_lat,
    _m2deg_lon,
)

LOCAL_OBS_DIM = STATE_DIM
GLOBAL_STATE_DIM = LOCAL_OBS_DIM * 2
JOINT_ACTION_DIM = ACTION_DIM * 2
TEAM_CONFLICT_LAMBDA = 5.0
NUM_AGENTS = 2


@dataclass
class AgentSlot:
    task: DeliveryTask
    drone: Drone
    pos_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    goal_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heading_deg: float = 0.0
    path: List[Dict] = field(default_factory=list)
    used_wh: float = 0.0
    step_count: int = 0
    done: bool = False
    success: bool = False
    init_dist: float = 1.0
    is_fixed_wing: float = 0.0


class MASACPathEnv:
    NUM_AGENTS = NUM_AGENTS
    GLOBAL_STATE_DIM = GLOBAL_STATE_DIM
    JOINT_ACTION_DIM = JOINT_ACTION_DIM

    def __init__(
        self,
        buildings: List[Building],
        lat_ref: float = 32.0,
        wind_lookup: Optional[WindTimeseries] = None,
    ):
        self.buildings = buildings
        self.lat_ref = lat_ref
        self.wind_lookup = wind_lookup
        self._kd_cx = 111000.0 * np.cos(np.radians(lat_ref))
        self._kd_cy = 111000.0

        if buildings:
            self._b_cx_m = np.empty(len(buildings), dtype=np.float64)
            self._b_cy_m = np.empty(len(buildings), dtype=np.float64)
            self._b_r_m = np.empty(len(buildings), dtype=np.float64)
            self._b_hmax_m = np.empty(len(buildings), dtype=np.float64)
            for i, b in enumerate(buildings):
                bd = b.get_bounds()
                self._b_cx_m[i] = 0.5 * (bd['min_lon'] + bd['max_lon']) * self._kd_cx
                self._b_cy_m[i] = 0.5 * (bd['min_lat'] + bd['max_lat']) * self._kd_cy
                self._b_r_m[i] = 0.5 * max(
                    (bd['max_lon'] - bd['min_lon']) * self._kd_cx,
                    (bd['max_lat'] - bd['min_lat']) * self._kd_cy,
                )
                self._b_hmax_m[i] = float(b.height)
        else:
            self._b_cx_m = self._b_cy_m = self._b_r_m = self._b_hmax_m = np.empty(0)

        self._sub_b_cx = np.empty(0)
        self._sub_b_cy = np.empty(0)
        self._sub_b_r = np.empty(0)
        self._sub_b_hmax = np.empty(0)
        self._sub_b_cx_abs = np.empty(0)
        self._sub_b_cy_abs = np.empty(0)

        self._origin_lon = 0.0
        self._origin_lat = 0.0
        self._wind_ms = 0.0
        self._wind_deg = 0.0
        self._wind_nx = 0.0
        self._wind_ny = 0.0
        self.agents: List[AgentSlot] = []
        self._global_step = 0

    def reset(
        self,
        tasks: List[DeliveryTask],
        drones: List[Drone],
    ) -> List[np.ndarray]:
        assert len(tasks) == NUM_AGENTS and len(drones) == NUM_AGENTS
        self._origin_lon = 0.5 * (tasks[0].sender_lon + tasks[1].sender_lon)
        self._origin_lat = 0.5 * (tasks[0].sender_lat + tasks[1].sender_lat)
        self.lat_ref = 0.5 * (tasks[0].sender_lat + tasks[1].sender_lat)
        self._kd_cx = 111000.0 * np.cos(np.radians(self.lat_ref))
        self._global_step = 0

        ws, wd = 0.0, 0.0
        if self.wind_lookup is not None and len(self.wind_lookup) > 0:
            dt0 = getattr(tasks[0], 'dronetime', None)
            ws, wd = self.wind_lookup.at(dt0) if dt0 is not None else self.wind_lookup.mean()
        self._wind_ms, self._wind_deg = float(ws), float(wd)
        wr = np.radians(self._wind_deg)
        self._wind_nx = float(self._wind_ms * np.cos(wr) / MAX_WIND_MS)
        self._wind_ny = float(self._wind_ms * np.sin(wr) / MAX_WIND_MS)

        self._build_corridor_subset(tasks)
        self.agents = []
        for i in range(NUM_AGENTS):
            t, d = tasks[i], drones[i]
            oc = float(getattr(t, 'order_cnt', 1.0) or 1.0)
            d.current_payload_kg = oc
            ct = getattr(d, 'craft_type', 'multirotor')
            is_fw = 1.0 if ('fixed_wing' in ct or 'tiltrotor' in ct) else 0.0

            sx = _deg2m_x(t.sender_lon - self._origin_lon, self.lat_ref)
            sy = _deg2m_y(t.sender_lat - self._origin_lat)
            gx = _deg2m_x(t.receiver_lon - self._origin_lon, self.lat_ref)
            gy = _deg2m_y(t.receiver_lat - self._origin_lat)

            pos = np.array([sx, sy, float(CRUISE_HEIGHT)], dtype=np.float64)
            goal = np.array([gx, gy, float(CRUISE_HEIGHT)], dtype=np.float64)
            init_d = max(float(np.linalg.norm(goal - pos)), 1.0)
            hd = float(np.degrees(np.arctan2(gx - sx, gy - sy)) % 360)

            path = [
                {'lon': t.sender_lon, 'lat': t.sender_lat,
                 'height': 0.0, 'action': '起飞点'},
                {'lon': t.sender_lon, 'lat': t.sender_lat,
                 'height': float(CRUISE_HEIGHT),
                 'action': f'爬升到{CRUISE_HEIGHT}米'},
            ]
            self.agents.append(AgentSlot(
                task=t, drone=d, pos_m=pos, goal_m=goal,
                heading_deg=hd, path=path, init_dist=init_d,
                is_fixed_wing=is_fw,
            ))
        return self._get_obs_list()

    def _build_corridor_subset(self, tasks: List[DeliveryTask]) -> None:
        orig_ax = self._origin_lon * self._kd_cx
        orig_ay = self._origin_lat * self._kd_cy
        bx_rel = self._b_cx_m - orig_ax
        by_rel = self._b_cy_m - orig_ay

        if len(self.buildings) == 0:
            self._sub_b_cx = self._sub_b_cy = self._sub_b_r = self._sub_b_hmax = np.empty(0)
            self._sub_b_cx_abs = self._sub_b_cy_abs = np.empty(0)
            return

        mask = np.zeros(len(self.buildings), dtype=bool)
        for t in tasks:
            sx = _deg2m_x(t.sender_lon - self._origin_lon, self.lat_ref)
            sy = _deg2m_y(t.sender_lat - self._origin_lat)
            gx = _deg2m_x(t.receiver_lon - self._origin_lon, self.lat_ref)
            gy = _deg2m_y(t.receiver_lat - self._origin_lat)
            cx, cy = (sx + gx) * 0.5, (sy + gy) * 0.5
            dist = float(np.hypot(gx - sx, gy - sy))
            r = dist * 0.5 * 1.5 + 500.0
            mask |= np.hypot(bx_rel - cx, by_rel - cy) <= r

        idx = np.where(mask)[0]
        self._sub_b_cx = bx_rel[idx]
        self._sub_b_cy = by_rel[idx]
        self._sub_b_r = self._b_r_m[idx]
        self._sub_b_hmax = self._b_hmax_m[idx]
        self._sub_b_cx_abs = self._b_cx_m[idx]
        self._sub_b_cy_abs = self._b_cy_m[idx]

    def step(
        self,
        actions: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        assert len(actions) == NUM_AGENTS
        if all(a.done for a in self.agents):
            return self._get_obs_list(), [0.0, 0.0], True, self._info_dict()

        pre_pos = [a.pos_m.copy() for a in self.agents]
        pre_heading = [a.heading_deg for a in self.agents]
        transitions = []
        for i, action in enumerate(actions):
            if self.agents[i].done:
                transitions.append(None)
            else:
                transitions.append(
                    self._compute_transition(i, action, pre_pos[i], pre_heading[i]))

        team_conflict = False
        if transitions[0] is not None and transitions[1] is not None:
            team_conflict = self._segments_conflict(
                transitions[0][1], transitions[0][2],
                pre_pos[1],
                transitions[1][1], transitions[1][2],
                pre_pos[0],
            )
        team_penalty = -TEAM_CONFLICT_LAMBDA if team_conflict else 0.0

        rewards: List[float] = []
        for i, trans in enumerate(transitions):
            slot = self.agents[i]
            if trans is None:
                rewards.append(0.0)
                continue

            new_pos, seg_from, seg_to, local_r, done, success, heading, seg_wh = trans
            slot.pos_m = new_pos
            slot.heading_deg = heading
            slot.step_count += 1
            slot.used_wh += seg_wh
            self._global_step = max(self._global_step, slot.step_count)

            cur_lon = self._origin_lon + _m2deg_lon(new_pos[0], self.lat_ref)
            cur_lat = self._origin_lat + _m2deg_lat(new_pos[1])
            slot.path.append({
                'lon': cur_lon, 'lat': cur_lat,
                'height': float(new_pos[2]),
                'action': f'MASAC-step{slot.step_count}',
            })

            if success and done:
                land_wp = {
                    'lon': slot.task.receiver_lon,
                    'lat': slot.task.receiver_lat,
                    'height': 0.0, 'action': '降落',
                }
                land_wh = self._segment_energy_wh(slot, seg_to, land_wp)
                slot.used_wh += land_wh
                if slot.drone is not None:
                    cap = float(getattr(slot.drone, 'battery_capacity_wh', MR_BATTERY_CAPACITY_WH)) + 1e-10
                    local_r += -2.0 * (land_wh / cap)
                slot.path.append(land_wp)
            elif done and not success:
                slot.path.append({
                    'lon': cur_lon, 'lat': cur_lat,
                    'height': 0.0, 'action': '超时降落',
                })

            slot.done = done
            slot.success = success
            rewards.append(float(np.clip(local_r + team_penalty, -30.0, 60.0)))

        episode_done = all(a.done for a in self.agents)
        if not episode_done and self._global_step >= MAX_STEPS:
            for slot in self.agents:
                if not slot.done:
                    slot.done = True
            episode_done = True

        return self._get_obs_list(), rewards, episode_done, self._info_dict(team_conflict)

    def _compute_transition(
        self,
        agent_idx: int,
        action: np.ndarray,
        prev_pos: np.ndarray,
        prev_heading: float,
    ) -> Tuple[np.ndarray, Dict, Dict, float, bool, bool, float, float]:
        action = np.clip(action, -1.0, 1.0)
        slot = self.agents[agent_idx]
        delta_h = float(action[0]) * MAX_HEADING_CHANGE
        delta_z = float(action[1]) * MAX_HEIGHT_CHANGE
        heading = (prev_heading + delta_h) % 360
        raw_z = float(np.clip(prev_pos[2] + delta_z, MIN_ALT_M, MAX_ALT_M))
        new_z = snap_to_layer(raw_z)
        rad = np.radians(heading)
        dx = STEP_DIST_M * np.sin(rad)
        dy = STEP_DIST_M * np.cos(rad)
        new_pos = np.array([
            prev_pos[0] + dx, prev_pos[1] + dy, new_z,
        ], dtype=np.float64)

        prev_dist = float(np.linalg.norm(slot.goal_m - prev_pos))
        curr_dist = float(np.linalg.norm(slot.goal_m - new_pos))
        reward = 0.0
        if prev_dist > 1.0:
            reward += 5.0 * (prev_dist - curr_dist) / (prev_dist + 1e-10)
            reward += -5.0 * REWARD_SCALE

        seg_from = self._pos_to_wp(prev_pos, prev_pos[2])
        seg_to = self._pos_to_wp(new_pos, new_pos[2])
        task = slot.task

        near_s = (abs(seg_from['lon'] - task.sender_lon) < EXCL_DEG and
                  abs(seg_from['lat'] - task.sender_lat) < EXCL_DEG)
        near_r = (abs(seg_from['lon'] - task.receiver_lon) < EXCL_DEG and
                  abs(seg_from['lat'] - task.receiver_lat) < EXCL_DEG)
        if seg_from['height'] > 0 and not near_s and not near_r:
            reward += self._segment_obstacle_reward(seg_from, seg_to) * REWARD_SCALE

        other = self.agents[1 - agent_idx]
        if float(other.pos_m[2]) > 0 and not other.done:
            reward += self._segment_vs_point_reward(
                seg_from, seg_to, other.pos_m) * REWARD_SCALE

        seg_wh = self._segment_energy_wh(slot, seg_from, seg_to)
        if slot.drone is not None:
            cap = float(getattr(slot.drone, 'battery_capacity_wh', MR_BATTERY_CAPACITY_WH)) + 1e-10
            reward += -2.0 * (seg_wh / cap)

        done = False
        success = False
        if curr_dist < GOAL_THRESH_M:
            reward += 50.0
            done = True
            success = True
        elif slot.step_count + 1 >= MAX_STEPS:
            reward -= 20.0 * (curr_dist / slot.init_dist)
            done = True

        return new_pos, seg_from, seg_to, reward, done, success, heading, seg_wh

    def _pos_to_wp(self, pos_m: np.ndarray, height: float) -> Dict:
        return {
            'lon': self._origin_lon + _m2deg_lon(pos_m[0], self.lat_ref),
            'lat': self._origin_lat + _m2deg_lat(pos_m[1]),
            'height': float(height),
        }

    def _predict_point(self, p1: Dict, p2: Dict) -> Tuple[float, float, float]:
        x1 = p1['lon'] * self._kd_cx
        y1 = p1['lat'] * self._kd_cy
        z1 = float(p1['height'])
        x2 = p2['lon'] * self._kd_cx
        y2 = p2['lat'] * self._kd_cy
        z2 = float(p2['height'])
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        seg = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        ratio = min(PREDICT_UAV_M / seg, 1.0) if seg > 1e-6 else 0.0
        return x1 + dx * ratio, y1 + dy * ratio, z1 + dz * ratio

    def _segment_vs_point_reward(
        self, p1: Dict, p2: Dict, other_pos_m: np.ndarray,
    ) -> float:
        px, py, pz = self._predict_point(p1, p2)
        orig_ax = self._origin_lon * self._kd_cx
        orig_ay = self._origin_lat * self._kd_cy
        qx = orig_ax + float(other_pos_m[0])
        qy = orig_ay + float(other_pos_m[1])
        qz = float(other_pos_m[2])
        if qz <= 0:
            return 0.0
        h_dist = float(np.hypot(px - qx, py - qy)) - 2.0 * UAV_RADIUS_M
        v_diff = abs(pz - qz)
        if h_dist < UAV_SAFE_M and v_diff < UAV_VERT_SAFE_M:
            return -2000.0
        return 0.0

    def _segments_conflict(
        self,
        sf0: Dict, st0: Dict, other_pos0: np.ndarray,
        sf1: Dict, st1: Dict, other_pos1: np.ndarray,
    ) -> bool:
        if self._segment_vs_point_reward(sf0, st0, other_pos0) < 0:
            return True
        if self._segment_vs_point_reward(sf1, st1, other_pos1) < 0:
            return True
        return False

    def _segment_obstacle_reward(self, p1: Dict, p2: Dict) -> float:
        if len(self._sub_b_cx) == 0:
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
        bx, by, br, bh = (self._sub_b_cx_abs, self._sub_b_cy_abs,
                          self._sub_b_r, self._sub_b_hmax)
        thresh = br + OBS_SAFE_M
        d1 = np.hypot(n1x - bx, n1y - by)
        d5 = np.hypot(n5x - bx, n5y - by)
        hit1 = (d1 <= thresh) & (n1h > 0.0) & (n1h <= bh)
        hit5 = (d5 <= thresh) & (n5h > 0.0) & (n5h <= bh)
        return float(-800.0 * hit1.sum() - 200.0 * hit5.sum())

    def _segment_energy_wh(self, slot: AgentSlot, p1: Dict, p2: Dict) -> float:
        if slot.drone is None:
            return 0.0
        drone = slot.drone
        speed = max(float(getattr(drone, 'cruise_speed', 15.0)), 1.0)
        lat0 = float(p1['lat'])
        dlon = (p2['lon'] - p1['lon']) * 111000.0 * np.cos(np.radians(lat0))
        dlat = (p2['lat'] - p1['lat']) * 111000.0
        dist_h = float(np.hypot(dlon, dlat))
        dh = float(p2['height'] - p1['height'])
        base_power = (float(drone.cruise_power_w)
                      + float(drone.current_payload_kg) * float(drone.payload_power_factor))
        wh = 0.0
        if dist_h > 0.1:
            heading = float(np.degrees(np.arctan2(dlon, dlat)) % 360.0)
            if self._wind_ms > 0:
                delta = np.radians(self._wind_deg - heading)
                wind_along = self._wind_ms * float(np.cos(delta))
                ground = max(speed + wind_along, speed * 0.2)
                time_h = dist_h / (ground * 3600.0)
                wind_factor = float(np.clip(
                    1.0 + (WIND_POWER_FACTOR / speed) * (-wind_along), 0.5, 2.0,
                ))
            else:
                time_h = dist_h / (speed * 3600.0)
                wind_factor = 1.0
            wh += base_power * wind_factor * time_h
        # 能耗记账用爬升率（非运动学 vertical_speed，固定翼后者为 0 会漏计爬升）
        v_speed = float(getattr(drone, 'climb_rate_m_s', 3.0))
        if v_speed > 0 and abs(dh) > 1.0:
            action = p2.get('action', '')
            if dh > 0 and '起飞' not in action and '爬升' not in action:
                wh += float(drone.climb_power_w) * (abs(dh) / (v_speed * 3600.0))
            elif dh < 0 and '降落' not in action:
                wh += abs(float(drone.descent_power_w)) * (abs(dh) / (v_speed * 3600.0))
        return float(wh)

    def _nearest_building_info(self, pos_m: np.ndarray) -> Tuple[float, float, float, float]:
        if len(self._sub_b_cx) == 0:
            return MAX_DIST_M, 1.0, 0.0, -MAX_ALT_M
        dists = np.hypot(self._sub_b_cx - pos_m[0], self._sub_b_cy - pos_m[1])
        idx = int(np.argmin(dists))
        best_dist = float(dists[idx])
        bx, by = float(self._sub_b_cx[idx]), float(self._sub_b_cy[idx])
        angle = np.arctan2(bx - pos_m[0], by - pos_m[1])
        return best_dist, float(np.cos(angle)), float(np.sin(angle)), float(self._sub_b_hmax[idx] - pos_m[2])

    def _get_obs_for_agent(self, idx: int) -> np.ndarray:
        slot = self.agents[idx]
        other = self.agents[1 - idx]
        pos = slot.pos_m
        dx = (slot.goal_m[0] - pos[0]) / MAX_DIST_M
        dy = (slot.goal_m[1] - pos[1]) / MAX_DIST_M
        dz = (slot.goal_m[2] - pos[2]) / MAX_ALT_M
        nb_dist, nb_cos, nb_sin, nb_hdiff = self._nearest_building_info(pos)

        if float(other.pos_m[2]) > 0 and not other.done:
            od = other.pos_m - pos
            nu_dist = float(np.linalg.norm(od))
            if nu_dist > 1e-6:
                nu_cos = float(od[0] / nu_dist)
                nu_sin = float(od[1] / nu_dist)
            else:
                nu_dist, nu_cos, nu_sin = 0.0, 1.0, 0.0
        else:
            nu_dist, nu_cos, nu_sin = MAX_DIST_M, 1.0, 0.0

        hr = np.radians(slot.heading_deg)
        if slot.drone is not None:
            cap = float(getattr(slot.drone, 'battery_capacity_wh', MR_BATTERY_CAPACITY_WH)) + 1e-10
            soc = float(np.clip(1.0 - slot.used_wh / cap, 0.0, 1.0))
        else:
            soc = 1.0
        obs = np.array([
            dx, dy, dz,
            nb_dist / MAX_DIST_M, nb_cos, nb_sin, nb_hdiff / MAX_ALT_M,
            nu_dist / MAX_DIST_M, nu_cos, nu_sin,
            float(slot.is_fixed_wing),
            float(self._wind_nx), float(self._wind_ny),
            float(np.cos(hr)), float(np.sin(hr)),
            2.0 * soc - 1.0,
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _get_obs_list(self) -> List[np.ndarray]:
        return [self._get_obs_for_agent(i) for i in range(NUM_AGENTS)]

    def get_global_state(self) -> np.ndarray:
        return np.concatenate(self._get_obs_list(), axis=0).astype(np.float32)

    def get_paths(self) -> List[List[Dict]]:
        return [list(a.path) for a in self.agents]

    def _info_dict(self, team_conflict: bool = False) -> dict:
        return {
            'success': [a.success for a in self.agents],
            'paths': self.get_paths(),
            'team_conflict': team_conflict,
        }
