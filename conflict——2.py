import numpy as np
from typing import List, Tuple, Dict, Optional

# 沿航迹 `_t` 采样时的安全上限，防止异常时间跨度下分配巨量内存
_MAX_GEO_PATH_DURATION_S = 24 * 3600.0
_MAX_GEO_SAMPLES = 50_000
_CRUISE_MS_EST = 15.0
_VERT_MS_EST = 4.0
_GEOM_DURATION_TRUST_RATIO = 10.0


def _estimate_path_duration_lonlat_s(path: List[Dict]) -> float:
    """与 alns `_attach_timestamps` 同源：由航段水平/垂直位移估算飞行时长。"""
    total = 0.0
    for i in range(len(path) - 1):
        wp, nxt = path[i], path[i + 1]
        dlat = (float(nxt['lat']) - float(wp['lat'])) * 111000.0
        dlon = (float(nxt['lon']) - float(wp['lon'])) * 111000.0 * np.cos(
            np.radians(float(wp['lat'])))
        dh = float(nxt.get('height', 0)) - float(wp.get('height', 0))
        dist_h = float(np.sqrt(dlat * dlat + dlon * dlon))
        t_h = dist_h / _CRUISE_MS_EST if dist_h > 0.1 else 0.0
        t_v = abs(dh) / _VERT_MS_EST if (abs(dh) > 1.0 and dist_h < 0.1) else 0.0
        total += max(t_h, t_v)
    return max(total, 1e-6)


def _sanitize_path_time_window(path: List[Dict]) -> Tuple[float, float]:
    """由首尾 `_t` 得到有效时间窗；异常跨度改用语义上几何飞行时长。"""
    t_start = float(path[0].get('_t', 0.0))
    t_end_ws = float(path[-1].get('_t', t_start))
    delta = t_end_ws - t_start
    est = _estimate_path_duration_lonlat_s(path)
    max_d = _MAX_GEO_PATH_DURATION_S
    suspect_floor = max(_GEOM_DURATION_TRUST_RATIO * est + 300.0, 600.0)

    if delta <= 1e-9:
        return t_start, t_start
    if delta > max_d:
        return t_start, t_start + min(max(est, 1e-3), max_d)
    if delta > suspect_floor:
        return t_start, t_start + min(max(est, 1e-3), max_d)
    return t_start, t_end_ws


class UAVCollisionDetector:
    """
    基于几何的无人机冲突检测器：
      条件1：瞬时水平距离 < R1+R2 且垂直差在容差内
      条件2：在时间窗 T 内，两机水平预测线段相交且同垂直容差内
    """

    def __init__(self, time_step=0.5, T=15.0, height_tolerance_m: float = 1e-3):
        self.time_step = time_step
        self.T = T
        self.height_tolerance_m = float(height_tolerance_m)

    def _compute_heading_and_speed(self, path: List[List[float]], t: float, speed: float) -> Tuple[
        float, float, float, float, float, float]:
        cum_time = 0.0
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])
            seg_len = np.linalg.norm(p2 - p1)
            seg_time = seg_len / max(speed, 1e-6)
            if t <= cum_time + seg_time + 1e-8:
                alpha = (t - cum_time) / max(seg_time, 1e-6)
                pos = p1 + alpha * (p2 - p1)
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                heading = np.arctan2(dy, dx)
                vx = speed * np.cos(heading)
                vy = speed * np.sin(heading)
                return pos[0], pos[1], pos[2], vx, vy, heading
            cum_time += seg_time
        last = np.array(path[-1])
        prev = np.array(path[-2]) if len(path) >= 2 else last
        dx = last[0] - prev[0]
        dy = last[1] - prev[1]
        heading = np.arctan2(dy, dx)
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        return last[0], last[1], last[2], vx, vy, heading

    def _line_segments_intersect_2d(self, p1, p2, p3, p4) -> bool:
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def _future_segments_intersect(self, state1, state2, speed1, speed2, T) -> bool:
        x1, y1, z1, h1 = state1
        x2, y2, z2, h2 = state2
        if abs(z1 - z2) > self.height_tolerance_m:
            return False
        x1e = x1 + speed1 * T * np.cos(h1)
        y1e = y1 + speed1 * T * np.sin(h1)
        x2e = x2 + speed2 * T * np.cos(h2)
        y2e = y2 + speed2 * T * np.sin(h2)
        return self._line_segments_intersect_2d((x1, y1), (x1e, y1e), (x2, y2), (x2e, y2e))

    def _interpolate_state(self, states: List[Tuple[float, float, float, float, float]], t: float):
        if not states:
            return None
        if t <= states[0][0]:
            return states[0]
        if t >= states[-1][0]:
            return states[-1]
        for idx in range(len(states) - 1):
            t1, x1, y1, z1, h1 = states[idx]
            t2, x2, y2, z2, h2 = states[idx + 1]
            if t1 <= t <= t2:
                alpha = (t - t1) / max(t2 - t1, 1e-6)
                x = x1 + alpha * (x2 - x1)
                y = y1 + alpha * (y2 - y1)
                z = z1 + alpha * (z2 - z1)
                h = h1 + alpha * (h2 - h1)
                return (t, x, y, z, h)
        return None

    def detect_conflicts(self, tasks_with_paths: List[Tuple[dict, List[List[float]]]]) -> List[int]:
        if not tasks_with_paths:
            return []

        sampled_list = []
        for idx, (task, path) in enumerate(tasks_with_paths):
            speed = task['uav'].speed
            takeoff = task.get('takeoff_time', 0)
            if not path or len(path) < 2:
                continue
            total_len = 0.0
            for i in range(len(path) - 1):
                p1 = np.array(path[i])
                p2 = np.array(path[i + 1])
                total_len += np.linalg.norm(p2 - p1)
            total_time = min(total_len / max(speed, 1e-6), _MAX_GEO_PATH_DURATION_S)
            n = int(np.ceil(total_time / max(self.time_step, 1e-6))) + 1
            n = max(2, min(n, _MAX_GEO_SAMPLES))
            times = np.linspace(float(takeoff), float(takeoff + total_time), n, dtype=np.float64)
            states = []
            for t in times:
                x, y, z, vx, vy, heading = self._compute_heading_and_speed(
                    path, float(t) - float(takeoff), speed)
                states.append((float(t), x, y, z, heading))
            sampled_list.append((idx, task, states))

        conflict_indices = set()
        n = len(sampled_list)
        for i in range(n):
            idx_i, task_i, states_i = sampled_list[i]
            if not states_i:
                continue
            R_i = task_i['uav'].safety_radius
            speed_i = task_i['uav'].speed
            for j in range(i + 1, n):
                idx_j, task_j, states_j = sampled_list[j]
                if not states_j:
                    continue
                R_j = task_j['uav'].safety_radius
                speed_j = task_j['uav'].speed
                R = R_i + R_j
                t_i_start, t_i_end = states_i[0][0], states_i[-1][0]
                t_j_start, t_j_end = states_j[0][0], states_j[-1][0]
                if t_i_end < t_j_start or t_j_end < t_i_start:
                    continue
                overlap_start = max(t_i_start, t_j_start)
                overlap_end = min(t_i_end, t_j_end)
                overlap_span = overlap_end - overlap_start
                if overlap_span <= 0:
                    continue
                pair_n = int(np.ceil(overlap_span / max(self.time_step, 1e-6))) + 1
                pair_n = max(2, min(pair_n, _MAX_GEO_SAMPLES))
                for t in np.linspace(overlap_start, overlap_end, pair_n, dtype=np.float64):
                    state_i = self._interpolate_state(states_i, float(t))
                    state_j = self._interpolate_state(states_j, float(t))
                    if state_i is None or state_j is None:
                        continue
                    _, x_i, y_i, z_i, h_i = state_i
                    _, x_j, y_j, z_j, h_j = state_j
                    dist_hor = np.hypot(x_i - x_j, y_i - y_j)
                    if dist_hor < R and abs(z_i - z_j) < self.height_tolerance_m:
                        conflict_indices.add(idx_i)
                        conflict_indices.add(idx_j)
                        break
                    if self._future_segments_intersect(
                            (x_i, y_i, z_i, h_i), (x_j, y_j, z_j, h_j),
                            speed_i, speed_j, self.T):
                        conflict_indices.add(idx_i)
                        conflict_indices.add(idx_j)
                        break
        return sorted(list(conflict_indices))

    @staticmethod
    def _wp_lonlat_to_m(wp: Dict, lat_ref: float) -> Tuple[float, float, float]:
        kx = 111000.0 * np.cos(np.radians(lat_ref))
        ky = 111000.0
        return (
            float(wp['lon']) * kx,
            float(wp['lat']) * ky,
            float(wp.get('height', 0.0)),
        )

    def _interpolate_geo_at_t(
        self, path: List[Dict], t: float, lat_ref: float
    ) -> Optional[Tuple[float, float, float, float]]:
        if len(path) < 2:
            return None
        t = float(t)
        t0 = float(path[0].get('_t', 0.0))
        tn = float(path[-1].get('_t', t0))
        if tn <= t0 + 1e-9:
            return None

        if t <= t0:
            seg, alpha = 0, 0.0
        elif t >= tn:
            seg, alpha = len(path) - 2, 1.0
        else:
            seg = None
            for idx in range(len(path) - 1):
                ta = float(path[idx].get('_t', t0))
                tb = float(path[idx + 1].get('_t', tn))
                if ta <= t <= tb:
                    seg = idx
                    alpha = (t - ta) / max(tb - ta, 1e-9)
                    break
            if seg is None:
                return None

        a, b = path[seg], path[seg + 1]
        lon = (1.0 - alpha) * float(a['lon']) + alpha * float(b['lon'])
        lat_pt = (1.0 - alpha) * float(a['lat']) + alpha * float(b['lat'])
        h = (1.0 - alpha) * float(a.get('height', 0.0)) + alpha * float(
            b.get('height', 0.0))
        x1, y1, _ = self._wp_lonlat_to_m(a, lat_ref)
        x2, y2, _ = self._wp_lonlat_to_m(b, lat_ref)
        dx, dy = x2 - x1, y2 - y1
        if dx * dx + dy * dy < 1e-12:
            heading = 0.0
        else:
            heading = float(np.arctan2(dy, dx))
        kx = 111000.0 * np.cos(np.radians(lat_ref))
        ky = 111000.0
        x, y = lon * kx, lat_pt * ky
        return x, y, h, heading

    def build_sampled_states_geo(
        self, path: List[Dict], lat_ref: float
    ) -> List[Tuple[float, float, float, float, float]]:
        if len(path) < 2:
            return []
        t_start, t_end = _sanitize_path_time_window(path)
        span = t_end - t_start
        if span <= 1e-9:
            return []
        n = int(np.ceil(span / max(self.time_step, 1e-6))) + 1
        n = max(2, min(n, _MAX_GEO_SAMPLES))
        times = np.linspace(t_start, t_end, n, dtype=np.float64)
        states: List[Tuple[float, float, float, float, float]] = []
        for t in times:
            q = self._interpolate_geo_at_t(path, float(t), lat_ref)
            if q is None:
                continue
            x, y, z, heading = q
            states.append((float(t), x, y, z, heading))
        return states

    def pairwise_conflict_from_states(
        self,
        states_i: List[Tuple[float, float, float, float, float]],
        states_j: List[Tuple[float, float, float, float, float]],
        speed_i: float,
        speed_j: float,
        radius_sum: float,
    ) -> bool:
        """在已采样好的四维状态序列上做一次双机判定（避免对 path 重复 build）。"""
        if not states_i or not states_j:
            return False
        speed_i = float(max(speed_i, 1e-6))
        speed_j = float(max(speed_j, 1e-6))
        R = float(radius_sum)

        t_i_start, t_i_end = states_i[0][0], states_i[-1][0]
        t_j_start, t_j_end = states_j[0][0], states_j[-1][0]
        if t_i_end < t_j_start or t_j_end < t_i_start:
            return False

        overlap_start = max(t_i_start, t_j_start)
        overlap_end = min(t_i_end, t_j_end)
        overlap_span = overlap_end - overlap_start
        if overlap_span <= 0:
            return False

        pair_n = int(np.ceil(overlap_span / max(self.time_step, 1e-6))) + 1
        pair_n = max(2, min(pair_n, _MAX_GEO_SAMPLES))
        for t in np.linspace(overlap_start, overlap_end, pair_n, dtype=np.float64):
            state_i = self._interpolate_state(states_i, float(t))
            state_j = self._interpolate_state(states_j, float(t))
            if state_i is None or state_j is None:
                continue
            _, x_i, y_i, z_i, h_i = state_i
            _, x_j, y_j, z_j, h_j = state_j
            dist_hor = float(np.hypot(x_i - x_j, y_i - y_j))
            if dist_hor < R and abs(z_i - z_j) < self.height_tolerance_m:
                return True
            if self._future_segments_intersect(
                (x_i, y_i, z_i, h_i),
                (x_j, y_j, z_j, h_j),
                speed_i, speed_j, self.T,
            ):
                return True
        return False

    def pairwise_conflict_geo(
        self,
        path_a: List[Dict],
        path_b: List[Dict],
        lat_ref: float,
        speed_a: float,
        speed_b: float,
        radius_a: float,
        radius_b: float,
    ) -> bool:
        states_i = self.build_sampled_states_geo(path_a, lat_ref)
        states_j = self.build_sampled_states_geo(path_b, lat_ref)
        spd_i = float(max(speed_a, 1e-6))
        spd_j = float(max(speed_b, 1e-6))
        return self.pairwise_conflict_from_states(
            states_i,
            states_j,
            spd_i,
            spd_j,
            float(radius_a) + float(radius_b),
        )


class SimpleUAV:
    def __init__(self, speed, safety_radius):
        self.speed = speed
        self.safety_radius = safety_radius


if __name__ == "__main__":
    path1 = [[0, 0, 100], [100, 0, 100], [200, 50, 100]]
    path2 = [[0, 30, 100], [100, 30, 100], [200, 30, 100]]

    uav1 = SimpleUAV(speed=20.0, safety_radius=15.0)
    uav2 = SimpleUAV(speed=20.0, safety_radius=15.0)

    tasks = [
        ({'uav': uav1, 'takeoff_time': 0}, path1),
        ({'uav': uav2, 'takeoff_time': 0}, path2),
    ]

    detector = UAVCollisionDetector(time_step=0.5, T=15.0)
    conflicts = detector.detect_conflicts(tasks)
    print("冲突的无人机索引:", conflicts)
