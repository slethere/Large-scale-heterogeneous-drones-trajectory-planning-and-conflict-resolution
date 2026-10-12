#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MASAC 任务配对工具：训练/推理共用。"""

from __future__ import annotations

import math
from typing import Any, List, Set, Tuple

# 与 SAC2 宏观 distancing 对齐
DEFAULT_NEAR_KM = 0.8
DEFAULT_MAX_DT_S = 120.0


def _haversine_km(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
    lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * math.asin(math.sqrt(a)) * 6371.0


def _spatial_near(task_a: Any, task_b: Any, near_km: float) -> bool:
    return (
        _haversine_km(task_a.sender, task_b.sender) < near_km
        or _haversine_km(task_a.receiver, task_b.receiver) < near_km
    )


def _pair_distance(task_a: Any, task_b: Any) -> float:
    ds = _haversine_km(task_a.sender, task_b.sender)
    dr = _haversine_km(task_a.receiver, task_b.receiver)
    return min(ds, dr)


def build_spatial_task_pairs(
    tasks: List[Any],
    near_km: float = DEFAULT_NEAR_KM,
) -> List[Tuple[Any, Any]]:
    """
    从任务列表构造空间相近对；每个 task 最多参与 1 对（贪心按距离升序）。
    """
    n = len(tasks)
    if n < 2:
        return []

    candidates: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if _spatial_near(tasks[i], tasks[j], near_km):
                candidates.append((_pair_distance(tasks[i], tasks[j]), i, j))
    candidates.sort(key=lambda x: x[0])

    used: Set[int] = set()
    pairs: List[Tuple[Any, Any]] = []
    for _, i, j in candidates:
        if i in used or j in used:
            continue
        pairs.append((tasks[i], tasks[j]))
        used.add(i)
        used.add(j)
    return pairs


def filter_temporal_overlap(
    pairs: List[Tuple[Any, Any]],
    max_dt_s: float = DEFAULT_MAX_DT_S,
) -> List[Tuple[Any, Any]]:
    """保留起飞时刻差 <= max_dt_s 的任务对。"""
    out: List[Tuple[Any, Any]] = []
    for ta, tb in pairs:
        ta_ts = float(getattr(ta, 'dronetime_ts', 0) or 0)
        tb_ts = float(getattr(tb, 'dronetime_ts', 0) or 0)
        if abs(ta_ts - tb_ts) <= max_dt_s:
            out.append((ta, tb))
    return out


def pair_assigned_tasks(
    uav_tasks: List[List[Any]],
    near_km: float = DEFAULT_NEAR_KM,
    max_dt_s: float = DEFAULT_MAX_DT_S,
) -> Tuple[List[Tuple[Any, Any, int, int]], List[Tuple[int, Any]]]:
    """
    从专家系统分配结果拆出 MASAC 联合规划对与单机任务。

    Returns:
        pairs: [(task_a, task_b, uav_idx_a, uav_idx_b), ...]
        singles: [(uav_idx, task), ...]  未配对的任务
    """
    flat: List[Tuple[int, Any]] = []
    for uav_idx, ts in enumerate(uav_tasks):
        for t in ts:
            flat.append((uav_idx, t))

    if len(flat) < 2:
        return [], flat

    tasks_only = [t for _, t in flat]
    spatial = build_spatial_task_pairs(tasks_only, near_km=near_km)
    spatial = filter_temporal_overlap(spatial, max_dt_s=max_dt_s)

    id_to_slot = {id(t): (uav_idx, t) for uav_idx, t in flat}
    used_ids: Set[int] = set()
    pairs_out: List[Tuple[Any, Any, int, int]] = []

    for ta, tb in spatial:
        if id(ta) in used_ids or id(tb) in used_ids:
            continue
        ua, _ = id_to_slot.get(id(ta), (-1, ta))
        ub, _ = id_to_slot.get(id(tb), (-1, tb))
        if ua < 0 or ub < 0:
            continue
        pairs_out.append((ta, tb, ua, ub))
        used_ids.add(id(ta))
        used_ids.add(id(tb))

    singles = [(u, t) for u, t in flat if id(t) not in used_ids]
    return pairs_out, singles
