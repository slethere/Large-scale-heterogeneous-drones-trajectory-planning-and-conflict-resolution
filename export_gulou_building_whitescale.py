#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
南京市鼓楼区建筑白模导出（3D + 俯视）

从 nanjing_buildings.csv 按鼓楼区近似经纬范围筛选，挤出多边形柱体并保存 PNG。
运行（在 algrithm 目录）:
    python export_gulou_building_whitescale.py
"""

from __future__ import annotations

import ast
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 鼓楼区近似范围（WGS84，可按需微调）
GULOU_LON_MIN = 118.72
GULOU_LON_MAX = 118.82
GULOU_LAT_MIN = 32.04
GULOU_LAT_MAX = 32.12

_BASE = os.path.dirname(os.path.abspath(__file__))
_BUILDINGS_CSV = os.path.join(_BASE, 'nanjing_buildings.csv')
if not os.path.isfile(_BUILDINGS_CSV):
    _BUILDINGS_CSV = os.path.join(os.path.dirname(_BASE), 'nanjing_buildings', 'nanjing_buildings.csv')

OUT_3D = os.path.join(_BASE, 'gulou_buildings_whitescale_3d.png')
OUT_2D = os.path.join(_BASE, 'gulou_buildings_whitescale_2d.png')


def _parse_vertices(raw) -> list:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw.strip())
        except (SyntaxError, ValueError):
            return []
    return list(raw)


def _extrude_faces(lons: np.ndarray, lats: np.ndarray, h: float) -> list:
    """底面四角 + 顶面 → 5 个四边形面（白模柱体）。"""
    x0, x1 = float(lons.min()), float(lons.max())
    y0, y1 = float(lats.min()), float(lats.max())
    if x1 - x0 < 1e-9 or y1 - y0 < 1e-9:
        return []
    v = [
        [x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0],
        [x0, y0, h], [x1, y0, h], [x1, y1, h], [x0, y1, h],
    ]
    return [
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[0], v[3], v[7], v[4]],
        [v[1], v[2], v[6], v[5]],
        [v[4], v[5], v[6], v[7]],
    ]


def _polygon_faces(verts: list, h: float) -> list:
    if len(verts) < 3:
        return []
    xs = [float(p[0]) for p in verts]
    ys = [float(p[1]) for p in verts]
    n = len(xs)
    bottom = [[xs[i], ys[i], 0.0] for i in range(n)]
    top = [[xs[i], ys[i], h] for i in range(n)]
    faces = [bottom, top]
    for i in range(n):
        j = (i + 1) % n
        faces.append([bottom[i], bottom[j], top[j], top[i]])
    return faces


def load_gulou_buildings(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    m = (
        (df['longitude'] >= GULOU_LON_MIN) &
        (df['longitude'] <= GULOU_LON_MAX) &
        (df['latitude'] >= GULOU_LAT_MIN) &
        (df['latitude'] <= GULOU_LAT_MAX)
    )
    sub = df.loc[m].copy()
    return sub


def collect_all_faces(sub: pd.DataFrame) -> list:
    all_faces = []
    for _, row in sub.iterrows():
        h = float(row.get('height', 0) or 0)
        if h <= 0:
            continue
        verts = _parse_vertices(row.get('polygon_vertices'))
        if len(verts) >= 3:
            fs = _polygon_faces(verts, h)
        else:
            lon = float(row['longitude'])
            lat = float(row['latitude'])
            fs = _extrude_faces(
                np.array([lon - 1e-5, lon + 1e-5]),
                np.array([lat - 1e-5, lat + 1e-5]),
                h,
            )
        all_faces.extend(fs)
    return all_faces


def _white_props(h: float) -> dict:
    t = float(np.clip(h / 200.0, 0.0, 1.0))
    gray = 0.88 + 0.10 * t
    return {
        'facecolor': (gray, gray, gray),
        'edgecolor': '#B0B0B0',
        'alpha': 0.92,
    }


def plot_3d_whitescale(faces: list, out_path: str) -> None:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(14, 10), facecolor='#E8E8E8')
    ax = fig.add_subplot(111, projection='3d', facecolor='#E8E8E8')
    ax.set_title(
        f'南京市鼓楼区建筑白模（{GULOU_LON_MIN}°–{GULOU_LON_MAX}°E, '
        f'{GULOU_LAT_MIN}°–{GULOU_LAT_MAX}°N）',
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel('经度 (°)')
    ax.set_ylabel('纬度 (°)')
    ax.set_zlabel('高度 (m)')

    # 分批绘制，避免单次 Poly3DCollection 面数过多
    batch = 800
    for i in range(0, len(faces), batch):
        chunk = faces[i : i + batch]
        coll = Poly3DCollection(
            chunk,
            facecolors=(0.92, 0.92, 0.92),
            edgecolors='#C8C8C8',
            linewidths=0.08,
            alpha=0.9,
        )
        ax.add_collection3d(coll)

    xs, ys, zs = [], [], []
    for f in faces:
        for p in f:
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
    if xs:
        ax.set_xlim(min(xs), max(xs))
        ax.set_ylim(min(ys), max(ys))
        ax.set_zlim(0, max(zs) * 1.05)
    ax.view_init(elev=28, azim=-58)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'已保存 3D 白模: {out_path}')


def plot_2d_footprint(sub: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#F0F0F0')
    ax.set_facecolor('#F0F0F0')
    ax.set_title('南京市鼓楼区建筑底面轮廓（俯视）', fontsize=14)
    ax.set_xlabel('经度 (°)')
    ax.set_ylabel('纬度 (°)')
    ax.set_aspect('equal')

    for _, row in sub.iterrows():
        verts = _parse_vertices(row.get('polygon_vertices'))
        if len(verts) >= 3:
            xs = [p[0] for p in verts] + [verts[0][0]]
            ys = [p[1] for p in verts] + [verts[0][1]]
            ax.fill(xs, ys, facecolor='#DDDDDD', edgecolor='#AAAAAA', linewidth=0.3)
        else:
            ax.plot(
                float(row['longitude']),
                float(row['latitude']),
                's',
                color='#CCCCCC',
                ms=2,
            )

    ax.set_xlim(GULOU_LON_MIN, GULOU_LON_MAX)
    ax.set_ylim(GULOU_LAT_MIN, GULOU_LAT_MAX)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'已保存 2D 俯视: {out_path}')


def main() -> int:
    if not os.path.isfile(_BUILDINGS_CSV):
        print(f'未找到建筑数据: {_BUILDINGS_CSV}', file=sys.stderr)
        return 1

    sub = load_gulou_buildings(_BUILDINGS_CSV)
    print(f'数据文件: {_BUILDINGS_CSV}')
    print(f'鼓楼区范围内建筑: {len(sub)} 栋')
    if sub.empty:
        print(
            '范围内无建筑。可放宽 GULOU_LON_MIN/MAX、GULOU_LAT_MIN/MAX，'
            '或确认 CSV 是否覆盖鼓楼。',
            file=sys.stderr,
        )
        return 2

    faces = collect_all_faces(sub)
    print(f'挤出面片数: {len(faces)}')
    if not faces:
        print('无有效高度面片。', file=sys.stderr)
        return 3

    plot_3d_whitescale(faces, OUT_3D)
    plot_2d_footprint(sub, OUT_2D)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
