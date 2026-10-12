#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
masac_train.py — CTDE-MASAC 双机路径规划训练。

用法:
    python masac_train.py [--episodes 15000] [--resume]

输出（均在 masac_models/ 目录）:
    masac_models/masac_actor.pth
    masac_models/masac_epXXX.pth
    masac_models/masac_train_history.npz   # 奖励/成功率序列，可单独重绘
    masac_models/masac_training_curve.png

重绘曲线（无需重新训练，需已有 history.npz）:
    python masac_train.py --plot-only
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drone_obstacle_avoidance import Drone, DronePathPlanner, DeliveryTask, load_wind_timeseries
from drone_obstacle_avoidance import FLEET_FIXED_WING_RATIO
from masac_path_env import (
    GLOBAL_STATE_DIM,
    JOINT_ACTION_DIM,
    LOCAL_OBS_DIM,
    MASACPathEnv,
    MAX_STEPS,
    NUM_AGENTS,
)
from masac_utils import build_spatial_task_pairs
from sac_path_env import ACTION_DIM, _deg2m_x, _deg2m_y

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DELIVERY_CSV = os.path.join(BASE_DIR, 'nanjing_delivery_700_2026_01.csv')
BUILDINGS_CSV = os.path.join(BASE_DIR, 'nanjing_buildings.csv')
WIND_CSV = os.path.join(BASE_DIR, 'wind.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'masac_models')
ACTOR_PATH = os.path.join(MODEL_DIR, 'masac_actor.pth')
HISTORY_PATH = os.path.join(MODEL_DIR, 'masac_train_history.npz')
CURVE_PATH = os.path.join(MODEL_DIR, 'masac_training_curve.png')

EP_MAX = 15000
GAMMA = 0.95
TAU = 1e-2
BATCH_SIZE = 128
MEM_CAPACITY = 200000
POLICY_LR = 1e-3
Q_LR = 3e-4
ALPHA_LR = 3e-4
TARGET_ENT = -float(ACTION_DIM)
HIDDEN = 256
SAVE_EVERY = 500
FIXED_WING_RATIO = FLEET_FIXED_WING_RATIO   # 5 固翼 + 25 多旋翼 → 1/6
# 课程学习分阶段（与 EP_MAX=10000 匹配；按配对内较长航段距离筛选）
CURRICULUM_PHASES = [
    (0,     1500,  2000),    # 阶段1：≤2km
    (1500,  4000,  3500),    # 阶段2：≤3.5km
    (4000,  7500,  5500),    # 阶段3：≤5.5km
    (7500,  12000,  8000),    # 阶段4：≤8km
    (12000,  15000, None),    # 阶段5：全部
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(MODEL_DIR, exist_ok=True)


class SharedActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LOCAL_OBS_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.mean = nn.Linear(HIDDEN, ACTION_DIM)
        self.log_std = nn.Linear(HIDDEN, ACTION_DIM)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mean(x)
        lsg = torch.clamp(self.log_std(x), -20, 2)
        return mu, lsg.exp()

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, std = self(x)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z)
        log_p = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        return a, log_p.sum(-1, keepdim=True)


class CentralCritic(nn.Module):
    def __init__(self):
        super().__init__()
        inp = GLOBAL_STATE_DIM + JOINT_ACTION_DIM

        def _mlp():
            return nn.Sequential(
                nn.Linear(inp, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, 1),
            )

        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, gs: torch.Tensor, ja: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([gs, ja], dim=-1)
        return self.q1(x), self.q2(x)


class MASACReplayBuffer:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.buf: List = []
        self.ptr = 0

    def push(self, o0, o1, a0, a1, r0, r1, o0n, o1n, done):
        if len(self.buf) < self.cap:
            self.buf.append(None)
        self.buf[self.ptr] = (
            np.array(o0, dtype=np.float32),
            np.array(o1, dtype=np.float32),
            np.array(a0, dtype=np.float32),
            np.array(a1, dtype=np.float32),
            float(r0), float(r1),
            np.array(o0n, dtype=np.float32),
            np.array(o1n, dtype=np.float32),
            float(done),
        )
        self.ptr = (self.ptr + 1) % self.cap

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        o0, o1, a0, a1, r0, r1, o0n, o1n, d = zip(*batch)
        to = lambda x: torch.FloatTensor(np.array(x)).to(device)
        gs = np.concatenate([np.array(o0), np.array(o1)], axis=-1)
        gsn = np.concatenate([np.array(o0n), np.array(o1n)], axis=-1)
        ja = np.concatenate([np.array(a0), np.array(a1)], axis=-1)
        r_team = 0.5 * (np.array(r0) + np.array(r1))
        return (
            to(gs),
            to(ja),
            torch.FloatTensor(r_team).unsqueeze(1).to(device),
            to(gsn),
            torch.FloatTensor(d).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buf)


class MASACAgent:
    """CTDE-MASAC：共享 Actor + 联合 Critic。"""

    def __init__(self):
        self.actor = SharedActor().to(device)
        self.critic = CentralCritic().to(device)
        self.critic_target = CentralCritic().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=POLICY_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=Q_LR)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=ALPHA_LR)
        self.buffer = MASACReplayBuffer(MEM_CAPACITY)

    def select_actions(
        self,
        obs_list: List[np.ndarray],
        deterministic: bool = False,
    ) -> List[np.ndarray]:
        actions = []
        for obs in obs_list:
            s = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                if deterministic:
                    mu, _ = self.actor(s)
                    a = torch.tanh(mu).cpu().numpy()[0]
                else:
                    a, _ = self.actor.sample(s)
                    a = a.cpu().numpy()[0]
            actions.append(a)
        return actions

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        gs, ja, r, gsn, done = self.buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            o0n = gsn[:, :LOCAL_OBS_DIM]
            o1n = gsn[:, LOCAL_OBS_DIM:]
            a0n, log_p0 = self.actor.sample(o0n)
            a1n, log_p1 = self.actor.sample(o1n)
            jan = torch.cat([a0n, a1n], dim=-1)
            q1n, q2n = self.critic_target(gsn, jan)
            log_p = log_p0 + log_p1
            q_tgt = r + GAMMA * (1 - done) * (torch.min(q1n, q2n) - self.alpha * log_p)

        q1, q2 = self.critic(gs, ja)
        c_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)
        self.critic_opt.zero_grad()
        c_loss.backward()
        self.critic_opt.step()

        o0 = gs[:, :LOCAL_OBS_DIM]
        o1 = gs[:, LOCAL_OBS_DIM:]
        a0, log_p0 = self.actor.sample(o0)
        a1, log_p1 = self.actor.sample(o1)
        ja_new = torch.cat([a0, a1], dim=-1)
        q1n, q2n = self.critic(gs, ja_new)
        log_p = log_p0 + log_p1
        a_loss = (self.alpha * log_p - torch.min(q1n, q2n)).mean()
        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()

        ent_loss = -(self.log_alpha * (log_p + TARGET_ENT * NUM_AGENTS).detach()).mean()
        self.alpha_opt.zero_grad()
        ent_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp()

        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.actor.load_state_dict(ckpt['actor'])
        if 'critic' in ckpt:
            self.critic.load_state_dict(ckpt['critic'])
        if 'critic_target' in ckpt:
            self.critic_target.load_state_dict(
                ckpt.get('critic_target', ckpt['critic']))
        if 'log_alpha' in ckpt:
            with torch.no_grad():
                self.log_alpha.copy_(ckpt['log_alpha'].to(device))
            self.alpha = self.log_alpha.exp()


def _load_training_data():
    from alns_obstacle_wind import load_delivery_tasks

    alns_tasks = load_delivery_tasks(DELIVERY_CSV)
    buildings_df = pd.read_csv(BUILDINGS_CSV)
    planner = DronePathPlanner(buildings_df)
    buildings = planner.buildings

    delivery_tasks = []
    for t in alns_tasks:
        dt = DeliveryTask(
            task_id=t.task_id,
            sender_lon=t.sender[0],
            sender_lat=t.sender[1],
            receiver_lon=t.receiver[0],
            receiver_lat=t.receiver[1],
        )
        dt.dronetime = t.dronetime
        dt.order_cnt = t.order_cnt
        delivery_tasks.append(dt)
    lat_ref = float(np.mean([t.sender_lat for t in delivery_tasks]))
    return alns_tasks, delivery_tasks, buildings, lat_ref


def _alns_task_dist_m(t, lat_ref: float) -> float:
    dx = _deg2m_x(t.receiver[0] - t.sender[0], lat_ref)
    dy = _deg2m_y(t.receiver[1] - t.sender[1])
    return float(np.hypot(dx, dy))


def _pair_max_dist_m(ta, tb, lat_ref: float) -> float:
    return max(_alns_task_dist_m(ta, lat_ref), _alns_task_dist_m(tb, lat_ref))


def _get_curriculum_pairs(
    ep: int,
    pairs_with_dist: List[Tuple],
    all_pairs: List[Tuple],
) -> List[Tuple]:
    for ep_start, ep_end, max_dist in CURRICULUM_PHASES:
        if ep_start <= ep < ep_end:
            if max_dist is None:
                return all_pairs
            filtered = [(a, b) for a, b, d in pairs_with_dist if d <= max_dist]
            return filtered or all_pairs
    return all_pairs


def _curriculum_label(ep: int) -> str:
    for ep_start, ep_end, max_dist in CURRICULUM_PHASES:
        if ep_start <= ep < ep_end:
            if max_dist is None:
                return '全距离'
            return f'≤{max_dist / 1000:.0f}km'
    return '全距离'


def _alns_to_delivery(t) -> DeliveryTask:
    dt = DeliveryTask(
        task_id=t.task_id,
        sender_lon=t.sender[0],
        sender_lat=t.sender[1],
        receiver_lon=t.receiver[0],
        receiver_lat=t.receiver[1],
    )
    dt.dronetime = t.dronetime
    dt.order_cnt = t.order_cnt
    return dt


def train(ep_max: int = EP_MAX, resume: bool = False):
    print('=' * 60)
    print('  MASAC 双机路径规划训练 (CTDE K=2)')
    print(f'  设备: {device}  |  训练轮数: {ep_max}')
    print('=' * 60)

    alns_tasks, delivery_tasks, buildings, lat_ref = _load_training_data()
    all_pairs = build_spatial_task_pairs(alns_tasks)
    pairs_with_dist = [
        (ta, tb, _pair_max_dist_m(ta, tb, lat_ref))
        for ta, tb in all_pairs
    ]
    pairs_with_dist.sort(key=lambda x: x[2])
    print(f'  订单数: {len(alns_tasks)}  |  建筑数: {len(buildings)}')
    print(f'  空间相近训练对: {len(all_pairs)}')
    if pairs_with_dist:
        print(f'  配对航段距离: {pairs_with_dist[0][2]:.0f}m ~ {pairs_with_dist[-1][2]:.0f}m')
    n_short = sum(1 for _, _, d in pairs_with_dist if d <= 2000)
    n_mid = sum(1 for _, _, d in pairs_with_dist if d <= 8000)
    print(f'  课程可用对: ≤2km={n_short}  ≤8km={n_mid}  全部={len(all_pairs)}')
    if not all_pairs:
        print('  警告: 无训练对，退出')
        return None

    wind_ts = load_wind_timeseries(WIND_CSV)
    mr_drone = Drone(1, callsign='MASAC-MR', craft_type='multirotor')
    fw_drone = Drone(2, callsign='MASAC-FW', craft_type='pure_fixed_wing')

    env = MASACPathEnv(
        buildings=buildings,
        lat_ref=lat_ref,
        wind_lookup=wind_ts if len(wind_ts) > 0 else None,
    )
    agent = MASACAgent()

    if resume and os.path.exists(ACTOR_PATH):
        try:
            agent.load(ACTOR_PATH)
            print('  接续上次训练')
        except Exception as e:
            print(f'  加载失败({e})，从头训练')
    else:
        print('  从头训练')

    ep_rewards: List[float] = []
    ep_joint_success: List[float] = []
    ep_conflict_rate: List[float] = []

    for ep in range(ep_max):
        pairs = _get_curriculum_pairs(ep, pairs_with_dist, all_pairs)
        ta, tb = random.choice(pairs)
        dt_a = _alns_to_delivery(ta)
        dt_b = _alns_to_delivery(tb)
        d_a = fw_drone if random.random() < FIXED_WING_RATIO else mr_drone
        d_b = fw_drone if random.random() < FIXED_WING_RATIO else mr_drone

        obs = env.reset([dt_a, dt_b], [d_a, d_b])
        ep_r = 0.0
        conflict_steps = 0
        total_steps = 0

        for _ in range(MAX_STEPS):
            actions = agent.select_actions(obs, deterministic=False)
            obs_n, rewards, done, info = env.step(actions)
            agent.buffer.push(
                obs[0], obs[1], actions[0], actions[1],
                rewards[0], rewards[1], obs_n[0], obs_n[1], float(done),
            )
            agent.update()
            ep_r += rewards[0] + rewards[1]
            total_steps += 1
            if info.get('team_conflict'):
                conflict_steps += 1
            obs = obs_n
            if done:
                break

        succ = info.get('success', [False, False])
        joint_ok = float(succ[0] and succ[1])
        ep_rewards.append(ep_r)
        ep_joint_success.append(joint_ok)
        ep_conflict_rate.append(conflict_steps / max(total_steps, 1))

        if (ep + 1) % 50 == 0:
            print(f'  Ep {ep+1:5d}/{ep_max}  '
                  f'课程[{_curriculum_label(ep)}]  '
                  f'奖励 {np.mean(ep_rewards[-50:]):8.1f}  '
                  f'双机成功率 {np.mean(ep_joint_success[-50:])*100:5.1f}%  '
                  f'冲突步占比 {np.mean(ep_conflict_rate[-50:])*100:5.1f}%  '
                  f'α={float(agent.alpha):.4f}  '
                  f'缓冲 {len(agent.buffer)}')

        if (ep + 1) % SAVE_EVERY == 0:
            ckpt = os.path.join(MODEL_DIR, f'masac_ep{ep+1}.pth')
            agent.save(ckpt)
            print(f'    检查点: {ckpt}')

    agent.save(ACTOR_PATH)
    print(f'\n  最终模型: {ACTOR_PATH}')

    np.savez_compressed(
        HISTORY_PATH,
        ep_rewards=np.asarray(ep_rewards, dtype=np.float32),
        ep_joint_success=np.asarray(ep_joint_success, dtype=np.float32),
        ep_conflict_rate=np.asarray(ep_conflict_rate, dtype=np.float32),
    )
    print(f'  训练历史: {HISTORY_PATH}')
    plot_training_curve(ep_rewards, ep_joint_success, CURVE_PATH)
    return agent


def plot_training_curve(
    ep_rewards,
    ep_joint_success,
    out_png: str = CURVE_PATH,
    start_ep: int = 4000,
    end_ep: int = 14000,
) -> str:
    """绘制训练曲线；图例放在图外下方，避免遮挡曲线。

    start_ep / end_ep: 横轴显示区间（默认 4000–14000，含端点）。
    """
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    ep_rewards = np.asarray(ep_rewards, dtype=np.float64)
    ep_joint_success = np.asarray(ep_joint_success, dtype=np.float64)
    n = len(ep_rewards)
    start = int(max(0, min(start_ep, max(n - 1, 0))))
    end = int(end_ep) if end_ep is not None else (n - 1)
    end = int(max(start, min(end, n - 1)))

    from matplotlib import font_manager

    # Windows 优先 Times New Roman；找不到则回退 serif
    _tnr = 'Times New Roman'
    try:
        font_manager.findfont(_tnr, fallback_to_default=False)
    except Exception:
        _tnr = 'serif'
    plt.rcParams.update({
        'font.family': _tnr,
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 11,
        'axes.unicode_minus': False,
        'mathtext.fontset': 'stix',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })
    fp = font_manager.FontProperties(family=_tnr)

    fig, ax1 = plt.subplots(figsize=(11, 5.2))
    lines = []
    x = np.arange(n)
    m_raw = (x >= start) & (x <= end)
    lines += ax1.plot(
        x[m_raw], ep_rewards[m_raw],
        alpha=0.25, color='#4C78A8', lw=0.8, label='episode reward',
    )
    if n >= 50:
        kernel = np.ones(50) / 50
        ma_r = np.convolve(ep_rewards, kernel, mode='valid')
        x_ma = np.arange(len(ma_r))
        m = (x_ma >= start) & (x_ma <= end)
        lines += ax1.plot(
            x_ma[m], ma_r[m],
            color='#F58518', lw=2.0, label='reward ma50',
        )
    ax1.set_xlabel('Episode', fontproperties=fp)
    ax1.set_ylabel('Team reward', fontproperties=fp)
    ax1.set_xlim(start, end)
    ax1.grid(True, alpha=0.25)
    ax1.tick_params(direction='in', top=True, right=False, which='both', length=5)
    for lab in ax1.get_xticklabels() + ax1.get_yticklabels():
        lab.set_fontproperties(fp)

    ax2 = ax1.twinx()
    if len(ep_joint_success) >= 50:
        ma_s = np.convolve(ep_joint_success, np.ones(50) / 50, mode='valid')
        x_ms = np.arange(len(ma_s))
        m2 = (x_ms >= start) & (x_ms <= end)
        lines += ax2.plot(
            x_ms[m2], ma_s[m2],
            color='#54A24B', lw=2.0, label='joint success ma50',
        )
    ax2.set_ylabel('Joint success', fontproperties=fp)
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(direction='in', which='both', length=5)
    for lab in ax2.get_yticklabels():
        lab.set_fontproperties(fp)

    labels = [ln.get_label() for ln in lines]
    ax1.legend(
        lines, labels,
        loc='upper center', bbox_to_anchor=(0.5, -0.16),
        ncol=3, frameon=False, fontsize=10,
        prop=fp,
    )
    ax1.set_title('MASAC Training', pad=10, fontproperties=fp)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  训练曲线: {out_png}  （显示 Episode {start}–{end}，字体 {_tnr}）')
    return out_png


def plot_from_history(history_path: str = HISTORY_PATH, out_png: str = CURVE_PATH) -> str:
    """从 masac_train_history.npz 重绘曲线（不依赖 .pth）。"""
    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f'未找到训练历史 {history_path}。\n'
            '说明: .pth 只含网络权重，不含 reward/success 序列，无法据此重绘。\n'
            '需重新训练一次（新版本会自动写出 history.npz），或提供已有 npz。'
        )
    data = np.load(history_path)
    return plot_training_curve(
        data['ep_rewards'],
        data['ep_joint_success'],
        out_png,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=EP_MAX)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--plot-only', action='store_true',
        help='仅从 masac_train_history.npz 重绘曲线，不训练',
    )
    args = parser.parse_args()
    if args.plot_only:
        plot_from_history()
        return
    train(ep_max=args.episodes, resume=args.resume)


if __name__ == '__main__':
    main()
