"""
ddpg_train.py  —  DDPG 无人机路径规划训练脚本
===============================================
独立运行即可开始训练：
    python ddpg_train.py [--episodes 2000] [--resume]

训练完成后输出：
    ddpg_actor.pth               (推理时主函数调用，与 sac_actor.pth 对称)
    ddpg_models/ddpg_epXXX.pth  (检查点)
    ddpg_training_curve.png      (训练曲线)

算法：DDPG（Deep Deterministic Policy Gradient，Lillicrap et al. 2016）
  · Actor  ：确定性策略，s → a in [-1, 1]（tanh 输出）
  · Critic ：Q(s, a) → 标量值（评价当前策略）
  · 探索   ：Ornstein-Uhlenbeck 噪声（自相关连续噪声，适合连续动作空间）
  · 记忆库 ：Experience Replay Buffer（与 SAC 相同）
  · 更新   ：软更新（Polyak averaging, tau=0.01）

与 SAC 的核心区别：
  · 确定性策略（Actor 输出固定动作，不采样分布）
  · 不需要熵正则化和温度 alpha
  · 单 Q 网络（非双 Q）
  · OU 噪声替代 reparameterization 探索
  · Actor/Critic 网络结构参考 main_DDPG.py（ReLU+Sigmoid+tanh）

复用环境：直接使用 sac_path_env.SACPathEnv（状态/动作空间完全相同）
DDPGAgent 类同时供 alns_obstacle_wind.py 导入用于推理。
"""

import os
import sys
import argparse
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 复用与 SAC 完全相同的路径规划环境（状态/动作维度一致）
from sac_path_env import (
    SACPathEnv, STATE_DIM, ACTION_DIM, MAX_STEPS,
    _deg2m_x, _deg2m_y,
)
from drone_obstacle_avoidance import (
    DeliveryTask, DronePathPlanner, Drone,
    load_wind_timeseries,
)

# ══════════════════════════════════════════════════════════════════════════════
# § 超参数
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DELIVERY_CSV  = os.path.join(BASE_DIR, 'nanjing_delivery_700_2026_01.csv')
BUILDINGS_CSV = os.path.join(BASE_DIR, 'nanjing_buildings.csv')
WIND_CSV      = os.path.join(BASE_DIR, 'wind.csv')
MODEL_DIR     = os.path.join(BASE_DIR, 'ddpg_models')
ACTOR_PATH    = os.path.join(BASE_DIR, 'ddpg_actor.pth')

EP_MAX       = 20000
GAMMA        = 0.95
TAU          = 1e-2       # 软更新系数（与 main_DDPG.py 保持一致）
BATCH_SIZE   = 128
MEM_CAPACITY = 50000   # 扩容，覆盖更多训练多样性
ACTOR_LR     = 1e-3       # main_DDPG.py LR_A
CRITIC_LR    = 1e-3       # main_DDPG.py LR_C
HIDDEN       = 256
SAVE_EVERY   = 500     # 减少检查点 I/O 开销

# Ornstein-Uhlenbeck 噪声参数
OU_THETA = 0.15
OU_SIGMA = 0.2

# OU 噪声衰减：前 NOISE_DECAY_EP 轮线性衰减到 0.05（与 EP_MAX=20000 匹配）
NOISE_DECAY_EP = 15000

# 课程学习（与 EP_MAX=20000 匹配）
CURRICULUM_PHASES = [
    (0,     5000,  2000),    # 阶段1：短距离任务（≤2km），建立基础策略
    (5000,  12000, 8000),    # 阶段2：中距离任务（≤8km），扩展泛化能力
    (12000, 20000, None),    # 阶段3：全部任务，完整策略
]

# 训练时随机注入他机路径的概率（消除训练/推理分布偏移）
OTHER_PATH_INJECT_PROB = 0.7   # 提升至 70%，与推理端频率对齐
OTHER_PATH_INJECT_MAX  = 5     # 单次最多注入条数（推理端固定采样 4 条）

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# § 网络定义（参考 main_DDPG.py 的 ActorNet / CriticNet）
# ══════════════════════════════════════════════════════════════════════════════

class DDPGActorNet(nn.Module):
    """
    确定性 Actor：s -> a in [-1, 1]。
    隐藏层全用 ReLU（避免 sigmoid 梯度消失导致策略崩溃），
    输出层 tanh 压缩到 [-1, 1]。
    权重用 Xavier 初始化，防止大初始权重导致 tanh/sigmoid 饱和。
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),    nn.ReLU(),
            nn.Linear(HIDDEN, ACTION_DIM),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(s))   # a in [-1, 1]


class DDPGCriticNet(nn.Module):
    """
    Critic：Q(s, a) -> 标量。隐藏层全用 ReLU，Xavier 初始化。
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),                 nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))   # (B, 1)


# ══════════════════════════════════════════════════════════════════════════════
# § Ornstein-Uhlenbeck 噪声（参考 main_DDPG.py）
# ══════════════════════════════════════════════════════════════════════════════

class OUNoise:
    """
    Ornstein-Uhlenbeck 过程噪声——自相关、均值回归，
    比纯高斯噪声在连续动作空间中探索效果更好。
    """
    def __init__(self, action_dim: int,
                 theta: float = OU_THETA,
                 sigma: float = OU_SIGMA):
        self.mu    = np.zeros(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.x = np.zeros_like(self.mu)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.x) + \
             self.sigma * np.random.randn(len(self.mu))
        self.x += dx
        return self.x.copy()


# ══════════════════════════════════════════════════════════════════════════════
# § 经验回放缓冲
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.buf: List = []
        self.ptr = 0

    def push(self, s, a, r: float, s_next, done: float):
        if len(self.buf) < self.cap:
            self.buf.append(None)
        self.buf[self.ptr] = (
            np.array(s,      dtype=np.float32),
            np.array(a,      dtype=np.float32),
            float(r),
            np.array(s_next, dtype=np.float32),
            float(done),
        )
        self.ptr = (self.ptr + 1) % self.cap

    def sample(self, n: int):
        batch          = random.sample(self.buf, n)
        s, a, r, s_, d = zip(*batch)
        to = lambda x: torch.FloatTensor(np.array(x)).to(device)
        return (
            to(s), to(a),
            torch.FloatTensor(r).unsqueeze(1).to(device),
            to(s_),
            torch.FloatTensor(d).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buf)


# ══════════════════════════════════════════════════════════════════════════════
# § DDPG Agent（供训练和推理共用）
# ══════════════════════════════════════════════════════════════════════════════

class DDPGAgent:
    """
    单智能体 DDPG。
    训练：调用 update()。
    推理：调用 select_action(state, noise_scale=0.0)。
    """

    def __init__(self):
        # Actor：估计网络 + 目标网络
        self.actor        = DDPGActorNet().to(device)
        self.actor_target = DDPGActorNet().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt    = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        # Critic：估计网络 + 目标网络
        self.critic        = DDPGCriticNet().to(device)
        self.critic_target = DDPGCriticNet().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt    = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.buffer   = ReplayBuffer(MEM_CAPACITY)
        self.ou_noise = OUNoise(ACTION_DIM)

    # ── 动作选取 ──────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray,
                      noise_scale: float = 0.0) -> np.ndarray:
        """
        noise_scale > 0 时叠加 OU 噪声（训练探索）；
        noise_scale = 0 时纯确定性推理。
        """
        state = np.nan_to_num(
            np.asarray(state, dtype=np.float32),
            nan=0.0, posinf=1.0, neginf=-1.0,
        )
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]
        if noise_scale > 0:
            noise  = self.ou_noise.sample() * noise_scale
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    # ── 单步更新 ──────────────────────────────────────────────────────────────

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        s, a, r, s_, done = self.buffer.sample(BATCH_SIZE)

        # ── Critic 更新（Bellman TD 目标）──────────────────────────────────────
        with torch.no_grad():
            a_next   = self.actor_target(s_)
            q_next   = self.critic_target(s_, a_next)
            q_target = r + GAMMA * (1 - done) * q_next

        q_pred = self.critic(s, a)
        c_loss = F.mse_loss(q_pred, q_target)

        self.critic_opt.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.critic_opt.step()

        # ── Actor 更新（策略梯度：最大化 Q）──────────────────────────────────────
        a_pred = self.actor(s)
        a_loss = -self.critic(s, a_pred).mean()

        self.actor_opt.zero_grad()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.actor_opt.step()

        # ── 软更新目标网络（Polyak averaging）────────────────────────────────────
        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

    # ── 保存 / 加载 ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            'actor':         self.actor.state_dict(),
            'critic':        self.critic.state_dict(),
            'actor_target':  self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_target.load_state_dict(
            ckpt.get('actor_target', ckpt['actor']))
        self.critic_target.load_state_dict(
            ckpt.get('critic_target', ckpt['critic']))


# ══════════════════════════════════════════════════════════════════════════════
# § 数据加载
# ══════════════════════════════════════════════════════════════════════════════

def _load_training_data():
    from alns_obstacle_wind import load_delivery_tasks

    alns_tasks   = load_delivery_tasks(DELIVERY_CSV)
    buildings_df = pd.read_csv(BUILDINGS_CSV)
    planner      = DronePathPlanner(buildings_df)
    buildings    = planner.buildings

    delivery_tasks = []
    for t in alns_tasks:
        dt = DeliveryTask(
            task_id      = t.task_id,
            sender_lon   = t.sender[0],
            sender_lat   = t.sender[1],
            receiver_lon = t.receiver[0],
            receiver_lat = t.receiver[1],
        )
        dt.dronetime = t.dronetime    # 供 env.reset 从 wind_lookup 查任务时刻的风
        dt.order_cnt = t.order_cnt    # 供 env.reset 设 drone.current_payload_kg
        delivery_tasks.append(dt)
    lat_ref = float(np.mean([t.sender_lat for t in delivery_tasks]))
    return delivery_tasks, buildings, lat_ref


# ══════════════════════════════════════════════════════════════════════════════
# § 训练主函数
# ══════════════════════════════════════════════════════════════════════════════

def train(ep_max: int = EP_MAX, resume: bool = False):
    print("=" * 60)
    print("  DDPG 无人机路径规划训练")
    print(f"  设备: {device}  |  训练轮数: {ep_max}")
    print(f"  OU噪声: theta={OU_THETA}, sigma={OU_SIGMA}"
          f"，前 {NOISE_DECAY_EP} 轮线性衰减至 0.05")
    print("=" * 60)

    delivery_tasks, buildings, lat_ref = _load_training_data()
    print(f"  订单数: {len(delivery_tasks)}  |  建筑数: {len(buildings)}")

    def _task_dist(t: DeliveryTask) -> float:
        dx = _deg2m_x(t.receiver_lon - t.sender_lon, lat_ref)
        dy = _deg2m_y(t.receiver_lat - t.sender_lat)
        return float(np.hypot(dx, dy))

    tasks_with_dist = [(t, _task_dist(t)) for t in delivery_tasks]
    tasks_with_dist.sort(key=lambda x: x[1])
    print(f"  任务距离范围: {tasks_with_dist[0][1]:.0f}m ~ {tasks_with_dist[-1][1]:.0f}m")

    def _get_curriculum_tasks(ep: int) -> list:
        for ep_start, ep_end, max_dist in CURRICULUM_PHASES:
            if ep_start <= ep < ep_end:
                if max_dist is None:
                    return delivery_tasks
                return [t for t, d in tasks_with_dist if d <= max_dist] or delivery_tasks
        return delivery_tasks

    # 时变风 + 双机型 Drone（与 SAC 训练一致）
    wind_ts = load_wind_timeseries(WIND_CSV)
    if len(wind_ts) > 0:
        _mw, _md = wind_ts.mean()
        print(f"  风场: 时变 {len(wind_ts)} 条样本，日均 {_mw:.2f} m/s / {_md:.1f}°")
    else:
        print("  风场: wind.csv 未加载，风/P4 disabled")
    mr_drone = Drone(1, callsign='DDPG-MR', craft_type='multirotor')
    fw_drone = Drone(2, callsign='DDPG-FW', craft_type='fixed_wing')
    FIXED_WING_RATIO = 0.3
    print(f"  双机型训练: MR + FW, ratio FW={FIXED_WING_RATIO:.0%}")

    env   = SACPathEnv(
        buildings   = buildings,
        lat_ref     = lat_ref,
        drone       = mr_drone,
        wind_lookup = wind_ts if len(wind_ts) > 0 else None,
    )
    agent = DDPGAgent()

    if resume and os.path.exists(ACTOR_PATH):
        try:
            agent.load(ACTOR_PATH)
            print("  接续上次训练")
        except Exception as e:
            print(f"  加载失败（{e}），从头训练")
    else:
        print("  从头训练")

    ep_rewards:   List[float] = []
    ep_successes: List[float] = []

    for ep in range(ep_max):
        # OU 噪声尺度：从 1.0 线性衰减到 0.05
        noise_scale = max(0.05, 1.0 - ep / NOISE_DECAY_EP)

        task = random.choice(_get_curriculum_tasks(ep))
        env.set_drone(fw_drone if random.random() < FIXED_WING_RATIO else mr_drone)
        env.set_task(task)
        agent.ou_noise.reset()
        state   = env.reset()
        ep_r    = 0.0
        success = False

        for _ in range(MAX_STEPS):
            action = agent.select_action(state, noise_scale=noise_scale)
            next_s, reward, done, info = env.step(action)
            agent.buffer.push(state, action, reward, next_s, float(done))
            agent.update()
            state  = next_s
            ep_r  += reward
            if done:
                success = info.get('success', False)
                break

        ep_rewards.append(ep_r)
        ep_successes.append(float(success))

        if (ep + 1) % 10 == 0:
            mean_r = float(np.mean(ep_rewards[-10:]))
            succ_r = float(np.mean(ep_successes[-10:]))
            print(f"  Ep {ep+1:4d}/{ep_max}  "
                  f"奖励 {mean_r:8.1f}  "
                  f"成功率 {succ_r*100:5.1f}%  "
                  f"噪声 {noise_scale:.3f}  "
                  f"缓冲 {len(agent.buffer)}")

        if (ep + 1) % SAVE_EVERY == 0:
            ckpt = os.path.join(MODEL_DIR, f'ddpg_ep{ep+1}.pth')
            agent.save(ckpt)
            print(f"    检查点已保存: {ckpt}")

    agent.save(ACTOR_PATH)
    print(f"\n  最终模型已保存: {ACTOR_PATH}")
    _plot_curve(ep_rewards, ep_successes, ep_max)
    print("  训练完成！")
    return agent


# ══════════════════════════════════════════════════════════════════════════════
# § 训练曲线绘制
# ══════════════════════════════════════════════════════════════════════════════

def _plot_curve(rewards: List[float], successes: List[float], ep_max: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    w = min(20, len(rewards))

    axes[0].plot(rewards, alpha=0.4, color='#e75840', label='原始')
    smooth = np.convolve(rewards, np.ones(w) / w, mode='valid')
    axes[0].plot(range(w - 1, len(rewards)), smooth,
                 color='#222222', lw=2, label=f'平滑(w={w})')
    axes[0].set_title('DDPG Episode Reward')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].legend()

    axes[1].plot(successes, alpha=0.3, color='#3cb44b', label='逐轮')
    smooth_s = np.convolve(successes, np.ones(w) / w, mode='valid')
    axes[1].plot(range(w - 1, len(successes)), smooth_s,
                 color='#f58231', lw=2, label=f'平滑(w={w})')
    axes[1].set_title(f'DDPG Success Rate (窗口={w})')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Rate')
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.suptitle('DDPG Training Curve', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, 'ddpg_training_curve.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  训练曲线: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# § 命令行入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDPG 路径规划训练')
    parser.add_argument('--episodes', type=int, default=EP_MAX,
                        help=f'训练轮数（默认 {EP_MAX}）')
    parser.add_argument('--resume', action='store_true',
                        help='从已有检查点继续训练')
    args = parser.parse_args()
    train(ep_max=args.episodes, resume=args.resume)
