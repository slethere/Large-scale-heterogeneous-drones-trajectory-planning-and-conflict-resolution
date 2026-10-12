"""
qdac_train.py  —  QR-DAC (Quantile-Regression Distributional Actor-Critic) 训练脚本
====================================================================================
独立运行即可开始训练：
    python qdac_train.py [--episodes 2000] [--resume]

训练完成后输出：
    qdac_actor.pth               (推理时主函数调用)
    qdac_models/qdac_epXXX.pth  (检查点)
    qdac_training_curve.png      (训练曲线)

算法：DAC = Distributional Actor-Critic（参考 main_DAC.py）
  与 SAC 的对比：
  ┌─────────────┬────────────────────────────┬──────────────────────────────────┐
  │ 组件         │ SAC                        │ DAC (本文件)                     │
  ├─────────────┼────────────────────────────┼──────────────────────────────────┤
  │ Actor       │ 高斯随机策略 + tanh          │ 完全相同                         │
  │ Critic 输出  │ 标量 Q(s,a) → (batch,1)    │ 分位数分布 Z(s,a) → (batch, N)   │
  │ Critic 损失  │ MSE                        │ 分位数 Huber 损失（QR-DQN）      │
  │ Actor 损失   │ α*log_π - min(Q1,Q2)       │ α*log_π - min(E[Z1],E[Z2])       │
  │ 目标 Critic  │ min(Q1_target,Q2_target)   │ 期望值较小的分位数分布            │
  │ 熵系数       │ 自适应 α                    │ 自适应 α（相同）                 │
  └─────────────┴────────────────────────────┴──────────────────────────────────┘

  · N_QUANTILES=51 个分位数对回报分布建模，比标量 Q 更抗 reward 偏斜/异方差。
  · 双头分位数网络（Q1/Q2）保持 Clipped Double Q 防止高估。
  · Actor 仍用 reparameterization trick 保证梯度可回传。

复用环境：sac_path_env.SACPathEnv（状态/动作空间完全相同）
QDACAgent 类同时供 alns_obstacle_wind.py 导入用于推理。
"""

import os
import sys
import argparse
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
MODEL_DIR     = os.path.join(BASE_DIR, 'qdac_models')
ACTOR_PATH    = os.path.join(BASE_DIR, 'qdac_actor.pth')

EP_MAX       = 20000
GAMMA        = 0.95
TAU          = 1e-2
BATCH_SIZE   = 128
MEM_CAPACITY = 50000   # 扩容，覆盖更多训练多样性
POLICY_LR    = 1e-3
Q_LR         = 3e-4
ALPHA_LR     = 3e-4
HIDDEN       = 256
SAVE_EVERY   = 500     # 减少检查点 I/O 开销

# 训练时随机注入他机路径的概率（消除训练/推理分布偏移）
OTHER_PATH_INJECT_PROB = 0.7   # 提升至 70%，与推理端频率对齐
OTHER_PATH_INJECT_MAX  = 5     # 单次最多注入条数（推理端固定采样 4 条）

# 分位数超参数（参考 main_DAC.py）
N_QUANTILES = 51    # 分位数个数：越多分布越精细，计算量越大
KAPPA       = 1.0   # Huber 损失平滑阈值

# 熵目标（参考 main_DAC.py: -0.1；比 SAC 的 -ACTION_DIM 更宽松）
TARGET_ENT  = -0.1
LOG_ALPHA_MIN = -5.0
LOG_ALPHA_MAX =  1.0

# 课程学习（三阶段，与 EP_MAX=20000 匹配）
CURRICULUM_PHASES = [
    (0,     5000,  2000),    # 阶段1：短距离任务（≤2km），建立基础策略
    (5000,  12000, 8000),    # 阶段2：中距离任务（≤8km），扩展泛化能力
    (12000, 20000, None),    # 阶段3：全部任务，完整策略
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# § 网络定义
# ══════════════════════════════════════════════════════════════════════════════

class QDACActorNet(nn.Module):
    """
    随机策略网络（与 main_DAC.py ActorNet 结构一致）。
    输出高斯分布的 mean 和 std，动作通过 reparameterization + tanh 生成。
    """
    def __init__(self):
        super().__init__()
        self.fc1         = nn.Linear(STATE_DIM, HIDDEN)
        self.fc2         = nn.Linear(HIDDEN, HIDDEN)
        self.mean_out    = nn.Linear(HIDDEN, ACTION_DIM)
        self.log_std_out = nn.Linear(HIDDEN, ACTION_DIM)
        for layer in (self.fc1, self.fc2, self.mean_out, self.log_std_out):
            nn.init.normal_(layer.weight, 0, 0.1)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (raw_mu, std)，mu 未经 tanh 压缩（标准 SAC Actor 实现）。"""
        x       = F.relu(self.fc1(x))
        x       = F.relu(self.fc2(x))
        mean    = self.mean_out(x)                      # raw，不加 tanh
        log_std = torch.clamp(self.log_std_out(x), -20, 2)
        return mean, log_std.exp()

    def sample(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """重参数化采样，返回 (action ∈ [-1,1], log_prob)。"""
        mean, std = self(s)
        dist      = torch.distributions.Normal(mean, std)
        z         = dist.rsample()
        a         = torch.tanh(z)                      # 只在这里 tanh 一次
        log_p     = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        return a, log_p.sum(-1, keepdim=True)


class QuantileCriticNet(nn.Module):
    """
    双头分位数 Critic（参考 main_DAC.py QuantileCriticNet）。

    与 SAC CriticNet 的区别：
      SAC:  输出 (batch, 1)         ← 标量期望 Q
      DAC:  输出 (batch, N_QUANTILES) ← Q 值的分位数分布（双头）

    输入 cat([s, a])，每头独立预测 N_QUANTILES 个分位数。
    """
    def __init__(self):
        super().__init__()
        in_dim = STATE_DIM + ACTION_DIM
        # Q1 分位数流
        self.q1_fc1 = nn.Linear(in_dim, HIDDEN)
        self.q1_fc2 = nn.Linear(HIDDEN,  HIDDEN)
        self.q1_out = nn.Linear(HIDDEN,  N_QUANTILES)
        # Q2 分位数流
        self.q2_fc1 = nn.Linear(in_dim, HIDDEN)
        self.q2_fc2 = nn.Linear(HIDDEN,  HIDDEN)
        self.q2_out = nn.Linear(HIDDEN,  N_QUANTILES)
        for layer in (self.q1_fc1, self.q1_fc2, self.q1_out,
                      self.q2_fc1, self.q2_fc2, self.q2_out):
            nn.init.normal_(layer.weight, 0, 0.1)
            nn.init.zeros_(layer.bias)

    def forward(self, s: torch.Tensor, a: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (q1, q2)，各 shape = (batch, N_QUANTILES)。"""
        x  = torch.cat([s, a], dim=-1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2


# ══════════════════════════════════════════════════════════════════════════════
# § 分位数 Huber 损失（QR-DQN，参考 main_DAC.py）
# ══════════════════════════════════════════════════════════════════════════════

def quantile_huber_loss(current_q: torch.Tensor,
                        target_q:  torch.Tensor) -> torch.Tensor:
    """
    分位数 Huber 损失（取代 SAC 中的 MSELoss）。

    Args:
        current_q : (batch, N) — 当前网络预测的 N 个分位数
        target_q  : (batch, N) — 贝尔曼目标分位数

    Returns:
        标量损失（比 MSE 更能处理偏态/重尾回报分布）
    """
    N   = current_q.shape[1]
    # 分位数中点 τ_i = (2i-1)/(2N)
    tau = torch.FloatTensor(
        [(2 * i - 1) / (2.0 * N) for i in range(1, N + 1)]
    ).to(current_q.device).view(1, N, 1)  # (1, N, 1)

    z_ij  = current_q.unsqueeze(2)   # (batch, N, 1)
    y_ij  = target_q.unsqueeze(1)    # (batch, 1, N)
    delta = y_ij - z_ij              # (batch, N, N)

    huber = torch.where(
        delta.abs() <= KAPPA,
        0.5 * delta.pow(2),
        KAPPA * (delta.abs() - 0.5 * KAPPA),
    )
    weight = (tau - (delta.detach() < 0).float()).abs()
    return (weight * huber / KAPPA).mean(dim=2).sum(dim=1).mean()


# ══════════════════════════════════════════════════════════════════════════════
# § 经验回放缓冲
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.cap  = capacity
        self.buf: List = []
        self.ptr  = 0

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
# § QDAC Agent（供训练和推理共用）
# ══════════════════════════════════════════════════════════════════════════════

class QDACAgent:
    """
    单智能体 Distributional Actor-Critic（QR-DAC）。
    训练：调用 update()。
    推理：调用 select_action(state, deterministic=True)。
    """

    def __init__(self):
        self.actor         = QDACActorNet().to(device)
        self.critic        = QuantileCriticNet().to(device)
        self.critic_target = QuantileCriticNet().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=POLICY_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=Q_LR)

        # 可学习熵温度 α（参考 main_DAC.py target_entropy=-0.1）
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha     = self.log_alpha.exp().item()
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=ALPHA_LR)

        self.buffer = ReplayBuffer(MEM_CAPACITY)

    # ── 动作选取 ──────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        state = np.nan_to_num(
            np.asarray(state, dtype=np.float32),
            nan=0.0, posinf=1.0, neginf=-1.0,
        )
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(s)
                return torch.tanh(mean).cpu().numpy()[0]  # raw mu → tanh → [-1,1]
            else:
                a, _ = self.actor.sample(s)
                return a.cpu().numpy()[0]

    # ── 单步更新 ──────────────────────────────────────────────────────────────

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        s, a, r, s_, done = self.buffer.sample(BATCH_SIZE)
        alpha = self.log_alpha.exp()

        # ── Critic 更新（分位数贝尔曼目标）──────────────────────────────────────
        with torch.no_grad():
            a_, log_p_ = self.actor.sample(s_)
            tq1, tq2   = self.critic_target(s_, a_)

            # 双头保守估计：选期望值较小的分位数分布（参考 main_DAC.py）
            tq1_mean = tq1.mean(dim=1, keepdim=True)    # (B, 1)
            tq2_mean = tq2.mean(dim=1, keepdim=True)
            mask         = (tq1_mean <= tq2_mean).float()
            target_dist  = mask * tq1 + (1 - mask) * tq2    # (B, N_QUANTILES)

            # 熵修正后的贝尔曼目标（广播：r (B,1) + gamma*(target_dist (B,N) - ent (B,1))）
            ent_penalty = alpha * log_p_                      # (B, 1)
            target_q    = r + GAMMA * (1 - done) * (target_dist - ent_penalty)  # (B, N)

        cur_q1, cur_q2 = self.critic(s, a)
        c_loss = (quantile_huber_loss(cur_q1, target_q.detach()) +
                  quantile_huber_loss(cur_q2, target_q.detach()))

        self.critic_opt.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.critic_opt.step()

        # ── Actor 更新（最大化分布期望 Q - 熵正则）──────────────────────────────
        a_new, log_p = self.actor.sample(s)
        cq1, cq2     = self.critic(s, a_new)
        # 取双头期望 Q 最小值指导策略（与 main_DAC.py 一致）
        q_mean = torch.min(
            cq1.mean(dim=1, keepdim=True),
            cq2.mean(dim=1, keepdim=True),
        )   # (B, 1)
        a_loss = (alpha.detach() * log_p - q_mean).mean()

        self.actor_opt.zero_grad()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.actor_opt.step()

        # ── 熵系数 α 更新 ──────────────────────────────────────────────────────
        ent_loss = -(self.log_alpha * (log_p + TARGET_ENT).detach()).mean()
        self.alpha_opt.zero_grad()
        ent_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(LOG_ALPHA_MIN, LOG_ALPHA_MAX)
        self.alpha = self.log_alpha.exp().item()

        # ── 软更新目标 Critic ──────────────────────────────────────────────────
        for tp, p in zip(self.critic_target.parameters(),
                         self.critic.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

    # ── 保存 / 加载 ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            'actor':         self.actor.state_dict(),
            'critic':        self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha':     self.log_alpha.detach().cpu(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(
            ckpt.get('critic_target', ckpt['critic']))
        if 'log_alpha' in ckpt:
            with torch.no_grad():
                self.log_alpha.copy_(ckpt['log_alpha'].to(device))
            self.alpha = self.log_alpha.exp().item()


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
        dt.dronetime = t.dronetime
        dt.order_cnt = t.order_cnt
        delivery_tasks.append(dt)
    lat_ref = float(np.mean([t.sender_lat for t in delivery_tasks]))
    return delivery_tasks, buildings, lat_ref


# ══════════════════════════════════════════════════════════════════════════════
# § 训练主函数
# ══════════════════════════════════════════════════════════════════════════════

def train(ep_max: int = EP_MAX, resume: bool = False):
    print("=" * 60)
    print("  QR-DAC (Distributional Actor-Critic) 无人机路径规划训练")
    print(f"  设备: {device}  |  训练轮数: {ep_max}")
    print(f"  分位数数量: {N_QUANTILES}  |  Huber kappa: {KAPPA}")
    print(f"  目标熵: {TARGET_ENT}  |  log_alpha 范围: [{LOG_ALPHA_MIN}, {LOG_ALPHA_MAX}]")
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

    # ── 他机路径注入池（消除训练/推理分布偏移）──────────────────────────────────
    # 预先用规则规划器生成所有任务的参考路径，训练时随机抽取 1~3 条作为 other_paths。
    # 目的：让模型在训练时也看到"有他机存在"的状态，与推理时一致。
    print("  预计算参考路径池（用于训练时 other_paths 注入）...")
    from drone_obstacle_avoidance import DronePathPlanner as _Planner
    _planner = _Planner(pd.read_csv(BUILDINGS_CSV))
    _ref_paths: List = []
    for _t in delivery_tasks:
        try:
            _res = _planner.plan_path(_t, verbose=False)
            _ref_paths.append(_res['path'])
        except Exception:
            pass
    print(f"  参考路径池大小: {len(_ref_paths)} 条")

    # 时变风 + 双机型 Drone
    wind_ts = load_wind_timeseries(WIND_CSV)
    if len(wind_ts) > 0:
        _mw, _md = wind_ts.mean()
        print(f"  风场: 时变 {len(wind_ts)} 条样本，日均 {_mw:.2f} m/s / {_md:.1f}°")
    else:
        print("  风场: wind.csv 未加载，风/P4 disabled")
    mr_drone = Drone(1, callsign='QDAC-MR', craft_type='multirotor')
    fw_drone = Drone(2, callsign='QDAC-FW', craft_type='fixed_wing')
    FIXED_WING_RATIO = 0.3
    print(f"  双机型训练: MR + FW, ratio FW={FIXED_WING_RATIO:.0%}")

    env   = SACPathEnv(
        buildings   = buildings,
        lat_ref     = lat_ref,
        drone       = mr_drone,
        wind_lookup = wind_ts if len(wind_ts) > 0 else None,
    )
    agent = QDACAgent()

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
        task = random.choice(_get_curriculum_tasks(ep))

        # 按 FIXED_WING_RATIO 随机切换机型
        env.set_drone(fw_drone if random.random() < FIXED_WING_RATIO else mr_drone)

        # 注入空间最近的他机路径（比随机采样更贴近真实推理场景）
        if _ref_paths and random.random() < OTHER_PATH_INJECT_PROB:
            n_inject = random.randint(1, min(OTHER_PATH_INJECT_MAX, len(_ref_paths)))
            def _d(p):
                if not p: return float('inf')
                wp = p[0]
                dx = (wp['lon'] - task.sender_lon) * 111000.0 * np.cos(np.radians(task.sender_lat))
                dy = (wp['lat'] - task.sender_lat) * 111000.0
                return (dx*dx + dy*dy) ** 0.5
            injected = sorted(_ref_paths, key=_d)[:n_inject]
        else:
            injected = []
        env.set_other_paths(injected)
        env.set_task(task)
        state   = env.reset()
        ep_r    = 0.0
        success = False

        for _ in range(MAX_STEPS):
            action = agent.select_action(state, deterministic=False)
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

        if (ep + 1) % 50 == 0:
            mean_r = float(np.mean(ep_rewards[-50:]))
            succ_r = float(np.mean(ep_successes[-50:]))
            print(f"  Ep {ep+1:5d}/{ep_max}  "
                  f"奖励 {mean_r:8.1f}  "
                  f"成功率 {succ_r*100:5.1f}%  "
                  f"α={agent.alpha:.4f}  "
                  f"缓冲 {len(agent.buffer)}")

        if (ep + 1) % SAVE_EVERY == 0:
            ckpt = os.path.join(MODEL_DIR, f'qdac_ep{ep+1}.pth')
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

    axes[0].plot(rewards, alpha=0.4, color='#3a86ff', label='原始')
    smooth = np.convolve(rewards, np.ones(w) / w, mode='valid')
    axes[0].plot(range(w - 1, len(rewards)), smooth,
                 color='#222222', lw=2, label=f'平滑(w={w})')
    axes[0].set_title('QR-DAC Episode Reward')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].legend()

    axes[1].plot(successes, alpha=0.3, color='#3cb44b', label='逐轮')
    smooth_s = np.convolve(successes, np.ones(w) / w, mode='valid')
    axes[1].plot(range(w - 1, len(successes)), smooth_s,
                 color='#f58231', lw=2, label=f'平滑(w={w})')
    axes[1].set_title(f'QR-DAC Success Rate (窗口={w})')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Rate')
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.suptitle('QR-DAC (Distributional Actor-Critic) Training', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, 'qdac_training_curve.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  训练曲线: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# § 命令行入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QR-DAC 路径规划训练')
    parser.add_argument('--episodes', type=int, default=EP_MAX,
                        help=f'训练轮数（默认 {EP_MAX}）')
    parser.add_argument('--resume', action='store_true',
                        help='从已有检查点继续训练')
    args = parser.parse_args()
    train(ep_max=args.episodes, resume=args.resume)
