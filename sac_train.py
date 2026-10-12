"""
sac_train.py  —  SAC 无人机路径规划训练脚本
===========================================
独立运行即可开始训练：
    python sac_train.py [--episodes 500] [--resume]

训练完成后输出：
    sac_actor.pth          (推理时主函数调用)
    sac_models/sac_epXXX.pth  (检查点)
    sac_training_curve.png (训练曲线)

SACAgent 类同时供 alns_obstacle_wind.py 导入用于推理。
"""

import os
import sys
import argparse
import random
from typing import List, Tuple, Optional

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
    Building, DeliveryTask, DronePathPlanner, Drone,
    load_wind_timeseries,
    FLEET_FIXED_WING_RATIO,
)

# ══════════════════════════════════════════════════════════════════════════════
# § 超参数（可通过命令行覆盖）
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DELIVERY_CSV = os.path.join(BASE_DIR, 'nanjing_delivery_700_2026_01.csv')
BUILDINGS_CSV= os.path.join(BASE_DIR, 'nanjing_buildings.csv')
WIND_CSV     = os.path.join(BASE_DIR, 'wind.csv')
MODEL_DIR    = os.path.join(BASE_DIR, 'sac_models')
ACTOR_PATH   = os.path.join(BASE_DIR, 'sac_actor.pth')

EP_MAX       = 20000
GAMMA        = 0.95
TAU          = 1e-2
BATCH_SIZE   = 128
MEM_CAPACITY = 50000   # 扩容，覆盖更多训练多样性
POLICY_LR    = 1e-3
Q_LR         = 3e-4
ALPHA_LR     = 3e-4
TARGET_ENT   = -1.0     # 降低目标熵（=-ACTION_DIM），减少无效探索，加速收敛
HIDDEN       = 256
SAVE_EVERY   = 500     # 减少检查点 I/O 开销
# 课程学习分阶段（与 EP_MAX=20000 匹配）
CURRICULUM_PHASES = [
    (0,     5000,  2000),    # 阶段1：短距离任务（≤2km），建立基础策略
    (5000,  12000, 8000),    # 阶段2：中距离任务（≤8km），扩展泛化能力
    (12000, 20000, None),    # 阶段3：全部任务，完整策略
]
# 训练时随机注入他机路径的概率（消除训练/推理分布偏移）
# 推理时固定注入 4 条路径，训练需与之对齐：
#   · 注入概率提升至 0.7（覆盖更多有他机的场景）
#   · 注入数量扩展到 1-5 条（覆盖 0-4 的典型推理范围）
OTHER_PATH_INJECT_PROB = 0.7
OTHER_PATH_INJECT_MAX  = 5    # 单次最多注入条数（与推理端 _SAC_OTHER_PATHS_MAX=4 对齐）

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# § 网络定义
# ══════════════════════════════════════════════════════════════════════════════

class ActorNet(nn.Module):
    """
    随机策略网络，输出动作均值（raw，未 tanh）与对数标准差。

    标准 SAC 实现：forward() 输出 raw mu，tanh 只在 sample() 和
    确定性推理时各调用一次，避免双重 tanh 导致梯度信号失真、
    Actor 权重饱和到极限值的训练崩溃问题。
    """
    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(STATE_DIM, HIDDEN)
        self.fc2     = nn.Linear(HIDDEN, HIDDEN)
        self.mean    = nn.Linear(HIDDEN, ACTION_DIM)
        self.log_std = nn.Linear(HIDDEN, ACTION_DIM)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (raw_mu, std)，mu 未经 tanh 压缩。"""
        x   = F.relu(self.fc1(x))
        x   = F.relu(self.fc2(x))
        mu  = self.mean(x)                              # raw，不加 tanh
        lsg = torch.clamp(self.log_std(x), -20, 2)
        return mu, lsg.exp()

    def sample(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """重参数化采样，返回 (action ∈ [-1,1], log_prob)。"""
        mu, std = self(s)
        dist    = torch.distributions.Normal(mu, std)
        z       = dist.rsample()
        a       = torch.tanh(z)                        # 只在这里 tanh 一次
        log_p   = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        return a, log_p.sum(-1, keepdim=True)


class CriticNet(nn.Module):
    """双 Q 网络（Clipped Double Q-learning）。"""
    def __init__(self):
        super().__init__()
        inp = STATE_DIM + ACTION_DIM
        def _mlp():
            return nn.Sequential(
                nn.Linear(inp, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, 1),
            )
        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, s: torch.Tensor, a: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


# ══════════════════════════════════════════════════════════════════════════════
# § 经验回放缓冲
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.cap  = capacity
        self.buf: List = []
        self.ptr  = 0

    def push(self, s, a, r, s_, done):
        if len(self.buf) < self.cap:
            self.buf.append(None)
        self.buf[self.ptr] = (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.float32),
            float(r),
            np.array(s_, dtype=np.float32),
            float(done),
        )
        self.ptr = (self.ptr + 1) % self.cap

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
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
# § SAC Agent（供训练和推理共用）
# ══════════════════════════════════════════════════════════════════════════════

class SACAgent:
    """
    单智能体 SAC。
    训练：调用 update()。
    推理：调用 select_action(state, deterministic=True)。
    """

    def __init__(self):
        self.actor         = ActorNet().to(device)
        self.critic        = CriticNet().to(device)
        self.critic_target = CriticNet().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=POLICY_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=Q_LR)

        self.log_alpha  = torch.zeros(1, requires_grad=True, device=device)
        self.alpha      = self.log_alpha.exp()
        self.alpha_opt  = torch.optim.Adam([self.log_alpha], lr=ALPHA_LR)

        self.buffer     = ReplayBuffer(MEM_CAPACITY)

    # ── 动作选取 ──────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(s)
                return torch.tanh(mu).cpu().numpy()[0]  # raw mu → tanh → [-1,1]
            else:
                a, _ = self.actor.sample(s)
                return a.cpu().numpy()[0]

    # ── 单步更新 ──────────────────────────────────────────────────────────────

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        s, a, r, s_, done = self.buffer.sample(BATCH_SIZE)

        # ── Critic 更新 ────────────────────────────────────────────────────
        with torch.no_grad():
            a_, log_p_ = self.actor.sample(s_)
            q1_, q2_   = self.critic_target(s_, a_)
            q_target   = r + GAMMA * (1 - done) * (
                torch.min(q1_, q2_) - self.alpha * log_p_
            )
        q1, q2 = self.critic(s, a)
        c_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        c_loss.backward()
        self.critic_opt.step()

        # ── Actor 更新 ─────────────────────────────────────────────────────
        a_new, log_p = self.actor.sample(s)
        q1_n, q2_n   = self.critic(s, a_new)
        a_loss = (self.alpha * log_p - torch.min(q1_n, q2_n)).mean()
        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()

        # ── 熵温度 α 更新 ──────────────────────────────────────────────────
        ent_loss = -(self.log_alpha * (log_p + TARGET_ENT).detach()).mean()
        self.alpha_opt.zero_grad()
        ent_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp()

        # ── 软更新 Target Critic ───────────────────────────────────────────
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
        self.critic_target.load_state_dict(ckpt.get('critic_target',
                                                      ckpt['critic']))
        if 'log_alpha' in ckpt:
            with torch.no_grad():
                self.log_alpha.copy_(ckpt['log_alpha'].to(device))
            self.alpha = self.log_alpha.exp()


# ══════════════════════════════════════════════════════════════════════════════
# § 数据加载
# ══════════════════════════════════════════════════════════════════════════════

def _load_training_data():
    """
    从 CSV 文件加载订单和建筑物。
    返回 (DeliveryTask 列表, Building 列表, lat_ref)。

    每个 DeliveryTask 上动态附加 .dronetime（datetime，来自 alns_task）以便
    SACPathEnv 在 reset() 时按该时刻查时变风。
    """
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
        dt.dronetime = t.dronetime    # env.reset 据此从 wind_lookup 查任务时刻的风
        dt.order_cnt = t.order_cnt    # env.reset 据此设 drone.current_payload_kg (kg)
        delivery_tasks.append(dt)
    lat_ref = float(np.mean([t.sender_lat for t in delivery_tasks]))
    return delivery_tasks, buildings, lat_ref


# ══════════════════════════════════════════════════════════════════════════════
# § 训练主函数
# ══════════════════════════════════════════════════════════════════════════════

def train(ep_max: int = EP_MAX, resume: bool = False):
    print("=" * 60)
    print("  SAC 无人机路径规划训练")
    print(f"  设备: {device}  |  训练轮数: {ep_max}")
    print("=" * 60)

    delivery_tasks, buildings, lat_ref = _load_training_data()
    print(f"  订单数: {len(delivery_tasks)}  |  建筑数: {len(buildings)}")

    # 时变风（按订单 dronetime 查），以及 SAC 环境共享的多旋翼 Drone（P4 电量用）
    wind_ts = load_wind_timeseries(WIND_CSV)
    if len(wind_ts) > 0:
        _mean_ms, _mean_deg = wind_ts.mean()
        print(f"  风场: 时变 {len(wind_ts)} 条样本，日均 {_mean_ms:.2f} m/s / {_mean_deg:.1f}°")
    else:
        print("  风场: wind.csv 未加载到数据，P4 / 风 disabled")
    # 同时训练多旋翼 + 固定翼两种机型：每 episode 按 FIXED_WING_RATIO 采样
    # payload 会在每个 episode reset() 时按 task.order_cnt（1~5 kg）动态覆盖
    mr_drone = Drone(1, callsign='SAC-MR',  craft_type='multirotor')
    # craft_type 与 build_fleet_micro_drones 部署的机队、masac_train 保持一致
    fw_drone = Drone(2, callsign='SAC-FW',  craft_type='pure_fixed_wing')
    FIXED_WING_RATIO = FLEET_FIXED_WING_RATIO   # 5 固翼 + 25 多旋翼
    print(f"  训练 drones: MR(cap={mr_drone.battery_capacity_wh:.0f}Wh cruise={mr_drone.cruise_power_w:.0f}W) "
          f"+ FW(cap={fw_drone.battery_capacity_wh:.0f}Wh cruise={fw_drone.cruise_power_w:.0f}W) "
          f"ratio FW={FIXED_WING_RATIO:.0%} (机队 5FW+25MR)")

    # 按任务距离排序，供课程学习使用
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

    # 预计算参考路径池（消除训练/推理分布偏移）
    print("  预计算参考路径池（用于训练时 other_paths 注入）...")
    _planner = DronePathPlanner(pd.read_csv(BUILDINGS_CSV))
    _ref_paths: List = []
    for _t in delivery_tasks:
        try:
            _ref_paths.append(_planner.plan_path(_t, verbose=False)['path'])
        except Exception:
            pass
    print(f"  参考路径池大小: {len(_ref_paths)} 条")

    env   = SACPathEnv(
        buildings   = buildings,
        lat_ref     = lat_ref,
        drone       = mr_drone,
        wind_lookup = wind_ts if len(wind_ts) > 0 else None,
    )
    agent = SACAgent()

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

        # 按 FIXED_WING_RATIO 随机切换机型，使单一策略覆盖两种能耗模型
        env.set_drone(fw_drone if random.random() < FIXED_WING_RATIO else mr_drone)

        # 注入空间最近的他机路径（比随机采样更贴近真实推理场景）
        if _ref_paths and random.random() < OTHER_PATH_INJECT_PROB:
            n = random.randint(1, min(OTHER_PATH_INJECT_MAX, len(_ref_paths)))
            # 按与当前任务起点的距离排序，取最近 n 条
            def _d(p):
                if not p: return float('inf')
                wp = p[0]
                dx = (wp['lon'] - task.sender_lon) * 111000.0 * np.cos(np.radians(task.sender_lat))
                dy = (wp['lat'] - task.sender_lat) * 111000.0
                return (dx*dx + dy*dy) ** 0.5
            injected = sorted(_ref_paths, key=_d)[:n]
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
            alpha  = float(agent.alpha)
            print(f"  Ep {ep+1:5d}/{ep_max}  "
                  f"奖励 {mean_r:8.1f}  "
                  f"成功率 {succ_r*100:5.1f}%  "
                  f"α={alpha:.4f}  "
                  f"缓冲 {len(agent.buffer)}")

        if (ep + 1) % SAVE_EVERY == 0:
            ckpt = os.path.join(MODEL_DIR, f'sac_ep{ep+1}.pth')
            agent.save(ckpt)
            print(f"    检查点已保存: {ckpt}")

    # 保存最终模型
    agent.save(ACTOR_PATH)
    print(f"\n  最终模型已保存: {ACTOR_PATH}")

    # 绘制训练曲线
    _plot_curve(ep_rewards, ep_successes, ep_max)
    print("  训练完成！")
    return agent


def _plot_curve(rewards: List[float], successes: List[float], ep_max: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(rewards, alpha=0.6, color='#4363d8')
    w = min(20, len(rewards))
    smooth = np.convolve(rewards, np.ones(w)/w, mode='valid')
    axes[0].plot(range(w-1, len(rewards)), smooth, color='#e6194b', lw=2)
    axes[0].set_title('Episode Reward')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')

    axes[1].plot(successes, alpha=0.4, color='#3cb44b')
    smooth_s = np.convolve(successes, np.ones(w)/w, mode='valid')
    axes[1].plot(range(w-1, len(successes)), smooth_s, color='#f58231', lw=2)
    axes[1].set_title(f'Success Rate (窗口={w})')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Rate')
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, 'sac_training_curve.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  训练曲线: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# § 命令行入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAC 路径规划训练')
    parser.add_argument('--episodes', type=int, default=EP_MAX,
                        help=f'训练轮数（默认 {EP_MAX}）')
    parser.add_argument('--resume', action='store_true',
                        help='从已有检查点继续训练')
    args = parser.parse_args()
    train(ep_max=args.episodes, resume=args.resume)
