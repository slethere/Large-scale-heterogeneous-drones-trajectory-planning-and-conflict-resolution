# 训练时将本文件RENDER=False，Switch=0，以及player.py中的dt=1
# 测试时将本文件RENDER=True，Switch=1，以及player.py中的dt=0.1
#
# DAC: Distributional Actor-Critic（分布式Actor-Critic）
# 核心改进：Critic 不再输出标量 Q 值，而是输出 N_QUANTILES 个分位数，
# 描述回报分布 Z(s,a)；使用分位数 Huber 损失（QR-DQN）训练 Critic；
# Actor 最大化分布期望值（分位数均值）并保留 SAC 的熵正则化。
from rl_env.path_env import RlGame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle as pkl

shoplistfile      = 'F:/MASAC-UAVs-path-planning-master\MADAC'
shoplistfile_test = 'F:/MASAC-UAVs-path-planning-master\MADAC_test'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

N_Agent      = 3
M_Enemy      = 0
RENDER       = False
TRAIN_NUM    = 1
TEST_EPIOSDE = 100
env          = RlGame(n=N_Agent, m=M_Enemy, render=RENDER).unwrapped

state_number  = 7 + (N_Agent - 1)
action_number = env.action_space.shape[0]
max_action    = env.action_space.high[0]
min_action    = env.action_space.low[0]

EP_MAX         = 1000
EP_LEN         = 500
GAMMA          = 0.95
q_lr           = 3e-4
policy_lr      = 1e-3
BATCH          = 128
tau_update     = 1e-2
MemoryCapacity = 20000
Switch         = 1   # 0=训练  1=测试

# ---------- 分布强化学习超参数 ----------
N_QUANTILES = 51    # 分位数个数（越多分布越精细，计算量越大）
KAPPA       = 1.0   # Huber损失的平滑阈值


# ============================================================
#  网络结构
# ============================================================

class ActorNet(nn.Module):
    """
    随机策略网络（与 MASAC 相同），输出高斯分布的均值 mean 和标准差 std。
    DAC 的 Actor 结构与 SAC 完全一致，区别在于 Critic 的形式。
    """
    def __init__(self, inp, outp):
        super(ActorNet, self).__init__()
        self.fc1         = nn.Linear(inp, 256)
        self.fc2         = nn.Linear(256, 256)
        self.mean_out    = nn.Linear(256, outp)
        self.log_std_out = nn.Linear(256, outp)
        for layer in [self.fc1, self.fc2, self.mean_out, self.log_std_out]:
            layer.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x       = F.relu(self.fc1(x))
        x       = F.relu(self.fc2(x))
        mean    = max_action * torch.tanh(self.mean_out(x))
        log_std = torch.clamp(self.log_std_out(x), -20, 2)
        std     = log_std.exp()
        return mean, std


class QuantileCriticNet(nn.Module):
    """
    分位数Critic网络（双网络结构，防止过估计）。

    与 SAC 的 CriticNet 的区别：
      - SAC CriticNet 输出形状: (batch, 1)  ← 标量期望 Q 值
      - DAC QuantileCriticNet 输出形状: (batch, N_QUANTILES) ← Q 值的分布

    双头 Q1/Q2 结构与 SAC 保持一致，选取保守估计防止高估。
    """
    def __init__(self, state_dim, action_dim):
        super(QuantileCriticNet, self).__init__()
        in_dim = state_dim + action_dim
        # Q1 分位数流
        self.q1_fc1 = nn.Linear(in_dim, 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_out = nn.Linear(256, N_QUANTILES)
        # Q2 分位数流
        self.q2_fc1 = nn.Linear(in_dim, 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_out = nn.Linear(256, N_QUANTILES)
        for layer in [self.q1_fc1, self.q1_fc2, self.q1_out,
                      self.q2_fc1, self.q2_fc2, self.q2_out]:
            layer.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x  = torch.cat((s, a), dim=1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)   # (batch, N_QUANTILES)
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)   # (batch, N_QUANTILES)
        return q1, q2


# ============================================================
#  分位数 Huber 损失
# ============================================================

def quantile_huber_loss(current_q, target_q):
    """
    QR-DQN 分位数 Huber 损失。

    与 SAC 的 MSELoss 的区别：
      SAC:  L = MSE(Q_pred, Q_target)   ← 均方误差，只学期望值
      DAC:  L = QR-Huber(Z_pred, Z_target) ← 分位数加权Huber，学习整个分布

    Args:
        current_q: (batch, N) — 当前网络预测的 N 个分位数
        target_q:  (batch, N) — 目标分位数（贝尔曼目标）

    Returns:
        标量损失值
    """
    N   = current_q.shape[1]
    # 分位数中点 τ̂_i = (2i−1)/(2N), i=1,...,N
    tau = torch.FloatTensor([(2 * i - 1) / (2.0 * N) for i in range(1, N + 1)])
    tau = tau.view(1, N, 1)   # (1, N, 1)

    # 扩维以计算所有 (i,j) 对的 TD 误差
    # δ_{ij} = y_j − z_i  (目标分位数 − 当前分位数)
    z_ij   = current_q.unsqueeze(2)  # (batch, N, 1)
    y_ij   = target_q.unsqueeze(1)   # (batch, 1, N)
    delta  = y_ij - z_ij             # (batch, N, N)

    # Huber 损失（平滑 L1）
    huber = torch.where(
        delta.abs() <= KAPPA,
        0.5 * delta.pow(2),
        KAPPA * (delta.abs() - 0.5 * KAPPA)
    )   # (batch, N, N)

    # 分位数非对称权重 |τ̂_i − 1(δ_{ij} < 0)|
    weight = (tau - (delta.detach() < 0).float()).abs()   # (batch, N, N)

    # 归一化并聚合
    loss = (weight * huber / KAPPA).mean(dim=2).sum(dim=1).mean()
    return loss


# ============================================================
#  经验回放
# ============================================================

class Memory():
    def __init__(self, capacity, dims):
        self.capacity       = capacity
        self.mem            = np.zeros((capacity, dims))
        self.memory_counter = 0

    def store_transition(self, s, a, r, s_):
        tran  = np.hstack((s, a, r, s_))
        index = self.memory_counter % self.capacity
        self.mem[index, :] = tran
        self.memory_counter += 1

    def sample(self, n):
        assert self.memory_counter >= self.capacity, '记忆库没有存满记忆'
        sample_index = np.random.choice(self.capacity, n)
        return self.mem[sample_index, :]


# ============================================================
#  Actor 类
# ============================================================

class Actor():
    """
    与 MASAC 的 Actor 结构完全相同，包含：
    - choose_action: 随机采样动作（与环境交互）
    - evaluate:      重参数化采样 + 计算 log 概率（用于梯度更新）
    - learn:         根据 Actor 损失更新网络参数
    """
    def __init__(self):
        self.action_net = ActorNet(state_number, action_number)
        self.optimizer  = torch.optim.Adam(self.action_net.parameters(), lr=policy_lr)

    def choose_action(self, s):
        with torch.no_grad():
            mean, std = self.action_net(torch.FloatTensor(s))
            dist      = torch.distributions.Normal(mean, std)
            action    = torch.clamp(dist.sample(), min_action, max_action)
        return action.numpy()

    def evaluate(self, s):
        mean, std    = self.action_net(s)
        dist         = torch.distributions.Normal(mean, std)
        z            = torch.distributions.Normal(0, 1).sample(mean.shape)
        action       = torch.tanh(mean + std * z)
        action       = torch.clamp(action, min_action, max_action)
        log_prob     = dist.log_prob(mean + std * z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob

    def learn(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ============================================================
#  可学习熵系数
# ============================================================

class Entroy():
    """与 MASAC 完全相同的可学习熵系数模块"""
    def __init__(self):
        self.target_entropy = -0.1
        self.log_alpha      = torch.zeros(1, requires_grad=True)
        self.alpha          = self.log_alpha.exp().detach()
        self.optimizer      = torch.optim.Adam([self.log_alpha], lr=q_lr)

    def learn(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ============================================================
#  Critic 类（核心区别：分布式 Q）
# ============================================================

class Critic():
    """
    DAC 的 Critic 与 MASAC Critic 的关键区别：
      MASAC Critic: CriticNet → (batch,1) 标量 Q，MSELoss
      DAC   Critic: QuantileCriticNet → (batch, N_QUANTILES) 分位数分布，分位数Huber损失

    其余结构（双网络、软更新、目标网络）保持与 MASAC 一致。
    """
    def __init__(self):
        state_dim  = state_number * (N_Agent + M_Enemy)
        self.net        = QuantileCriticNet(state_dim, action_number)
        self.target_net = QuantileCriticNet(state_dim, action_number)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer  = torch.optim.Adam(self.net.parameters(), lr=q_lr, eps=1e-5)

    def soft_update(self):
        for tp, p in zip(self.target_net.parameters(), self.net.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau_update) + p.data * tau_update)

    def get_quantiles(self, s, a):
        """返回当前网络的双头分位数分布 (q1, q2)，各 shape=(batch, N_QUANTILES)"""
        return self.net(s, a)

    def get_target_quantiles(self, s, a):
        """返回目标网络的双头分位数（不计算梯度）"""
        with torch.no_grad():
            return self.target_net(s, a)

    def learn(self, q1_pred, q2_pred, target_q):
        """
        分位数 Huber 损失（取代 MSELoss）。
        q1_pred, q2_pred: (batch, N_QUANTILES)
        target_q: (batch, N_QUANTILES)
        """
        loss = quantile_huber_loss(q1_pred, target_q.detach()) + \
               quantile_huber_loss(q2_pred, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ============================================================
#  主训练/测试逻辑
# ============================================================

def main():
    run(env)


def run(env):
    # --------------------------------------------------------
    #  训练模式
    # --------------------------------------------------------
    if Switch == 0:
        print('MADAC训练中...')
        all_ep_r  = [[] for _ in range(TRAIN_NUM)]
        all_ep_r0 = [[] for _ in range(TRAIN_NUM)]
        all_ep_r1 = [[] for _ in range(TRAIN_NUM)]
        all_ep_r2 = [[] for _ in range(TRAIN_NUM)]

        for k in range(TRAIN_NUM):
            actors  = [Actor()  for _ in range(N_Agent + M_Enemy)]
            critics = [Critic() for _ in range(N_Agent + M_Enemy)]
            entroys = [Entroy() for _ in range(N_Agent + M_Enemy)]
            M = Memory(
                MemoryCapacity,
                2 * state_number * (N_Agent + M_Enemy) +
                action_number * (N_Agent + M_Enemy) +
                1 * (N_Agent + M_Enemy)
            )
            action = np.zeros(((N_Agent + M_Enemy), action_number))

            for episode in range(EP_MAX):
                observation = env.reset()
                reward_totle = reward_totle0 = reward_totle1 = reward_totle2 = 0

                for timestep in range(EP_LEN):
                    for i in range(N_Agent + M_Enemy):
                        action[i] = actors[i].choose_action(observation[i])
                    action = np.clip(action, -max_action, max_action)

                    observation_, reward, done, win, team_counter = env.step(action)
                    M.store_transition(
                        observation.flatten(), action.flatten(),
                        reward.flatten(), observation_.flatten()
                    )

                    if M.memory_counter > MemoryCapacity:
                        b_M  = M.sample(BATCH)
                        n_sa = state_number * (N_Agent + M_Enemy)
                        n_a  = action_number * (N_Agent + M_Enemy)
                        n_r  = 1 * (N_Agent + M_Enemy)
                        b_s  = torch.FloatTensor(b_M[:, :n_sa])
                        b_a  = torch.FloatTensor(b_M[:, n_sa: n_sa + n_a])
                        b_r  = torch.FloatTensor(b_M[:, n_sa + n_a: n_sa + n_a + n_r])
                        b_s_ = torch.FloatTensor(b_M[:, n_sa + n_a + n_r:])

                        for i in range(N_Agent + M_Enemy):
                            if not win[i]:
                                s_i  = state_number * i
                                s_i1 = state_number * (i + 1)
                                a_i  = action_number * i
                                a_i1 = action_number * (i + 1)

                                # ===== Critic 更新 =====
                                new_action, log_prob_ = actors[i].evaluate(b_s_[:, s_i:s_i1])

                                # 目标网络分位数（双头取保守估计）
                                tq1, tq2 = critics[i].get_target_quantiles(b_s_, new_action)
                                tq1_mean = tq1.mean(dim=1, keepdim=True)  # (batch,1)
                                tq2_mean = tq2.mean(dim=1, keepdim=True)
                                # 选取期望值较小的分位数分布（悲观估计，防止高估）
                                mask         = (tq1_mean <= tq2_mean).float()      # (batch,1)
                                target_dist  = mask * tq1 + (1 - mask) * tq2       # (batch, N_QUANTILES)

                                # 熵修正后的贝尔曼目标分布
                                r_i         = b_r[:, i:(i + 1)]                   # (batch,1)
                                ent_penalty = entroys[i].alpha * log_prob_.mean(dim=1, keepdim=True)
                                target_q    = r_i + GAMMA * (target_dist - ent_penalty)  # (batch, N_QUANTILES)

                                cur_q1, cur_q2 = critics[i].get_quantiles(b_s, b_a[:, a_i:a_i1])
                                critics[i].learn(cur_q1, cur_q2, target_q)

                                # ===== Actor 更新 =====
                                a, log_prob = actors[i].evaluate(b_s[:, s_i:s_i1])
                                cq1, cq2   = critics[i].get_quantiles(b_s, a)
                                # 取双头期望 Q 的最小值指导策略（与 SAC 逻辑一致）
                                q_mean     = torch.min(
                                    cq1.mean(dim=1, keepdim=True),
                                    cq2.mean(dim=1, keepdim=True)
                                )  # (batch,1)
                                actor_loss = (entroys[i].alpha * log_prob - q_mean).mean()
                                actors[i].learn(actor_loss)

                                # ===== 熵系数更新 =====
                                alpha_loss = -(
                                    entroys[i].log_alpha.exp() *
                                    (log_prob + entroys[i].target_entropy).detach()
                                ).mean()
                                entroys[i].learn(alpha_loss)
                                entroys[i].alpha = entroys[i].log_alpha.exp().detach()

                                # ===== 软更新目标网络 =====
                                critics[i].soft_update()

                    observation   = observation_
                    reward_totle  += reward.mean()
                    reward_totle0 += float(reward[0])
                    reward_totle1 += float(reward[1])
                    reward_totle2 += float(reward[2])
                    if RENDER:
                        env.render()
                    if done:
                        break

                print("Ep: {} rewards: {}".format(episode, reward_totle))
                all_ep_r[k].append(reward_totle)
                all_ep_r0[k].append(reward_totle0)
                all_ep_r1[k].append(reward_totle1)
                all_ep_r2[k].append(reward_totle2)

                if episode % 20 == 0 and episode > 200:
                    torch.save(
                        {'net': actors[0].action_net.state_dict(), 'opt': actors[0].optimizer.state_dict()},
                        "F:/MASAC-UAVs-path-planning-master/Path_DAC_actor_L1.pth"
                    )
                    torch.save(
                        {'net': actors[1].action_net.state_dict(), 'opt': actors[1].optimizer.state_dict()},
                        "F:/MASAC-UAVs-path-planning-master/Path_DAC_actor_F1.pth"
                    )
                    torch.save(
                        {'net': actors[2].action_net.state_dict(), 'opt': actors[2].optimizer.state_dict()},
                        "F:/MASAC-UAVs-path-planning-master/Path_DAC_actor_E1.pth"
                    )

        # 统计与绘图
        all_ep_r_mean = np.mean(np.array(all_ep_r),  axis=0)
        all_ep_r_std  = np.std(np.array(all_ep_r),   axis=0)
        all_ep_L_mean = np.mean(np.array(all_ep_r0), axis=0)
        all_ep_L_std  = np.std(np.array(all_ep_r0),  axis=0)
        all_ep_F_mean = np.mean(np.array(all_ep_r1), axis=0)
        all_ep_F_std  = np.std(np.array(all_ep_r1),  axis=0)
        all_ep_E_mean = np.mean(np.array(all_ep_r2), axis=0)
        all_ep_E_std  = np.std(np.array(all_ep_r2),  axis=0)
        d = {
            "all_ep_r_mean": all_ep_r_mean, "all_ep_r_std": all_ep_r_std,
            "all_ep_L_mean": all_ep_L_mean, "all_ep_L_std": all_ep_L_std,
            "all_ep_F_mean": all_ep_F_mean, "all_ep_F_std": all_ep_F_std,
            "all_ep_E_mean": all_ep_E_mean, "all_ep_E_std": all_ep_E_std,
        }
        with open(shoplistfile, 'wb') as f:
            pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)

        all_ep_r_max = all_ep_r_mean + all_ep_r_std * 0.95
        all_ep_r_min = all_ep_r_mean - all_ep_r_std * 0.95
        all_ep_L_max = all_ep_L_mean + all_ep_L_std * 0.95
        all_ep_L_min = all_ep_L_mean - all_ep_L_std * 0.95
        all_ep_F_max = all_ep_F_mean + all_ep_F_std * 0.95
        all_ep_F_min = all_ep_F_mean - all_ep_F_std * 0.95
        all_ep_E_max = all_ep_E_mean + all_ep_E_std * 0.95
        all_ep_E_min = all_ep_E_mean - all_ep_E_std * 0.95

        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_r_mean)), all_ep_r_mean, label='MADAC', color='#3a86ff')
        plt.fill_between(np.arange(len(all_ep_r_mean)), all_ep_r_max, all_ep_r_min,
                         alpha=0.6, facecolor='#3a86ff')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.figure(2, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_L_mean)), all_ep_L_mean, label='MADAC', color='#3a86ff')
        plt.fill_between(np.arange(len(all_ep_L_mean)), all_ep_L_max, all_ep_L_min,
                         alpha=0.6, facecolor='#3a86ff')
        plt.xlabel('Episode')
        plt.ylabel('Hero0 reward')
        plt.figure(3, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_F_mean)), all_ep_F_mean, label='MADAC', color='#3a86ff')
        plt.fill_between(np.arange(len(all_ep_F_mean)), all_ep_F_max, all_ep_F_min,
                         alpha=0.6, facecolor='#3a86ff')
        plt.xlabel('Episode')
        plt.ylabel('Hero1 reward')
        plt.figure(4, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_E_mean)), all_ep_E_mean, label='MADAC', color='#3a86ff')
        plt.fill_between(np.arange(len(all_ep_E_mean)), all_ep_E_max, all_ep_E_min,
                         alpha=0.6, facecolor='#3a86ff')
        plt.xlabel('Episode')
        plt.ylabel('Hero2 reward')
        plt.legend()
        plt.show()
        env.close()

    # --------------------------------------------------------
    #  测试模式
    # --------------------------------------------------------
    else:
        print('MADAC测试中...')
        aa = Actor()
        checkpoint_aa = torch.load("F:/MASAC-UAVs-path-planning-master/Path_DAC_actor_L1.pth")
        aa.action_net.load_state_dict(checkpoint_aa['net'])
        bb = Actor()
        checkpoint_bb = torch.load("F:/MASAC-UAVs-path-planning-master/Path_DAC_actor_F1.pth")
        bb.action_net.load_state_dict(checkpoint_bb['net'])
        cc = Actor()
        checkpoint_cc = torch.load("F:/MASAC-UAVs-path-planning-master/Path_DAC_actor_E1.pth")
        cc.action_net.load_state_dict(checkpoint_cc['net'])

        action              = np.zeros((N_Agent + M_Enemy, action_number))
        win_times           = 0
        average_timestep    = 0
        average_integral_V  = 0
        average_integral_U  = 0
        average_integral_U0 = 0
        average_integral_U1 = 0
        average_integral_U2 = 0
        all_ep_V, all_ep_U, all_ep_T, all_ep_F = [], [], [], []

        for j in range(TEST_EPIOSDE):
            state         = env.reset()
            total_rewards = 0
            integral_V    = 0
            integral_U    = 0
            integral_U0   = integral_U1 = integral_U2 = 0

            for timestep in range(EP_LEN):
                action[0] = aa.choose_action(state[0])
                action[1] = bb.choose_action(state[1])
                action[2] = cc.choose_action(state[2])
                new_state, reward, done, win, team_counter = env.step(action)
                if win[0] and win[1] and win[2]:
                    win_times += 1
                integral_V  += state[0][2]
                integral_U  += abs(action[0]).sum()
                integral_U0 += action[0][0] ** 2 + action[0][1] ** 2
                integral_U1 += action[1][0] ** 2 + action[1][1] ** 2
                integral_U2 += action[2][0] ** 2 + action[2][1] ** 2
                total_rewards += reward.mean()
                state = new_state
                if RENDER:
                    env.render()
                if done:
                    break

            FKR                  = team_counter / timestep
            average_timestep    += timestep
            average_integral_V  += integral_V
            average_integral_U  += integral_U
            average_integral_U0 += integral_U0
            average_integral_U1 += integral_U1
            average_integral_U2 += integral_U2
            print("Score", total_rewards)
            all_ep_V.append(integral_V)
            all_ep_U.append(integral_U)
            all_ep_T.append(timestep)
            all_ep_F.append(FKR)

        print('任务完成率',       win_times / TEST_EPIOSDE)
        print('平均最短飞行时间', average_timestep    / TEST_EPIOSDE)
        print('平均最短飞行路程', average_integral_V  / TEST_EPIOSDE)
        print('平均最小能量损耗', average_integral_U  / TEST_EPIOSDE)
        print('hero0平均最小能量损耗', average_integral_U0 / TEST_EPIOSDE)
        print('hero1平均最小能量损耗', average_integral_U1 / TEST_EPIOSDE)
        print('hero2平均最小能量损耗', average_integral_U2 / TEST_EPIOSDE)
        env.close()


if __name__ == '__main__':
    main()
