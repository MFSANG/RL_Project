import numpy as np
import torch
import os
import json
from ppo import PPO  # 替换为 PPO 版本

class SecureCommEnvironment:
    def __init__(self, state_dim=8, action_dim=4, max_steps=200):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
        self.energy_capacity = 100.0  # 初始能量容量(J)
        self.energy_consumption_rate = 0.1
        self.training_phase = 0.0  # 训练进度(0-1)

        # 噪声和信道参数
        self.noise_power = 0.1
        self.channel_gain_legitimate = 1.0
        self.channel_gain_eavesdropper = 0.3

        self.reset()

    def reset(self):
        """重置环境状态"""
        self.current_step = 0
        self.energy_remaining = self.energy_capacity

        # 生成初始信道状态 (随机)
        self.channel_state = np.random.normal(0, 1, self.state_dim)
        return self.channel_state

    def step(self, action):
        """执行一步动作，返回新的状态、奖励、是否完成和额外信息"""
        self.current_step += 1

        # 计算传输功率 (基于动作)
        # 降低传输功率的影响
        transmit_power = np.sum(np.abs(action) ** 2) * 0.2

        # 计算能量消耗
        energy_consumed = transmit_power + self.energy_consumption_rate
        self.energy_remaining -= energy_consumed

        # 计算保密速率
        snr_legitimate = transmit_power * self.channel_gain_legitimate / self.noise_power
        snr_eavesdropper = transmit_power * self.channel_gain_eavesdropper / self.noise_power

        # 随机波动的信道增益
        legitimate_gain = self.channel_gain_legitimate * (1 + 0.2 * np.random.normal())
        eavesdropper_gain = self.channel_gain_eavesdropper * (1 + 0.1 * np.random.normal())

        # 计算实际SNR
        actual_snr_legitimate = transmit_power * legitimate_gain / self.noise_power
        actual_snr_eavesdropper = transmit_power * eavesdropper_gain / self.noise_power

        # 计算保密速率 (比特/秒/赫兹)
        secrecy_rate = max(0, np.log2(1 + actual_snr_legitimate) - np.log2(1 + actual_snr_eavesdropper))

        # 根据训练阶段调整奖励计算
        efficiency_weight = 0.2 + 0.3 * self.training_phase  # 随着训练进行，效率权重增加

        # 计算奖励 (保密速率与能量效率的平衡)
        reward = secrecy_rate * (1.0 - efficiency_weight) + (
                    self.energy_remaining / self.energy_capacity) * efficiency_weight

        # 能量耗尽或达到最大步数则结束
        done = (self.energy_remaining <= 0) or (self.current_step >= self.max_steps)

        # 更新信道状态 (加入一些随机性)
        self.channel_state = 0.7 * self.channel_state + 0.3 * np.random.normal(0, 1, self.state_dim)

        # 超出能量限制的惩罚
        if self.energy_remaining <= 0:
            reward = -1.0
            self.energy_remaining = 0

        info = {
            'secrecy_rate': secrecy_rate,
            'energy_remaining': self.energy_remaining,
            'energy_consumed': energy_consumed
        }

        return self.channel_state, reward, done, info

    def set_training_progress(self, episode, total_episodes):
        """设置训练进度，用于调整环境参数"""
        self.training_phase = episode / total_episodes

def main():

    np.random.seed(42)
    torch.manual_seed(42)

    state_dim = 8
    action_dim = 4
    action_bounds = (
        np.array([-1.0] * action_dim),
        np.array([1.0] * action_dim)
    )

    env = SecureCommEnvironment(state_dim=state_dim, action_dim=action_dim, max_steps=200)
    agent = PPO(state_dim, action_dim, action_bounds)
    num_episodes = 100

    print("开始 PPO 算法训练...")
    for episode in range(num_episodes):
        env.set_training_progress(episode, num_episodes)
        state = env.reset()
        ep_reward = 0
        ep_secrecy = []
        ep_energy = 0

        for _ in range(env.max_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.buffer.states.append(state)
            agent.buffer.actions.append(action)
            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(done)
            agent.buffer.log_probs.append(log_prob)
            agent.buffer.values.append(value)

            state = next_state
            ep_reward += reward
            ep_secrecy.append(info.get('secrecy_rate', 0.0))
            ep_energy = info.get('energy_remaining', 0.0)

            if done:
                break

        agent.update()
        avg_sr = np.mean(ep_secrecy)
        agent.record_metrics(ep_reward, avg_sr, ep_energy)

        if episode % 5 == 0:
            print(f"Episode {episode}/{num_episodes} | Reward: {ep_reward:.2f} | SR: {avg_sr:.4f} | Best SR: {agent.best_sr:.4f}")

    agent.load_best_model()
    agent.plot_learning_curve()
    agent.print_energy_stats()

    output_dir = 'results/ppo'
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'reward_history': [float(x) for x in agent.rewards_log],
        'avg_reward_history': [float(x) for x in agent.avg_reward_log],
        'secrecy_rate_history': [float(x) for x in agent.sr_log],
        'energy_history': [float(x) for x in agent.energy_log],
        'loss_log': [float(x) for x in agent.loss_log],
        'final_avg_reward': float(np.mean(agent.rewards_log[-10:])),
        'final_avg_secrecy_rate': float(max(agent.sr_log[-10:])),
    }

    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"PPO训练完成! 结果保存在 {output_dir}/metrics.json")
    print(f"最终平均奖励: {results['final_avg_reward']:.4f}")
    print(f"最终平均保密速率: {results['final_avg_secrecy_rate']:.4f} bps/Hz")


if __name__ == '__main__':
    main()