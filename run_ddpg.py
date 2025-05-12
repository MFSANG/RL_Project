import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from ddpg import DDPG

class SecureCommEnvironment:
    """
    安全通信环境模拟，用于DDPG训练
    """
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
        transmit_power = np.sum(np.abs(action)**2) * 0.2
        
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
        reward = secrecy_rate * (1.0 - efficiency_weight) + (self.energy_remaining / self.energy_capacity) * efficiency_weight
        
        # 能量耗尽或达到最大步数则结束
        done = (self.energy_remaining <= 0.1) or (self.current_step >= self.max_steps)
        
        
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
        self.training_phase = episode / total_episodes


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 环境参数设置
    state_dim = 8
    action_dim = 4
    action_bounds = (
        np.array([-1.0, -1.0, -1.0, -1.0]),  # 下限
        np.array([1.0, 1.0, 1.0, 1.0])       # 上限
    )
    
    # 创建环境
    env = SecureCommEnvironment(state_dim=state_dim, action_dim=action_dim, max_steps=200)
    
    # DDPG代理
    agent = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=action_bounds,
        buffer_size=100000,
        batch_size=64,
        gamma=0.98,
        tau=0.001,
        actor_lr=1e-4,
        critic_lr=1e-3
    )
    
    # 训练参数
    num_episodes = 100  # 总训练回合数
    
    print("开始DDPG算法训练...")
    
    # 开始训练
    results = agent.learn(env, num_episodes, max_steps=200)
    
    # 打印能量统计信息
    if results['energy_history']:
        print("\nEnergy statistics:")
        print(f"  Average remaining energy: {np.mean(results['energy_history']):.2f} J")
        print(f"  Min remaining energy: {min(results['energy_history']):.2f} J")
        print(f"  Max remaining energy: {max(results['energy_history']):.2f} J")
    
    # 保存训练结果
    output_dir = 'results/ddpg'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存指标数据为JSON
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # 绘制和保存学习曲线
    agent.plot_learning_curve()
    
    # 保存最终模型
    agent.save_model(f'{output_dir}/final_actor.pt', f'{output_dir}/final_critic.pt')
    
    print(f"DDPG训练完成! 结果保存在 {output_dir} 目录")
    print(f"最终平均奖励: {results['final_avg_reward']:.4f}")
    print(f"最终平均保密速率: {results['final_avg_secrecy_rate']:.4f} bps/Hz")
    
   
    plt.figure(figsize=(10, 6))
    plt.plot(results['reward_history'], 'b-', alpha=0.7)
    plt.plot(results['avg_reward_history'], 'r-')
    plt.title('DDPG Training Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(['Episode Reward', 'Average Reward'])
    plt.grid(True)
    plt.savefig(f'{output_dir}/reward_curve.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['secrecy_rate_history'], 'g-')
    plt.title('DDPG Training Secrecy Rate Curve')
    plt.xlabel('Episode')
    plt.ylabel('Secrecy Rate (bps/Hz)')
    plt.grid(True)
    plt.savefig(f'{output_dir}/secrecy_rate_curve.png')


if __name__ == "__main__":
    main() 