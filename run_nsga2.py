import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from nsga2 import NSGA2

class SecureCommEnvironment:
    """
    安全通信环境模拟，用于NSGA-II训练
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
        # 降低传输功率的影响
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
    # 设置随机种子以便结果可重现
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 环境参数设置
    state_dim = 8
    action_dim = 4
    
    # 创建环境
    env = SecureCommEnvironment(state_dim=state_dim, action_dim=action_dim, max_steps=200)
    
    # 动作范围
    variable_boundaries = [
        [-1.0, 1.0],  # 动作1范围
        [-1.0, 1.0],  # 动作2范围
        [-1.0, 1.0],  # 动作3范围
        [-1.0, 1.0]   # 动作4范围
    ]
    
    # 创建NSGA-II优化器，增加种群大小和迭代次数
    nsga2 = NSGA2(
        num_variables=action_dim,
        objectives=2,  # 保密速率和能量效率
        variable_boundaries=variable_boundaries,
        population_size=100,  # 从50增加到100
        max_generations=100   # 从50增加到100
    )
    
    print("开始NSGA-II算法优化...")
    
    # 运行NSGA-II优化
    pareto_front, objective_history = nsga2.optimize(env)
    
    # 保存结果
    output_dir = 'results/nsga2'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存统计数据
    statistics = nsga2.save_results(pareto_front, objective_history, output_dir)
    
    # 绘制帕累托前沿
    nsga2.plot_pareto_front(pareto_front, save_path=f'{output_dir}/pareto_front.png')
    
    # 绘制目标函数历史
    nsga2.plot_objective_history(save_dir=output_dir)
    
    # 绘制收敛历史
    nsga2.plot_convergence(save_dir=output_dir)
    

    best_solutions = []
    
    # 提取最大保密速率的解
    secrecy_rates = [-sol[action_dim] for sol in pareto_front]
    if secrecy_rates:
        max_sr_idx = np.argmax(secrecy_rates)
        best_sr_solution = {
            'type': 'max_secrecy_rate',
            'secrecy_rate': secrecy_rates[max_sr_idx],
            'energy_efficiency': -pareto_front[max_sr_idx][action_dim + 1],
            'variables': pareto_front[max_sr_idx][:action_dim].tolist()
        }
        best_solutions.append(best_sr_solution)
    
    # 提取最大能量效率的解
    energy_efficiencies = [-sol[action_dim + 1] for sol in pareto_front]
    if energy_efficiencies:
        max_ee_idx = np.argmax(energy_efficiencies)
        best_ee_solution = {
            'type': 'max_energy_efficiency',
            'secrecy_rate': -pareto_front[max_ee_idx][action_dim],
            'energy_efficiency': energy_efficiencies[max_ee_idx],
            'variables': pareto_front[max_ee_idx][:action_dim].tolist()
        }
        best_solutions.append(best_ee_solution)
    
    # 保存最优解
    with open(f'{output_dir}/best_solutions.json', 'w') as f:
        json.dump(best_solutions, f, indent=4)
    
    # 运行最优解进行可视化分析
    print("\n进行最优解分析...")
    
    if best_solutions:
        # 分析最大保密速率解
        if len(best_solutions) > 0:
            best_sr_vars = best_solutions[0]['variables']
            print(f"\n最大保密速率解 ({best_solutions[0]['secrecy_rate']:.4f} bps/Hz):")
            state = env.reset()
            total_sr = 0
            step_count = 0
            sr_values = []
            energy_values = []
            
            while True:
                action = np.array(best_sr_vars)
                next_state, reward, done, info = env.step(action)
                
                total_sr += info['secrecy_rate']
                step_count += 1
                sr_values.append(info['secrecy_rate'])
                energy_values.append(info['energy_remaining'])
                
                state = next_state
                if done:
                    break
            
            avg_sr = total_sr / step_count if step_count > 0 else 0
            print(f"  实际平均保密速率: {avg_sr:.4f} bps/Hz")
            print(f"  剩余能量: {energy_values[-1]:.2f}J")
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(sr_values)
            plt.title('Secrecy Rate Curve for Best SR Solution')
            plt.xlabel('Time Step')
            plt.ylabel('Secrecy Rate (bps/Hz)')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(energy_values)
            plt.title('Energy Curve for Best SR Solution')
            plt.xlabel('Time Step')
            plt.ylabel('Remaining Energy (J)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/best_sr_solution_analysis.png')
    
    print(f"\nNSGA-II优化完成! 结果保存在 {output_dir} 目录")
    print(f"帕累托前沿大小: {len(pareto_front)}")
    print(f"最大保密速率: {statistics['max_secrecy_rate']:.4f} bps/Hz")
    print(f"最大能量效率: {statistics['max_energy_efficiency']:.4f}")
    
    return statistics


if __name__ == "__main__":
    main() 