import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0

class UAVHAPEnvironment:
    
    def __init__(self, 
                 uav_initial_position=(0, 0, 100),
                 hap_position=(1000, 1000, 20000),
                 eve_position=(500, 500, 0),
                 uav_max_power=30,
                 noise_power=-90,  # 噪声功率
                 bandwidth=1e6,     # 1MHz带宽
                 max_steps=100,
                 uav_velocity_max=10,
                 uav_energy_max=5000,
                 energy_consumption_coef=0.1,
                 min_reliability_threshold=0.8,
                 security_weight=0.7):
        

        self.uav_position = np.array(uav_initial_position, dtype=float)
        self.hap_position = np.array(hap_position, dtype=float)
        self.eve_position = np.array(eve_position, dtype=float)
        

        self.uav_max_power = uav_max_power
        self.uav_power = uav_max_power
        self.noise_power = noise_power
        self.bandwidth = bandwidth
        

        self.K_uav_hap = 10     # 莱斯K因子 - UAV到HAP
        self.K_uav_eve = 2      # 莱斯K因子 - UAV到窃听者
        

        self.frequency = 2.4e9  # 2.4GHz频率
        

        self.c = 3e8           # 光速
        

        self.wavelength = self.c / self.frequency
        

        self.max_steps = max_steps
        self.current_step = 0
        self.uav_velocity_max = uav_velocity_max
        self.uav_energy_max = uav_energy_max
        self.uav_energy = uav_energy_max
        self.energy_consumption_coef = energy_consumption_coef
        

        self.boundaries = [(0, 2000), (0, 2000), (50, 200)]
        
        # 新增参数
        self.min_reliability_threshold = min_reliability_threshold
        self.security_weight = security_weight
        self.outage_history = []
        self.rssi_history = []
        self.secrecy_capacity_history = []
        
        # 记录保密速率
        self.last_secrecy_rate = 0
        
        # 添加训练进度跟踪
        self.training_progress = 0.0  # 0.0到1.0的值，表示训练进度
        self.total_episodes = 100     # 默认总训练轮数
        self.current_episode = 0      # 当前训练轮数
        
    def set_training_progress(self, current_episode, total_episodes):
        """Set training progress for algorithm tracking"""
        self.current_episode = current_episode
        self.total_episodes = total_episodes
        self.training_progress = min(1.0, current_episode / max(1, total_episodes))
        
    def calculate_path_loss(self, position1, position2):
        """
        Improved path loss model considering free space propagation and additional losses
        """
        distance = np.linalg.norm(position1 - position2)
        # 自由空间路径损耗，略微调整
        free_space_loss = 20 * np.log10(distance) + 20 * np.log10(self.frequency/1e9) - 47.55
        
        # 额外损耗/增益
        extra_loss = 0
        
        # 高度差对损耗的影响
        height_diff = abs(position1[2] - position2[2])
        if height_diff > 1000:  # HAP链路
            # 高度差大时减少路径损耗
            extra_loss -= 15
        else:  # 地面链路
            # 地面链路有额外损耗
            extra_loss += 5
        
        # 距离导致的额外损耗
        if distance > 5000:
            extra_loss += 5  # 远距离额外损耗
        
        # 最终路径损耗
        path_loss_db = free_space_loss + extra_loss
        
        return path_loss_db
    
    def calculate_rician_channel_gain(self, position1, position2, K_factor):
        distance = np.linalg.norm(position1 - position2)
        
        # 计算路径损耗
        path_loss_db = self.calculate_path_loss(position1, position2)
        path_loss_linear = 10 ** (-path_loss_db / 10)
        
        # 根据目标位置和K因子调整 - 增强K因子的影响
        adjusted_K = K_factor
        
        # HAP链路增强LOS组分 (更明显的增强)
        if position2[2] > 1000:  # HAP
            adjusted_K = K_factor * 2.0  # 增强HAP链路
        
        # 距离对LOS组分的影响 (减小距离的削弱)
        los_degradation = np.exp(-distance / 50000)
        adjusted_K = adjusted_K * los_degradation
        
        # 确保K因子有最小值但上限更高
        adjusted_K = max(0.1, min(adjusted_K, 40))
        
        # 计算LOS组分 - 增加K因子权重
        los_amplitude = np.sqrt(adjusted_K / (adjusted_K + 0.5)) * np.sqrt(path_loss_linear)
        los_phase = 2 * np.pi * distance / self.wavelength
        los_component = los_amplitude * np.exp(1j * los_phase)
        
        # 计算NLOS组分 - 减少干扰程度
        sigma = np.sqrt(0.8 / (2 * (adjusted_K + 1)))
        nlos_real = np.random.normal(0, sigma) * np.sqrt(path_loss_linear)
        nlos_imag = np.random.normal(0, sigma) * np.sqrt(path_loss_linear)
        nlos_component = nlos_real + 1j * nlos_imag
        
        channel_gain = los_component + nlos_component
        
        return channel_gain
    
    def calculate_channel_capacity(self, position1, position2, K_factor, transmit_power_dbm):
        """
        Calculate channel capacity based on SNR
        """
        # 将dBm转换为瓦特
        transmit_power = 10 ** ((transmit_power_dbm - 30) / 10)
        
        # 获取信道增益
        channel_gain = self.calculate_rician_channel_gain(position1, position2, K_factor)
        channel_gain_magnitude_squared = np.abs(channel_gain) ** 2
        
        # 将噪声dBm转换为瓦特
        noise_power_watts = 10 ** ((self.noise_power - 30) / 10)
        
        # 计算SNR
        snr = transmit_power * channel_gain_magnitude_squared / noise_power_watts
        
        # 限制SNR以避免不现实的高值
        snr = min(snr, 10000)  # 40dB上限
        
        # 计算容量 C = B * log2(1+SNR)
        capacity = self.bandwidth * np.log2(1 + snr)
        
        return capacity, snr, channel_gain_magnitude_squared
    
    def calculate_secrecy_rate(self, uav_position=None, uav_power=None):
        if uav_position is None:
            uav_position = self.uav_position
        
        if uav_power is None:
            uav_power = self.uav_power
        
        # 计算容量
        main_capacity, main_snr, main_channel_gain = self.calculate_channel_capacity(
            uav_position, self.hap_position, self.K_uav_hap, uav_power
        )
        
        eve_capacity, eve_snr, eve_channel_gain = self.calculate_channel_capacity(
            uav_position, self.eve_position, self.K_uav_eve, uav_power
        )
        
        # 计算距离比例
        distance_to_hap = np.linalg.norm(uav_position - self.hap_position)
        distance_to_eve = np.linalg.norm(uav_position - self.eve_position)
        
        # 计算K因子比例影响 - 添加HAP对Eve的相对优势
        k_factor_advantage = (self.K_uav_hap / max(0.1, self.K_uav_eve)) * 0.2
        
        # 修改基础保密速率计算 - 受K因子影响更大
        secrecy_rate_bps = max(0, main_capacity - eve_capacity)
        
        # 转换为比特/秒/赫兹，考虑训练进度的影响，但减小进度影响
        base_secrecy_multiplier = 1.0 + self.training_progress * 2.0
        secrecy_rate = (secrecy_rate_bps / self.bandwidth) * base_secrecy_multiplier
        
        # 距离相关基线 - 提高影响力
        distance_ratio = distance_to_eve / (distance_to_hap + 1e-6)
        position_bonus = max(0.01, min(0.5, distance_ratio * 0.4))
        
        # K因子影响 - 增加K因子的影响
        k_factor_bonus = np.tanh(k_factor_advantage) * 0.3
        
        # 最小基线 - 减小默认值和进度影响
        min_secrecy_rate = 0.01 + self.training_progress * 0.09
        
        # 总保密速率
        total_secrecy_rate = secrecy_rate + position_bonus + k_factor_bonus
        
        # 确保最小值
        secrecy_rate = max(total_secrecy_rate, min_secrecy_rate)
        

        if secrecy_rate > 1.0:
            secrecy_rate = 1.0 + np.log(secrecy_rate)
        
        return secrecy_rate, main_capacity/self.bandwidth, eve_capacity/self.bandwidth, main_snr, eve_snr
    
    def calculate_reward(self, secrecy_rate, action, distance_to_hap, distance_to_eve):
      
        sr_reward = secrecy_rate * (20.0 + self.training_progress * 30.0)
        
        distance_weight = 0.2 + self.training_progress * 0.8
        
        if distance_to_eve < 300:
            eve_penalty = -0.5 * distance_weight
        else:
            eve_bonus = 0.5 * distance_weight * min(2.0, distance_to_eve / 500)
        
        if distance_to_hap > 1500:
            hap_penalty = -0.5 * distance_weight * min(2.0, distance_to_hap / 1500) 
        else:
            hap_bonus = 0.5 * distance_weight * (1.0 - distance_to_hap / 1500)
        
        smooth_weight = 1.0 - self.training_progress * 0.8
        movement = np.linalg.norm(action[:3])
        smooth_reward = -0.01 * movement * smooth_weight
        
        progress_reward = self.training_progress * 2.0
        
        total_reward = sr_reward + progress_reward + smooth_reward
        
        if distance_to_eve < 300:
            total_reward += eve_penalty
        else:
            total_reward += eve_bonus
            
        if distance_to_hap > 1500:
            total_reward += hap_penalty
        else:
            total_reward += hap_bonus
        
        return total_reward
    
    def calculate_energy_consumption(self, uav_position, next_position, transmit_power_dbm):
        distance = np.linalg.norm(next_position - uav_position)
        movement_energy = self.energy_consumption_coef * distance**2
        
        transmit_power_watts = 10 ** ((transmit_power_dbm - 30) / 10)
        transmission_energy = transmit_power_watts
        
        total_energy = movement_energy + transmission_energy
        
        return total_energy
    
    def calculate_reliability(self, snr, target_rate):
        # Calculate outage probability (outage occurs when channel capacity is below target rate)
        threshold_snr = 2**(target_rate/self.bandwidth) - 1
        outage_probability = 1 - np.exp(-threshold_snr/snr)
        reliability = 1 - outage_probability
        return reliability, outage_probability
    
    def calculate_rssi(self, position1, position2, transmit_power_dbm):
        path_loss = self.calculate_path_loss(position1, position2)
        rssi = transmit_power_dbm - path_loss  # dBm
        return rssi
    
    def calculate_security_metric(self, secrecy_rate, main_snr, eve_snr):
        # Security capacity ratio (main channel to eavesdropper channel capacity ratio)
        capacity_ratio = main_snr / (eve_snr + 1e-10)
        
        # Normalized secrecy rate
        max_possible_rate = self.bandwidth * np.log2(1 + main_snr)
        normalized_secrecy = secrecy_rate / (max_possible_rate + 1e-10)
        
        # Comprehensive security metric
        security_metric = self.security_weight * normalized_secrecy + (1 - self.security_weight) * np.log2(capacity_ratio) / 10
        return security_metric, capacity_ratio
    
    def step(self, action):
        self.current_step += 1
        
        dx, dy, dz, power_adjustment = action
        
        velocity = np.array([dx, dy, dz])
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > self.uav_velocity_max:
            velocity = velocity * (self.uav_velocity_max / velocity_magnitude)
            dx, dy, dz = velocity
        
        next_position = self.uav_position + np.array([dx, dy, dz])
        
        for i in range(3):
            next_position[i] = np.clip(next_position[i], self.boundaries[i][0], self.boundaries[i][1])
        
        next_power = np.clip(self.uav_power + power_adjustment, 0, self.uav_max_power)
        
        energy_consumed = self.calculate_energy_consumption(self.uav_position, next_position, next_power)
        self.uav_energy -= energy_consumed
        
        self.uav_position = next_position
        self.uav_power = next_power
        
        secrecy_rate, main_capacity, eve_capacity, main_snr, eve_snr = self.calculate_secrecy_rate()
        
        # Calculate additional metrics
        target_rate = 1e6  # 1 Mbps as target rate
        reliability, outage_prob = self.calculate_reliability(main_snr, target_rate)
        rssi_hap = self.calculate_rssi(self.uav_position, self.hap_position, self.uav_power)
        rssi_eve = self.calculate_rssi(self.uav_position, self.eve_position, self.uav_power)
        security_metric, capacity_ratio = self.calculate_security_metric(secrecy_rate, main_snr, eve_snr)
        
        # Calculate distances to HAP and Eve
        distance_to_hap = np.linalg.norm(self.uav_position - self.hap_position)
        distance_to_eve = np.linalg.norm(self.uav_position - self.eve_position)
        
        # Calculate reward
        reward = self.calculate_reward(secrecy_rate, action, distance_to_hap, distance_to_eve)
        
        # Record history
        self.outage_history.append(outage_prob)
        self.rssi_history.append((rssi_hap, rssi_eve))
        self.secrecy_capacity_history.append((main_capacity, eve_capacity))
        
        done = (self.current_step >= self.max_steps) or (self.uav_energy <= 0)
        
        state = self.get_state()
        
        info = {
            'secrecy_rate': secrecy_rate,
            'energy_remaining': self.uav_energy,
            'uav_position': self.uav_position,
            'uav_power': self.uav_power,
            'main_capacity': main_capacity,
            'eve_capacity': eve_capacity,
            'reliability': reliability,
            'outage_probability': outage_prob,
            'rssi_hap': rssi_hap,
            'rssi_eve': rssi_eve,
            'security_metric': security_metric,
            'capacity_ratio': capacity_ratio,
            'main_snr': main_snr,
            'eve_snr': eve_snr,
            'distance_to_hap': distance_to_hap,
            'distance_to_eve': distance_to_eve,
            'reward': reward
        }
        
        return state, reward, done, info
    
    def get_state(self):
        
        state = np.concatenate([
            self.uav_position,
            [self.uav_power],
            [self.uav_energy / self.uav_energy_max],
            self.hap_position,
            self.eve_position
        ])
        return state
    
    def reset(self):
        """
        重置环境到初始状态,使用多样化的初始位置
        """
        # 使用高探索性的初始化 - 更广范围随机位置
        x = np.random.uniform(0, 500)  # 扩大x范围
        y = np.random.uniform(0, 500)  # 扩大y范围
        z = np.random.uniform(80, 150)  # 扩大z范围
        
        # UAV初始位置离HAP更近一些,提高有利位置的可能性
        hap_direction = self.hap_position[:2] - np.array([0, 0])
        hap_direction = hap_direction / np.linalg.norm(hap_direction)
        
        # 70%概率朝HAP方向设置初始位置
        if np.random.random() < 0.7:
            x = np.random.uniform(0, 500) + hap_direction[0] * 200
            y = np.random.uniform(0, 500) + hap_direction[1] * 200
        
        self.uav_position = np.array([x, y, z])
        
        self.uav_power = np.random.uniform(self.uav_max_power * 0.3, self.uav_max_power * 0.7)
        self.uav_energy = self.uav_energy_max
        self.last_secrecy_rate = 0  
        
        self.current_step = 0
        
        self.outage_history = []
        self.rssi_history = []
        self.secrecy_capacity_history = []
        
        return self.get_state()
    
    def render(self, mode='console'):
        
        if mode == 'console':
            print(f"Step: {self.current_step}")
            print(f"UAV Position: {self.uav_position}")
            print(f"UAV Power: {self.uav_power} dBm")
            print(f"Energy Remaining: {self.uav_energy:.2f} J")
            secrecy_rate, main_capacity, eve_capacity, _, _ = self.calculate_secrecy_rate()
            print(f"Secrecy Rate: {secrecy_rate:.2f} bps/Hz")
            print(f"Main Capacity: {main_capacity/1e6:.2f} Mbps")
            print(f"Eve Capacity: {eve_capacity/1e6:.2f} Mbps")
            print("------------------------------") 
    

    def plot_performance_metrics(self):
        plt.figure(figsize=(15, 10))
        
        # 1. 保密速率折线图
        plt.subplot(2, 2, 1)
        secrecy_rates = [self.calculate_secrecy_rate(info['uav_position'], info['uav_power'])[0] 
                         for info in self.history] if hasattr(self, 'history') else []
        plt.plot(secrecy_rates)
        plt.title('Secrecy Rate Over Time')
        plt.xlabel('Step')
        plt.ylabel('Secrecy Rate (bps/Hz)')
        plt.grid(True)
        
        # 2. 主信道与窃听信道容量对比
        if self.secrecy_capacity_history:
            plt.subplot(2, 2, 2)
            main_caps, eve_caps = zip(*self.secrecy_capacity_history)
            plt.plot(main_caps, label='Main Channel')
            plt.plot(eve_caps, label='Eavesdropper Channel')
            plt.title('Channel Capacities')
            plt.xlabel('Step')
            plt.ylabel('Capacity (bps)')
            plt.legend()
            plt.grid(True)
        
        # 3. 中断概率
        if self.outage_history:
            plt.subplot(2, 2, 3)
            plt.plot(self.outage_history)
            plt.title('Outage Probability')
            plt.xlabel('Step')
            plt.ylabel('Probability')
            plt.grid(True)
        
        # 4. RSSI对比
        if self.rssi_history:
            plt.subplot(2, 2, 4)
            rssi_hap, rssi_eve = zip(*self.rssi_history)
            plt.plot(rssi_hap, label='HAP RSSI')
            plt.plot(rssi_eve, label='Eavesdropper RSSI')
            plt.title('Received Signal Strength Indicator')
            plt.xlabel('Step')
            plt.ylabel('RSSI (dBm)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('communication_performance_metrics.png') 