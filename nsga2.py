import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
import json

class NSGA2:
    """
    NSGA-II (非支配排序遗传算法II) 实现
    用于解决多目标优化问题
    """
    def __init__(self, 
                 num_variables=4,
                 objectives=2,
                 variable_boundaries=None,
                 population_size=100,
                 max_generations=100):
        
        self.num_variables = num_variables  # 变量维度
        self.objectives = objectives        # 目标函数数量
        self.population_size = population_size
        self.max_generations = max_generations
        
        if variable_boundaries is None:
            self.variable_boundaries = np.array([[-1, 1] for _ in range(num_variables)])
        else:
            self.variable_boundaries = np.array(variable_boundaries)
        
        # 存储历史数据
        self.population_history = []
        self.front_history = []
        self.best_solutions = []
        self.convergence_history = []
        self.objective_history = {
            'secrecy_rate': [],
            'energy_efficiency': []
        }
        
    def initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            chromosome = []
            for j in range(self.num_variables):
                gene = random.uniform(
                    self.variable_boundaries[j][0],
                    self.variable_boundaries[j][1]
                )
                chromosome.append(gene)
            population.append(chromosome)
        return np.array(population)
    
    def calculate_objectives(self, chromosome, env):
        """计算目标函数值"""
        # 在通信环境中评估解决方案
        state = env.reset()
        total_reward = 0
        total_secrecy_rate = 0
        step_count = 0
        initial_energy = env.energy_remaining
        
        done = False
        while not done and step_count < env.max_steps:
            action = np.array(chromosome)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            total_secrecy_rate += info['secrecy_rate']
            state = next_state
            step_count += 1
        
        avg_secrecy_rate = total_secrecy_rate / step_count if step_count > 0 else 0
        energy_remaining = info['energy_remaining'] if 'energy_remaining' in info else 0
        energy_consumed = initial_energy - energy_remaining
        
        # 修改能量效率计算方式
        # 只有当能量消耗大于0时才计算效率，避免除以0
        if energy_consumed > 0.1:
            energy_efficiency = avg_secrecy_rate / energy_consumed
        else:
            energy_efficiency = avg_secrecy_rate * 0.1  # 默认能量效率
        
        # 缩放能量效率以与保密速率在同一数量级
        energy_efficiency = energy_efficiency * 10.0
        
        # 目标1：最大化保密速率, 目标2：最大化能量效率
        # 注意：由于NSGA-II默认是最小化问题，我们取负值
        return [-avg_secrecy_rate, -energy_efficiency]
    
    def fast_non_dominated_sort(self, population_with_objectives):
        """快速非支配排序"""
        fronts = [[]]
        S = [[] for _ in range(len(population_with_objectives))]
        n = [0] * len(population_with_objectives)
        rank = [0] * len(population_with_objectives)
        
        # 对于每个个体p
        for p in range(len(population_with_objectives)):
            S[p] = []
            n[p] = 0
            # 对于每个个体q
            for q in range(len(population_with_objectives)):
                # 检查p是否支配q
                if self.dominates(population_with_objectives[p], population_with_objectives[q]):
                    if q not in S[p]:
                        S[p].append(q)
                # 检查q是否支配p
                elif self.dominates(population_with_objectives[q], population_with_objectives[p]):
                    n[p] += 1
            
            if n[p] == 0:
                rank[p] = 0
                if p not in fronts[0]:
                    fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)
            i += 1
            fronts.append(Q)
        
        # 移除最后一个空前沿
        del fronts[-1]
        
        return fronts, rank
    
    def dominates(self, p, q):
        """检查个体p是否支配个体q"""
        # p必须至少在一个目标上严格优于q，且在其他目标上不差于q
        better_in_any = False
        for i in range(self.objectives):
            if p[self.num_variables + i] > q[self.num_variables + i]:  # p在某个目标上更差
                return False
            elif p[self.num_variables + i] < q[self.num_variables + i]:  # p在某个目标上更好
                better_in_any = True
        
        return better_in_any
    
    def calculate_crowding_distance(self, front, population_with_objectives):
        """计算拥挤度距离"""
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        distances = [0] * len(front)
        for i in range(len(front)):
            distances[i] = 0
        
        for objective in range(self.objectives):
            # 按目标函数值排序
            sorted_front = sorted(front, key=lambda x: population_with_objectives[x][self.num_variables + objective])
            
            # 边界点距离设为无穷大
            distances[0] = float('inf')
            distances[-1] = float('inf')
            
            # 计算中间点的距离
            for i in range(1, len(front) - 1):
                objective_range = (
                    population_with_objectives[sorted_front[-1]][self.num_variables + objective] - 
                    population_with_objectives[sorted_front[0]][self.num_variables + objective]
                )
                
                # 避免除零错误
                if objective_range == 0:
                    continue
                
                # 计算归一化的拥挤度距离
                distances[i] += (
                    population_with_objectives[sorted_front[i+1]][self.num_variables + objective] - 
                    population_with_objectives[sorted_front[i-1]][self.num_variables + objective]
                ) / objective_range
        
        return distances
    
    def selection(self, population_with_objectives, fronts, crowding_distances):
        """选择操作"""
        selected = []
        
        # 按前沿顺序选择个体
        i = 0
        while len(selected) + len(fronts[i]) <= self.population_size:
            # 添加整个前沿
            selected.extend(fronts[i])
            i += 1
        
        # 如果还需要更多个体，根据拥挤度选择
        if len(selected) < self.population_size:
            # 获取当前前沿
            current_front = fronts[i]
            
            # 按拥挤度距离排序
            current_front_distances = [(idx, crowding_distances[i][j]) for j, idx in enumerate(current_front)]
            current_front_sorted = sorted(current_front_distances, key=lambda x: x[1], reverse=True)
            
            # 选择拥挤度最大的个体
            remaining = self.population_size - len(selected)
            for j in range(remaining):
                selected.append(current_front_sorted[j][0])
        
        # 创建新的种群
        new_population = []
        for i in selected:
            new_population.append(population_with_objectives[i][:self.num_variables])
        
        return np.array(new_population)
    
    def crossover(self, parent1, parent2, crossover_rate=0.9):
        """模拟二进制交叉"""
        if random.random() > crossover_rate:
            return parent1, parent2
        
        child1, child2 = [], []
        # 模拟二进制交叉
        eta = 20  # 分布指数
        
        for i in range(self.num_variables):
            if random.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                    
                    lb = self.variable_boundaries[i][0]
                    ub = self.variable_boundaries[i][1]
                    
                    rand = random.random()
                    beta = 1.0 + (2.0 * (y1 - lb) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    
                    if rand <= (1.0 / alpha):
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    
                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    
                    beta = 1.0 + (2.0 * (ub - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    
                    if rand <= (1.0 / alpha):
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    
                    c1 = min(max(c1, lb), ub)
                    c2 = min(max(c2, lb), ub)
                    
                    if random.random() <= 0.5:
                        child1.append(c1)
                        child2.append(c2)
                    else:
                        child1.append(c2)
                        child2.append(c1)
                else:
                    child1.append(parent1[i])
                    child2.append(parent2[i])
            else:
                child1.append(parent1[i])
                child2.append(parent2[i])
        
        return np.array(child1), np.array(child2)
    
    def mutation(self, individual, mutation_rate=0.1):
        """多项式变异"""
        child = individual.copy()
        
        eta = 20  # 分布指数
        
        for i in range(self.num_variables):
            if random.random() <= mutation_rate:
                y = individual[i]
                lb = self.variable_boundaries[i][0]
                ub = self.variable_boundaries[i][1]
                
                delta1 = (y - lb) / (ub - lb)
                delta2 = (ub - y) / (ub - lb)
                
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    delta_q = 1.0 - val ** mut_pow
                
                y = y + delta_q * (ub - lb)
                y = min(max(y, lb), ub)
                
                child[i] = y
        
        return child
    
    def tournament_selection(self, population_with_objectives, ranks, crowding_distances, tournament_size=3):
        """锦标赛选择，增加锦标赛大小"""
        selected_indices = random.sample(range(len(population_with_objectives)), tournament_size)
        
        # 选择排名较高的个体
        best_idx = selected_indices[0]
        for idx in selected_indices[1:]:
            if ranks[idx] < ranks[best_idx] or (ranks[idx] == ranks[best_idx] and crowding_distances[idx] > crowding_distances[best_idx]):
                best_idx = idx
        
        return population_with_objectives[best_idx][:self.num_variables]
    
    def create_offspring(self, population, ranks, crowding_distances):
        """创建后代"""
        offspring = []
        
        # 精英保留策略 - 从第一前沿直接选择10%的最佳个体
        elite_count = max(2, int(self.population_size * 0.1))
        elite_indices = []
        
        # 找出第一前沿的精英个体
        for i in range(len(population)):
            if ranks[i] == 0:  # 第一前沿
                elite_indices.append(i)
        
        # 按拥挤度排序，选择最好的elite_count个
        if len(elite_indices) > elite_count:
            sorted_elites = sorted([(i, crowding_distances[i]) for i in elite_indices], 
                                  key=lambda x: x[1], reverse=True)
            elite_indices = [idx for idx, _ in sorted_elites[:elite_count]]
        
        # 直接添加精英个体到后代
        for idx in elite_indices:
            offspring.append(population[idx][:self.num_variables])
        
        # 创建剩余后代
        remaining = self.population_size - len(offspring)
        for _ in range(remaining // 2):
            # 锦标赛选择父母
            parent1 = self.tournament_selection(population, ranks, crowding_distances, tournament_size=3)
            parent2 = self.tournament_selection(population, ranks, crowding_distances, tournament_size=3)
            
            # 交叉
            child1, child2 = self.crossover(parent1, parent2, crossover_rate=0.95)
            
            # 自适应变异率 - 探索与开发的平衡
            if len(self.convergence_history) > 5:
                recent_conv = np.mean(self.convergence_history[-5:])
                # 如果收敛趋于稳定，增加变异率促进探索
                mutation_rate = 0.2 if recent_conv < 0.01 else 0.1
            else:
                mutation_rate = 0.1
            
            # 变异
            child1 = self.mutation(child1, mutation_rate=mutation_rate)
            child2 = self.mutation(child2, mutation_rate=mutation_rate)
            
            offspring.append(child1)
            offspring.append(child2)
        
        return np.array(offspring)
    
    def get_objective_values(self, population_with_objectives):
        """获取种群的目标函数值"""
        objective_values = []
        for individual in population_with_objectives:
            values = individual[self.num_variables:self.num_variables+self.objectives]
            # 由于我们是最小化问题，目标函数值是负的，所以取反
            objective_values.append([-v for v in values])
        return np.array(objective_values)
    
    def optimize(self, env):
        """运行NSGA-II优化"""
        # 初始化种群
        population = self.initialize_population()
        
        all_fronts = []
        best_secrecy_rate = 0
        best_energy_efficiency = 0
        
        for generation in tqdm(range(self.max_generations), desc="NSGA-II优化"):
            # 评估种群
            population_with_objectives = []
            for individual in population:
                objectives = self.calculate_objectives(individual, env)
                population_with_objectives.append(np.append(individual, objectives))
            
            population_with_objectives = np.array(population_with_objectives)
            
            # 记录当前最佳解
            objective_values = self.get_objective_values(population_with_objectives)
            current_secrecy_rates = objective_values[:, 0]
            current_energy_efficiencies = objective_values[:, 1]
            
            max_secrecy_rate = np.max(current_secrecy_rates)
            max_energy_efficiency = np.max(current_energy_efficiencies)
            
            if max_secrecy_rate > best_secrecy_rate:
                best_secrecy_rate = max_secrecy_rate
            
            if max_energy_efficiency > best_energy_efficiency:
                best_energy_efficiency = max_energy_efficiency
            
            self.objective_history['secrecy_rate'].append(max_secrecy_rate)
            self.objective_history['energy_efficiency'].append(max_energy_efficiency)
            
            # 非支配排序
            fronts, ranks = self.fast_non_dominated_sort(population_with_objectives)
            
            # 计算拥挤度距离
            crowding_distances = []
            for front in fronts:
                distances = self.calculate_crowding_distance(front, population_with_objectives)
                crowding_distances.append(distances)
            
            # 展平拥挤度距离列表
            flat_distances = []
            for i, front in enumerate(fronts):
                for j, idx in enumerate(front):
                    flat_distances.append(crowding_distances[i][j])
            
            # 创建后代
            offspring = self.create_offspring(population_with_objectives, ranks, flat_distances)
            
            # 评估后代
            offspring_with_objectives = []
            for individual in offspring:
                objectives = self.calculate_objectives(individual, env)
                offspring_with_objectives.append(np.append(individual, objectives))
            
            offspring_with_objectives = np.array(offspring_with_objectives)
            
            # 合并父代和子代
            combined = np.vstack((population_with_objectives, offspring_with_objectives))
            
            # 对合并种群进行非支配排序
            combined_fronts, combined_ranks = self.fast_non_dominated_sort(combined)
            
            # 计算拥挤度距离
            combined_crowding_distances = []
            for front in combined_fronts:
                distances = self.calculate_crowding_distance(front, combined)
                combined_crowding_distances.append(distances)
            
            # 选择下一代种群
            population = self.selection(combined, combined_fronts, combined_crowding_distances)
            
            # 保存第一个前沿的解
            front1_indices = combined_fronts[0]
            front1_solutions = [combined[i] for i in front1_indices]
            
            # 记录历史数据
            self.front_history.append(front1_solutions)
            
            # 收敛性度量 - 计算相邻两代帕累托前沿的平均距离
            if generation > 0:
                prev_front = np.array([sol[self.num_variables:self.num_variables+self.objectives] for sol in self.front_history[-2]])
                current_front = np.array([sol[self.num_variables:self.num_variables+self.objectives] for sol in front1_solutions])
                
                if len(prev_front) > 0 and len(current_front) > 0:
                    # 计算相邻两代前沿的平均距离
                    total_dist = 0
                    count = 0
                    for p1 in prev_front:
                        min_dist = float('inf')
                        for p2 in current_front:
                            dist = np.linalg.norm(p1 - p2)
                            min_dist = min(min_dist, dist)
                        total_dist += min_dist
                        count += 1
                    
                    if count > 0:
                        avg_dist = total_dist / count
                        self.convergence_history.append(avg_dist)
            
            # 存储全部种群历史
            self.population_history.append(population)
        
        # 返回帕累托前沿
        population_with_objectives = []
        for individual in population:
            objectives = self.calculate_objectives(individual, env)
            population_with_objectives.append(np.append(individual, objectives))
        
        fronts, _ = self.fast_non_dominated_sort(np.array(population_with_objectives))
        pareto_front = [population_with_objectives[i] for i in fronts[0]]
        
        return pareto_front, self.objective_history
    
    def plot_pareto_front(self, pareto_front, save_path=None):
        """绘制帕累托前沿"""
        plt.figure(figsize=(10, 6))
        
        # 提取目标函数值（注意：我们取反，因为原来是最小化问题）
        x = [-sol[self.num_variables] for sol in pareto_front]  # 保密速率
        y = [-sol[self.num_variables + 1] for sol in pareto_front]  # 能量效率
        
        plt.scatter(x, y, c='blue', s=50)
        plt.title('NSGA-II Pareto Front')
        plt.xlabel('Secrecy Rate (bps/Hz)')
        plt.ylabel('Energy Efficiency')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig('nsga2_pareto_front.png')
        
        plt.close()
    
    def plot_objective_history(self, save_dir=None):
        """绘制目标函数历史变化"""
        generations = range(len(self.objective_history['secrecy_rate']))
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(generations, self.objective_history['secrecy_rate'], 'b-')
        plt.title('Best Secrecy Rate vs Generation')
        plt.xlabel('Generation')
        plt.ylabel('Secrecy Rate (bps/Hz)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(generations, self.objective_history['energy_efficiency'], 'g-')
        plt.title('Best Energy Efficiency vs Generation')
        plt.xlabel('Generation')
        plt.ylabel('Energy Efficiency')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f'{save_dir}/nsga2_objective_history.png')
        else:
            plt.savefig('nsga2_objective_history.png')
        
        plt.close()
    
    def plot_convergence(self, save_dir=None):
        """绘制收敛历史"""
        if len(self.convergence_history) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.convergence_history) + 1), self.convergence_history)
            plt.title('NSGA-II Convergence History')
            plt.xlabel('Generation')
            plt.ylabel('Average Distance Between Fronts')
            plt.grid(True)
            
            if save_dir:
                plt.savefig(f'{save_dir}/nsga2_convergence.png')
            else:
                plt.savefig('nsga2_convergence.png')
            
            plt.close()
    
    def save_results(self, pareto_front, objective_history, save_dir=None):
        """保存优化结果"""
        if save_dir is None:
            save_dir = 'results/nsga2'
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存帕累托前沿
        pareto_data = []
        for solution in pareto_front:
            pareto_data.append({
                'variables': solution[:self.num_variables].tolist(),
                'objectives': {
                    'secrecy_rate': -solution[self.num_variables],
                    'energy_efficiency': -solution[self.num_variables + 1]
                }
            })
        
        with open(f'{save_dir}/pareto_front.json', 'w') as f:
            json.dump(pareto_data, f, indent=4)
        
        # 保存目标函数历史
        with open(f'{save_dir}/objective_history.json', 'w') as f:
            json.dump(objective_history, f, indent=4)
        
        # 保存收敛历史
        with open(f'{save_dir}/convergence_history.json', 'w') as f:
            json.dump(self.convergence_history, f, indent=4)
        
        # 计算一些统计指标
        secrecy_rates = [-sol[self.num_variables] for sol in pareto_front]
        energy_efficiencies = [-sol[self.num_variables + 1] for sol in pareto_front]
        
        statistics = {
            'max_secrecy_rate': max(secrecy_rates) if secrecy_rates else 0,
            'max_energy_efficiency': max(energy_efficiencies) if energy_efficiencies else 0,
            'min_secrecy_rate': min(secrecy_rates) if secrecy_rates else 0,
            'min_energy_efficiency': min(energy_efficiencies) if energy_efficiencies else 0,
            'avg_secrecy_rate': sum(secrecy_rates) / len(secrecy_rates) if secrecy_rates else 0,
            'avg_energy_efficiency': sum(energy_efficiencies) / len(energy_efficiencies) if energy_efficiencies else 0,
            'pareto_front_size': len(pareto_front),
            'total_generations': self.max_generations
        }
        
        with open(f'{save_dir}/statistics.json', 'w') as f:
            json.dump(statistics, f, indent=4)
        
        return statistics 