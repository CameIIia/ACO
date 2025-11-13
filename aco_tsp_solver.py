import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================
# 1. ACO 算法核心类
# ==========================================
class AntColonyOptimization:
    # ------------------------------------------
    # 步骤A: 初始化
    # ------------------------------------------
    def __init__(self, distances, num_ants, num_iterations, decay, alpha=1.0, beta=1.0, q=100.0):
        """
        初始化蚁群算法
        :param distances: (numpy.ndarray) 城市间的距离矩阵
        :param num_ants: (int) 蚂蚁数量
        :param num_iterations: (int) 迭代次数
        :param decay: (float) 信息素蒸发率 (rho)
        :param alpha: (float) 信息素重要性因子
        :param beta: (float) 启发式信息重要性因子
        :param q: (float) 信息素强度常数
        """
        self.distances = distances
        self.num_cities = distances.shape[0]
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.q = q

        # 初始化信息素矩阵 (tau)
        self.pheromone = np.ones((self.num_cities, self.num_cities))
    
        # 初始化启发式信息矩阵 (eta = 1/distance)
        # 为避免除以零，将对角线设为无穷小值
        with np.errstate(divide='ignore'):
            self.heuristic = 1.0 / self.distances
        self.heuristic[np.isinf(self.heuristic)] = 0

        # 记录全局最优解
        self.best_path = None
        self.best_path_length = np.inf

    # ------------------------------------------
    # 步骤B: 算法主循环
    # ------------------------------------------
    def run(self):
        """主循环，执行ACO算法"""
        for iteration in range(self.num_iterations):
            # 1. 所有蚂蚁构建路径
            all_paths = self._construct_solutions(iteration)
            
            # 2. 更新信息素
            self._update_pheromone(all_paths)
        
            # 3. 记录当前迭代的最优解
            current_best_path = min(all_paths, key=lambda x: x[1])
            if current_best_path[1] < self.best_path_length:
                self.best_path = current_best_path[0]
                self.best_path_length = current_best_path[1]

            # --- 新增：为前5次迭代打印详细信息 ---
            if iteration < 3:
                print(f"--- 迭代 {iteration+1} 详细信息 ---")
                # 打印每只蚂蚁的路径和长度
                for i, (path, length) in enumerate(all_paths):
                    print(f"  蚂蚁 {i+1}: 路径 {path}, 长度 {length:.2f}")
                print(f"  本次迭代最优路径: {current_best_path[0]}, 长度: {current_best_path[1]:.2f}")
                print(f"  截至本次迭代，全局最优路径长度: {self.best_path_length:.2f}\n")
            else:
                if (iteration + 1) % 10 == 0:
                    print(f"迭代 {iteration+1}/{self.num_iterations}: "
                        f"当前最短路径长度: {current_best_path[1]:.2f}, "
                        f"全局最短路径长度: {self.best_path_length:.2f}")

        return self.best_path, self.best_path_length

    # ------------------------------------------
    # 步骤C: 构建路径 (所有蚂蚁)
    # ------------------------------------------
    def _construct_solutions(self, iteration):
        """所有蚂蚁构建路径"""
        all_paths = []
        for ant in range(self.num_ants):
            path = []
            visited = [False] * self.num_cities
        
            # 随机选择一个起始城市
            start_city = random.randint(0, self.num_cities - 1)
            path.append(start_city)
            visited[start_city] = True
        
            current_city = start_city

            # --- 修改：为前三次迭代打印详细计算过程 ---
            if iteration < 3 and ant == 0:
                print(f"\n" + "="*25 + f" 迭代 {iteration + 1} 中概率计算的详细示例 " + "="*25)
                print(f"此示例展示 蚂蚁 {ant+1} 从起始城市 {current_city} 选择下一个城市的计算过程。\n")

                # 获取所有未访问的城市
                unvisited_cities = [i for i, v in enumerate(visited) if not v]
                
                # 计算每个可能选择的概率分子 (τ^α * η^β)
                prob_components = {}
                total_prob_numerator = 0
                for next_c in unvisited_cities:
                    tau = self.pheromone[current_city, next_c]
                    eta = self.heuristic[current_city, next_c]
                    prob_numerator = (tau ** self.alpha) * (eta ** self.beta)
                    prob_components[next_c] = {
                        "tau": tau,
                        "eta": eta,
                        "prob_numerator": prob_numerator
                    }
                    total_prob_numerator += prob_numerator

                print(f"当前超参数: alpha (α) = {self.alpha}, beta (β) = {self.beta}\n")
                print("目标城市 | τ (信息素) | η (启发式) | (τ^α)*(η^β) | 归一化概率 P")
                print("----------|-------------|-------------|---------------|--------------")

                # 打印每个城市的计算详情
                for city, comps in sorted(prob_components.items()):
                    probability = comps["prob_numerator"] / total_prob_numerator if total_prob_numerator > 0 else 0
                    print(f"    {city:<5} |    {comps['tau']:.4f}   |    {comps['eta']:.4f}   |   {comps['prob_numerator']:.8f}  |    {probability:.4f}")

                print("\n" + "="*70 + "\n")
            # --- 示例代码结束 ---
        
            # 逐步为路径选择下一个城市
            while len(path) < self.num_cities:
                next_city = self._select_next_city(current_city, visited)
                path.append(next_city)
                visited[next_city] = True
                current_city = next_city
        
            # 添加回到起点的路径，形成闭环
            path.append(start_city)
            path_length = self._calculate_path_length(path)
            all_paths.append((path, path_length))
        return all_paths

    # ------------------------------------------
    # 步骤D: 根据概率选择下一个城市
    # ------------------------------------------
    def _select_next_city(self, current_city, visited):
        """根据概率公式选择下一个城市"""
        pheromone_slice = self.pheromone[current_city, :]
        heuristic_slice = self.heuristic[current_city, :]
    
        # 计算分子部分：(τ^α) * (η^β)
        probabilities = (pheromone_slice ** self.alpha) * (heuristic_slice ** self.beta)
    
        # 将已访问城市的概率设为0
        probabilities[visited] = 0
    
        # 如果所有概率都为0（可能发生在最后一步），则随机选择一个未访问的
        if probabilities.sum() == 0:
            unvisited_cities = [i for i, v in enumerate(visited) if not v]
            return random.choice(unvisited_cities)

        # 归一化，得到概率分布
        probabilities /= probabilities.sum()
    
        # 轮盘赌选择下一个城市
        next_city = np.random.choice(range(self.num_cities), p=probabilities)
        return next_city

    # ------------------------------------------
    # 步骤E: 更新信息素
    # ------------------------------------------
    def _update_pheromone(self, all_paths):
        """更新信息素矩阵"""
        # 1. 信息素蒸发
        self.pheromone *= (1 - self.decay)
    
        # 2. 信息素增强
        for path, length in all_paths:
            pheromone_deposit = self.q / length
            for i in range(self.num_cities):
                city1 = path[i]
                city2 = path[i+1]
                self.pheromone[city1, city2] += pheromone_deposit
                self.pheromone[city2, city1] += pheromone_deposit # 对称路径

    def _calculate_path_length(self, path):
        """计算路径总长度"""
        length = 0
        for i in range(len(path) - 1):
            length += self.distances[path[i], path[i+1]]
        return length

# ==========================================
# 2. 主程序运行
# ==========================================
if __name__ == '__main__':
    # ------------------------------------------
    # 步骤F: 定义问题 (城市坐标和距离)
    # ------------------------------------------
    # 定义城市坐标
    city_coords = np.array([
        [10, 20], [80, 15], [75, 90], [5, 85],
        [45, 50], [20, 40], [95, 55], [60, 75]
    ])
    num_cities = len(city_coords)

    # --- 新增：打印城市坐标并绘制初始分布图 ---
    print("="*50)
    print("城市坐标信息:")
    print(city_coords)
    print("="*50)

    plt.figure(figsize=(10, 8))
    plt.scatter(city_coords[:, 0], city_coords[:, 1], c='blue', s=100)
    for i, (x, y) in enumerate(city_coords):
        plt.text(x, y, str(i), fontsize=12, ha='right')
    plt.title('Initial City Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.savefig('city_distribution.png')
    print("\n已生成城市分布图: city_distribution.png\n")
    # --- 新增代码结束 ---

    # 计算城市间的距离矩阵
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dist = np.linalg.norm(city_coords[i] - city_coords[j])
            distances[i, j] = distances[j, i] = dist

    # ------------------------------------------
    # 步骤G: 设置ACO超参数
    # ------------------------------------------
    aco_params = {
        'num_ants': 10,
        'num_iterations': 100,
        'decay': 0.1,  # rho
        'alpha': 1.0,
        'beta': 2.0,
        'q': 100.0
    }

    # ------------------------------------------
    # 步骤H: 运行ACO并输出结果
    # ------------------------------------------
    # 实例化并运行ACO
    aco = AntColonyOptimization(distances, **aco_params)
    best_path, best_length = aco.run()

    print("\n" + "="*50)
    print(f"算法运行结束")
    print(f"找到的最优路径: {best_path}")
    print(f"最优路径长度: {best_length:.2f}")
    print("="*50)

    # ------------------------------------------
    # 步骤I: 可视化结果
    # ------------------------------------------
    plt.figure(figsize=(10, 8))
    # 绘制城市点
    plt.scatter(city_coords[:, 0], city_coords[:, 1], c='red', s=100)
    for i, (x, y) in enumerate(city_coords):
        plt.text(x, y, str(i), fontsize=12, ha='right')

    # 绘制最优路径
    for i in range(len(best_path) - 1):
        start_node = best_path[i]
        end_node = best_path[i+1]
        plt.plot([city_coords[start_node, 0], city_coords[end_node, 0]],
                 [city_coords[start_node, 1], city_coords[end_node, 1]],
                 'b-')
  
    plt.title(f'ACO for TSP\nBest Path Length: {best_length:.2f}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.savefig('aco_tsp_result.png')