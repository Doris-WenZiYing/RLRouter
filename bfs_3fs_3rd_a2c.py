import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

@dataclass
class Task:
    """任務資料結構 - 明確定義輸入需求"""
    task_id: int
    source_host: str           # 起始host (如: 'h1')
    target_roadm: str          # 目標ROADM (如: 'r7') 
    required_compute: float    # 所需算力 (如: 300 FLOPS)
    data_size: float          # 數據大小 Di
    memory_requirement: float  # 記憶體需求 Mi
    max_delay: float          # 最大延遲限制 Tmax
    priority: int = 1         # 任務優先級 (1=低, 2=中, 3=高)

@dataclass
class ServiceFarm:
    """服務農場資料結構"""
    sf_id: str
    location_roadm: str       # 所在ROADM位置
    compute_power: float      # 可提供算力
    memory_capacity: float    # 記憶體容量  
    power_consumption: float  # 功耗
    threads: int
    is_available: bool = True

@dataclass
class AllocationResult:
    """分配結果結構"""
    success: bool
    selected_path: List[str]        # 路由路徑 [h1 -> r1 -> r2 -> r7]
    selected_sfs: List[ServiceFarm] # 選中的SF列表
    total_compute: float            # 總算力
    total_memory: float            # 總記憶體
    total_power: float             # 總功耗
    network_delay: float           # 網路延遲
    processing_time: float         # 處理時間
    total_delay: float            # 總延遲
    wavelengths_used: Dict        # 使用的波長分配


class TaskGenerator:
    """任務生成器 - 產生不同難度的任務組合"""
    
    def __init__(self, topology):
        self.topology = topology
        self.hosts = self._extract_hosts()
        self.roadms = list(topology.keys())
    
    def _extract_hosts(self):
        """提取所有host"""
        # TODO: 從topology中提取所有ConnectedHosts
        pass
    
    def generate_single_task(self, task_id, target_roadm, required_compute):
        """生成單一任務 - 明確指定目標和需求"""
        # TODO: 隨機選擇source_host (但不能是target_roadm所在的host)
        # TODO: 根據required_compute計算合理的data_size和memory_requirement
        # TODO: 設定適當的max_delay (基於網路大小和計算需求)
        # TODO: 返回Task物件
        pass
    
    def generate_task_batch(self, specifications):
        """根據規格批量生成任務
        specifications = [
            ('r7', 300),  # 目標r7, 需要300算力
            ('r3', 150),  # 目標r3, 需要150算力
            ('r5', 500),  # 目標r5, 需要500算力
        ]
        """
        # TODO: 根據specifications生成對應的任務列表
        # TODO: 確保任務之間有適當的多樣性
        # TODO: 返回Task列表
        pass
    
    def generate_difficulty_levels(self):
        """生成不同難度的任務集合"""
        # TODO: 簡單任務：就近分配，充足資源
        # TODO: 中等任務：需要跨ROADM，資源適中
        # TODO: 困難任務：遠距離，資源緊張，時間限制嚴格
        pass


class RMSAEnvironment:
    """RMSA環境 - 明確的輸入輸出介面"""
    
    def __init__(self, topology, roadm_distances, wavelength_capacity, sf_list):
        self.topology = topology
        self.roadm_distances = roadm_distances
        self.initial_wavelength_capacity = wavelength_capacity.copy()
        self.initial_sf_list = sf_list.copy()
        
        # 當前狀態
        self.current_wavelength_capacity = None
        self.current_sf_list = None
        self.current_task = None
        self.task_queue = deque()
        
        # 結果追蹤
        self.successful_allocations = []
        self.failed_tasks = []
        
        # TODO: 計算狀態空間維度
        self.state_dim = self._calculate_state_dimension()
        self.action_dim = 3  # 0=delay優先, 1=power優先, 2=balanced
    
    def reset(self, task_list: List[Task]):
        """重置環境並載入新任務列表"""
        # TODO: 恢復所有SF到可用狀態
        # TODO: 恢復所有波長到初始狀態  
        # TODO: 載入任務佇列，設定第一個當前任務
        # TODO: 清空結果追蹤列表
        # TODO: 返回初始狀態向量
        pass
    
    def _get_state(self) -> np.ndarray:
        """構建狀態向量 - 明確的狀態表示"""
        if not self.current_task:
            return np.zeros(self.state_dim)
        
        state_components = []
        
        # 1. 當前任務特徵 (正規化)
        task_features = [
            self.current_task.required_compute / 1000,  # 正規化到0-1
            self.current_task.data_size / 1e8,
            self.current_task.memory_requirement / 200,
            self.current_task.max_delay / 10,
            self.current_task.priority / 3
        ]
        state_components.extend(task_features)
        
        # 2. 從source到每個ROADM的最短路徑延遲
        source_roadm = self._find_roadm_for_host(self.current_task.source_host)
        # TODO: 使用BFS計算到所有ROADM的延遲
        # TODO: 正規化延遲值並加入state_components
        
        # 3. 目標ROADM的可用資源狀況
        target_roadm = self.current_task.target_roadm
        # TODO: 計算目標ROADM及其鄰居的SF可用性
        # TODO: 計算可用總算力、記憶體、功耗
        
        # 4. 每個SF的詳細狀態
        # TODO: 對每個SF: [is_available, compute_ratio, memory_ratio, power_level]
        
        # 5. 網路擁塞狀態
        # TODO: 計算每條link的波長使用率
        
        return np.array(state_components, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """執行動作 - 返回明確的分配結果"""
        if not self.current_task:
            return None, 0, True, {'error': 'No current task'}
        
        # 執行任務分配
        allocation_result = self._allocate_task_with_action(action)
        
        # 計算獎勵
        reward = self._calculate_reward(allocation_result, action)
        
        # 更新環境狀態
        self._update_environment(allocation_result)
        
        # 移動到下一個任務
        next_state, done = self._move_to_next_task()
        
        # 構建資訊字典
        info = {
            'allocation_result': allocation_result,
            'current_task_id': self.current_task.task_id if self.current_task else None,
            'success_count': len(self.successful_allocations),
            'failure_count': len(self.failed_tasks),
            'action_taken': action
        }
        
        return next_state, reward, done, info
    
    def _allocate_task_with_action(self, action: int) -> AllocationResult:
        """根據動作執行任務分配 - 明確的分配邏輯"""
        current_task = self.current_task
        
        # Step 1: 找到起始ROADM
        source_roadm = self._find_roadm_for_host(current_task.source_host)
        if not source_roadm:
            return AllocationResult(success=False, selected_path=[], selected_sfs=[], 
                                  total_compute=0, total_memory=0, total_power=0,
                                  network_delay=0, processing_time=0, total_delay=0,
                                  wavelengths_used={})
        
        # Step 2: 計算到目標ROADM的路徑和延遲
        # TODO: 使用BFS計算最短路徑 source_roadm -> target_roadm
        # TODO: 如果無法到達target_roadm，返回失敗結果
        
        # Step 3: 在目標ROADM附近尋找可用SF (擴展搜索範圍)
        # TODO: 先搜索target_roadm的SF
        # TODO: 如果資源不足，擴展到鄰居ROADM  
        # TODO: 構建候選SF列表 [(sf, additional_delay)]
        
        # Step 4: 根據action選擇排序策略
        if action == 0:  # delay優先
            # TODO: 按 (path_delay + additional_delay) 排序
            pass
        elif action == 1:  # power優先  
            # TODO: 按 power_consumption 排序
            pass
        else:  # action == 2, balanced
            # TODO: 按加權組合 α*delay + β*power 排序
            pass
        
        # Step 5: 貪婪選擇SF直到滿足需求
        selected_sfs = []
        total_compute = 0
        total_memory = 0
        total_power = 0
        
        # TODO: 遍歷候選SF，累加資源直到滿足required_compute和memory_requirement
        # TODO: 驗證時間約束：network_delay + processing_time <= max_delay
        # TODO: 分配波長資源
        
        # Step 6: 構建並返回結果
        # TODO: 計算最終的路徑、延遲、功耗等指標
        # TODO: 返回完整的AllocationResult
        pass
    
    def _calculate_reward(self, allocation_result: AllocationResult, action: int) -> float:
        """計算獎勵 - 多目標優化獎勵函數"""
        if not allocation_result.success:
            return -50.0  # 失敗重懲罰
        
        reward = 0.0
        
        # 1. 基礎成功獎勵
        reward += 100.0
        
        # 2. 效率獎勵
        # TODO: 時間效率：(max_delay - total_delay) / max_delay * 50
        # TODO: 資源效率：避免過度配置的懲罰
        # TODO: 功耗效率：低功耗方案獎勵
        
        # 3. 動作一致性獎勵
        # TODO: 如果action=0且實際選擇了最短路徑：+20
        # TODO: 如果action=1且實際選擇了低功耗方案：+20  
        # TODO: 如果action=2且達到了平衡：+15
        
        # 4. 任務優先級獎勵
        # TODO: 高優先級任務成功完成：額外獎勵
        
        return reward
    
    # === BFS和路由相關方法 ===
    def _find_roadm_for_host(self, host_id: str) -> Optional[str]:
        """找到host連接的ROADM"""
        # TODO: 從原程式碼移植
        pass
    
    def bfs_shortest_path(self, start_roadm: str, target_roadm: str) -> Tuple[List[str], float]:
        """使用BFS找到最短路徑和延遲"""
        # TODO: 基於原有bfs_delay_expansion修改
        # TODO: 返回路徑列表和總延遲
        pass
    
    def find_available_sfs_near_roadm(self, roadm: str, search_radius: int = 2) -> List[Tuple[ServiceFarm, float]]:
        """找到ROADM附近的可用SF"""
        # TODO: 在給定半徑內搜索可用SF
        # TODO: 計算到每個SF的額外延遲
        # TODO: 返回 [(sf, additional_delay)] 列表
        pass


class PolicyNetwork(nn.Module):
    """策略網路 - 輸出3種策略的機率"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super(PolicyNetwork, self).__init__()
        # TODO: 設計網路架構
        # Input: state_dim
        # Hidden: 3層，每層hidden_dim個神經元，ReLU激活
        # Output: 3 (delay優先, power優先, balanced)，Softmax激活
        pass
    
    def forward(self, state):
        # TODO: 前向傳播，輸出動作機率 [P(delay), P(power), P(balanced)]
        pass


class ValueNetwork(nn.Module):
    """價值網路 - 評估狀態價值"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super(ValueNetwork, self).__init__()
        # TODO: 設計網路架構，輸出單一價值
        pass
    
    def forward(self, state):
        # TODO: 前向傳播，輸出狀態價值V(s)
        pass


class A2CAgent:
    """強化學習智能體"""
    
    def __init__(self, state_dim: int, lr: float = 0.001, gamma: float = 0.95):
        self.gamma = gamma
        
        # TODO: 初始化策略網路和價值網路
        # TODO: 設定優化器
        # TODO: 初始化經驗緩衝區
        
        # 行為統計
        self.action_counts = [0, 0, 0]  # [delay, power, balanced]
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """選擇動作並返回動作和機率"""
        # TODO: 狀態轉tensor
        # TODO: 通過策略網路獲得機率分布
        # TODO: training=True: 按機率採樣, training=False: 選最大機率
        # TODO: 更新action_counts統計
        # TODO: 返回(action, probability)
        pass
    
    def get_action_statistics(self) -> Dict[str, float]:
        """獲取動作選擇統計"""
        total = sum(self.action_counts)
        if total == 0:
            return {'delay': 0, 'power': 0, 'balanced': 0}
        
        return {
            'delay': self.action_counts[0] / total,
            'power': self.action_counts[1] / total, 
            'balanced': self.action_counts[2] / total
        }


class PerformanceAnalyzer:
    """性能分析器 - 詳細分析RL vs BFS性能"""
    
    def __init__(self):
        self.rl_results = []
        self.bfs_results = []
    
    def record_rl_episode(self, episode_info):
        """記錄RL episode結果"""
        # TODO: 記錄success_rate, avg_delay, avg_power, action_distribution
        pass
    
    def record_bfs_baseline(self, bfs_results):
        """記錄BFS基準結果"""
        # TODO: 記錄BFS方法的性能指標
        pass
    
    def generate_comparison_report(self) -> Dict:
        """生成詳細比較報告"""
        report = {}
        
        # TODO: 成功率比較
        # TODO: 平均延遲比較
        # TODO: 功耗效率比較
        # TODO: 不同任務難度下的性能對比
        # TODO: 學習收斂曲線分析
        
        return report
    
    def plot_training_curves(self):
        """繪製訓練曲線"""
        # TODO: 成功率隨episode變化
        # TODO: 平均獎勵變化
        # TODO: 動作選擇分布變化
        pass


class DeepRMSATrainer:
    """訓練器 - 明確的訓練流程"""
    
    def __init__(self, env: RMSAEnvironment, agent: A2CAgent):
        self.env = env
        self.agent = agent
        self.analyzer = PerformanceAnalyzer()
    
    def train_with_curriculum(self, task_specifications_by_difficulty):
        """課程學習 - 從簡單到困難的任務"""
        # task_specifications_by_difficulty = {
        #     'easy': [('r2', 100), ('r3', 150)],      # 近距離，低算力需求
        #     'medium': [('r5', 300), ('r4', 250)],    # 中距離，中等需求  
        #     'hard': [('r7', 500), ('r6', 450)]       # 遠距離，高算力需求
        # }
        
        # TODO: 階段1：在easy任務上訓練200 episodes
        # TODO: 階段2：在medium任務上訓練300 episodes
        # TODO: 階段3：在hard任務上訓練500 episodes
        # TODO: 每個階段結束評估性能並調整學習率
        pass
    
    def evaluate_on_test_set(self, test_specifications):
        """在測試集上評估"""
        # TODO: 生成測試任務
        # TODO: 使用貪婪策略執行
        # TODO: 記錄詳細性能指標
        # TODO: 與BFS baseline比較
        pass
    
    def run_ablation_study(self):
        """消融實驗 - 分析各組件貢獻"""
        # TODO: 測試不同獎勵函數設計
        # TODO: 測試不同動作空間設計  
        # TODO: 測試不同網路架構
        pass


# ===== 使用範例 =====

def create_sample_tasks():
    """創建範例任務規格"""
    specifications = {
        'training': [
            ('r3', 200),   # 目標r3，需要200算力
            ('r5', 350),   # 目標r5，需要350算力
            ('r2', 150),   # 目標r2，需要150算力
            ('r4', 300),   # 目標r4，需要300算力
        ],
        'testing': [
            ('r7', 400),   # 挑戰性任務
            ('r6', 500),   # 高算力需求
            ('r1', 100),   # 簡單任務
        ]
    }
    return specifications

def main():
    """主程式流程"""
    
    print("=== DeepRMSA 強化學習訓練 ===")
    
    # TODO: 1. 從現有程式碼載入topology, SF_list等配置
    # TODO: 2. 創建環境和任務生成器  
    # TODO: 3. 生成訓練和測試任務集
    # TODO: 4. 初始化A2C智能體
    # TODO: 5. 執行課程學習訓練
    # TODO: 6. 在測試集上評估
    # TODO: 7. 與BFS baseline比較
    # TODO: 8. 生成詳細報告和可視化結果
    
    task_specs = create_sample_tasks()
    print(f"任務規格範例: {task_specs}")


if __name__ == "__main__":
    main()