import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class NetworkGraph:
    V: List[str]  # 節點集合 (ROADM, 資料中心)
    E: List[Tuple[str, str]]  # 光纖路徑集合
    A: List[int]  # 波長集合 A={1,2,...,|A|}
    
    def __post_init__(self):
        self.num_nodes = len(self.V)
        self.num_edges = len(self.E)
        self.num_wavelengths = len(self.A)
        self.C_v = 100  # 單一波長最大容量
        self.fiber_usage = {}  # key: (edge, wavelength, time), value: 0 or 1
        
    def f_ea(self, e: Tuple[str, str], wavelength: int, t: int) -> int:
        return self.fiber_usage.get((e, wavelength, t), 0)

@dataclass
class Request:
    request_id: int
    o_i: str  # 來源節點
    d_i: str  # 目的節點
    b_i: float  # 需求頻寬
    tau_i: float  # 服務時間
    F_i: float  # 算力需求
    M_i: float  # 記憶體需求
    p_i: int  # 任務優先級
    
    def required_wavelengths(self, C_v: float) -> int:
        return int(np.ceil(self.b_i / C_v))


class WavelengthAvailability:
    
    @staticmethod
    def calculate_n_r(b_r: float, c_v: float) -> int:
        return int(np.ceil(b_r / c_v))
    
    @staticmethod
    def check_wavelength_set_available(path_k: List[Tuple], S: Set[int], f_usage: Dict) -> float:
        for edge in path_k:
            for wavelength in S:
                if f_usage.get((edge, wavelength), 0) == 1:
                    return 0
        return 1


@dataclass
class PathFeatures:
    """路徑屬性特徵 P_k"""
    z1_k: float
    z2_k: float
    z3_k: float
    z4_k: float
    z5_k: float
    
    @staticmethod
    def compute_z1(path_k: List, wavelength_set: Set, f_usage: Dict) -> float:
        count = 0
        for wavelength in wavelength_set:
            available = True
            for edge in path_k:
                if f_usage.get((edge, wavelength), 0) == 1:
                    available = False
                    break
            if available:
                count += 1
        return float(count)
    
    @staticmethod
    def compute_z2(path_k: List, wavelength_set: Set, f_usage: Dict) -> float:
        max_continuous = 0
        current_continuous = 0
        
        for wavelength in sorted(wavelength_set):
            available = True
            for edge in path_k:
                if f_usage.get((edge, wavelength), 0) == 1:
                    available = False
                    break
            
            if available:
                current_continuous += 1
                max_continuous = max(max_continuous, current_continuous)
            else:
                current_continuous = 0
                
        return float(max_continuous)
    
    @staticmethod
    def compute_z3(z2_value: float, n_r: int) -> float:
        return 1.0 if z2_value >= n_r else 0.0
    
    @staticmethod
    def compute_z4(path_delay: float) -> float:
        return path_delay
    
    @staticmethod
    def compute_z5(C_avail: float, F_i: float, SF_set: Set, path_k: List) -> float:
        max_ratio = 0.0
        for sf in SF_set:
            if sf in path_k:
                ratio = C_avail / F_i if F_i > 0 else 0.0
                max_ratio = max(max_ratio, ratio)
        return max_ratio


@dataclass
class State:
    u_t: Request
    path_features: Dict[int, PathFeatures]  # 每條候選路徑k的特徵
    
    def get_state_vector(self) -> np.ndarray:
        # TODO: 實現狀態向量化
        pass

@dataclass
class ServiceFarm:
    sf_id: str  # 特定sf
    
    def __init__(self, sf_id: str, C_max: float, M_max: float):
        self.sf_id = sf_id
        self.C_max_sf = C_max
        self.M_max_sf = M_max
        self.C_avail_sf = C_max
        self.M_avail_sf = M_max
        
    def compute_q_i(self, t: int) -> Dict[str, float]:
        # 算力比
        C_tilde = self.C_avail_sf / self.C_max_sf if self.C_max_sf > 0 else 0.0
        
        # 記憶體比例
        M_tilde = self.M_avail_sf / self.M_max_sf if self.M_max_sf > 0 else 0.0
        
        # 功耗
        W_tilde = 0.0  # TODO: 根據實際功耗模型計算
        
        return {
            'avail': 1.0 if self.C_avail_sf > 0 and self.M_avail_sf > 0 else 0.0,
            'C_ratio': C_tilde,
            'M_ratio': M_tilde,
            'W_power': W_tilde
        }


@dataclass
class Action:
    k_i: int  #候選路徑索引
    sf_i: str  # 在該路徑上的節點中選一個
    wavelength_start: int
    wavelength_end: int
    
    @property
    def contiguous_wavelength_set(self) -> Set[int]:
        return set(range(self.wavelength_start, self.wavelength_end + 1))
    
    def execute_action(self, network: NetworkGraph, request: Request, sf: ServiceFarm):
        """執行動作，分配資源"""
        # TODO: 實現動作執行邏輯
        # 1. 在路徑k_i上分配波長集合S_k_i
        # 2. 在sf_i上分配算力F_i和記憶體M_i
        pass


class RewardCalculator:
    def __init__(self, 
                 R_succ: float = 100.0,
                 R_block: float = 50.0,
                 w_d: float = 1.0,
                 w_p: float = 1.0,
                 w_f: float = 1.0,
                 w_pri: float = 1.0,
                 upsilon: float = 0.5):
        """
        初始化權重參數
        
        Args:
            R_succ: 成功之基礎獎勵 (例如 1 或 100)
            R_block: 阻斷大懲罰 (例如 50 或 100)
            w_d: 延遲權重
            w_p: 功耗權重
            w_f: 碎片化權重
            w_pri: 優先級權重
            upsilon: 懲罰係數
        """
        self.R_succ = R_succ
        self.R_block = R_block
        self.w_d = w_d
        self.w_p = w_p
        self.w_f = w_f
        self.w_pri = w_pri
        self.upsilon = upsilon
        
    def compute_success_indicator(self, success: bool, 
                                  request: Request,
                                  sf: ServiceFarm,
                                  tau_i: float) -> int:
        if not success:
            return 0
            
        # 檢查SF資源是否足夠
        if sf.C_avail_sf >= request.F_i and sf.M_avail_sf >= request.M_i:
            return 1
        return 0
    
    def compute_delay_efficiency(self, tau_i: float, total_delay: float) -> float:
        if total_delay > tau_i:
            return 0.0
        return (tau_i - total_delay) / tau_i if tau_i > 0 else 0.0
    
    def compute_power_efficiency(self, P_used: float, P_max: float) -> float:
        return 1.0 - (P_used / P_max) if P_max > 0 else 0.0
    
    def compute_fragmentation_efficiency(self, 
                                        phi_t: float, 
                                        phi_t_minus: float,
                                        delta_phi: float) -> float:
        n_frag = np.exp(-self.upsilon * max(0.0, delta_phi))
        return n_frag
    
    def compute_priority_factor(self, p_i: int, alpha: float = 1.0) -> float:
        return alpha * p_i
    
    def compute_reward(self,
                      success: bool,
                      request: Request,
                      sf: ServiceFarm,
                      tau_i: float,
                      total_delay: float,
                      P_used: float,
                      P_max: float,
                      phi_t: float,
                      phi_t_minus: float) -> float:
        I_succ = self.compute_success_indicator(success, request, sf, tau_i)
        
        if I_succ == 1:
            n_delay = self.compute_delay_efficiency(tau_i, total_delay)
            n_power = self.compute_power_efficiency(P_used, P_max)
            delta_phi = phi_t - phi_t_minus
            n_frag = self.compute_fragmentation_efficiency(phi_t, phi_t_minus, delta_phi)
            g_p = self.compute_priority_factor(request.p_i)
            
            reward = self.R_succ + \
                    self.w_d * n_delay + \
                    self.w_p * n_power + \
                    self.w_f * n_frag + \
                    self.w_pri * g_p
        else:
            reward = -self.R_block
            
        return reward


class BaselineCalculator:
    
    def __init__(self, gamma: float = 0.99, window_length: int = None):
        """
        Args:
            gamma: 折扣因子
            window_length: 滑窗長度W (如果用FLX)，None則用EP
        """
        self.gamma = gamma
        self.window_length = window_length
        self.reward_history = []
        
    def compute_sliding_window_return(self, t: int) -> float:
        if self.window_length is None:
            return 0.0
            
        G_t = 0.0
        for i in range(self.window_length):
            if t + i < len(self.reward_history):
                G_t += (self.gamma ** i) * self.reward_history[t + i]
        return G_t
    
    def compute_episode_return(self, t: int, episode_length: int) -> float:
        """
        N_t: Episode長度
        """
        G_t = 0.0
        for i in range(episode_length - t):
            if t + i < len(self.reward_history):
                G_t += (self.gamma ** i) * self.reward_history[t + i]
        return G_t
    
    def compute_advantage(self, reward: float, state_value: float, 
                         next_state_value: float) -> float:
        advantage = reward + self.gamma * next_state_value - state_value
        return advantage


class LossCalculator:
    
    @staticmethod
    def policy_loss(log_probs: torch.Tensor, 
                   advantages: torch.Tensor,
                   alpha: float = 0.1) -> torch.Tensor:
        # Policy gradient term
        policy_gradient = -(advantages * log_probs).mean()
        
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        
        loss = policy_gradient - alpha * entropy
        return loss
    
    @staticmethod
    def value_loss(predicted_values: torch.Tensor, 
                  target_values: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(predicted_values, target_values)
        return loss


class PolicyNetwork(nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            state: 狀態向量
            
        Returns:
            action_probs: 動作機率分布
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs


class ValueNetwork(nn.Module):
    """
    Baseline Value Network: V(s_t; Θ_v)
    輸入: 狀態 s_t
    輸出: 狀態價值估計
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            state: 狀態向量
            
        Returns:
            state_value: 狀態價值估計
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        state_value = self.fc3(x)
        return state_value


class OpticalNetworkAgent:
    """光網路資源分配 RL Agent"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-4):
        """
        初始化Agent
        
        Args:
            state_dim: 狀態空間維度
            action_dim: 動作空間維度
            learning_rate: 學習率
        """
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=learning_rate
        )
        
        self.reward_calculator = RewardCalculator()
        self.baseline_calculator = BaselineCalculator()
        self.loss_calculator = LossCalculator()
        
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        根據當前狀態選擇動作
        
        Args:
            state: 當前狀態
            
        Returns:
            action: 選擇的動作
            log_prob: 動作的log概率
        """
        # TODO: 實現動作選擇邏輯
        # 1. 將state轉換為tensor
        # 2. 通過policy network獲得動作概率分布
        # 3. 採樣動作
        # 4. 返回動作和log概率
        pass
    
    def update(self, states: List, actions: List, rewards: List, 
              next_states: List, dones: List):
        """
        更新policy和value network
        
        Args:
            states: 狀態列表
            actions: 動作列表
            rewards: 獎勵列表
            next_states: 下一狀態列表
            dones: 終止標誌列表
        """
        # TODO: 實現網路更新邏輯
        # 1. 計算advantages
        # 2. 計算policy loss
        # 3. 計算value loss
        # 4. 反向傳播更新參數
        pass
    
    def train_episode(self, env, max_steps: int = 1000):
        """
        訓練一個episode
        
        Args:
            env: 環境
            max_steps: 最大步數
        """
        # TODO: 實現episode訓練邏輯
        pass


class OpticalNetworkEnvironment:
    """光網路資源分配環境"""
    
    def __init__(self, network: NetworkGraph, service_farms: List[ServiceFarm]):
        """
        初始化環境
        
        Args:
            network: 網路拓撲
            service_farms: Service Farm列表
        """
        self.network = network
        self.service_farms = service_farms
        self.current_time = 0
        self.request_queue = []
        
    def reset(self) -> State:
        """
        重置環境
        
        Returns:
            initial_state: 初始狀態
        """
        # TODO: 實現環境重置邏輯
        pass
    
    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        """
        執行動作，環境狀態轉移
        
        Args:
            action: 動作
            
        Returns:
            next_state: 下一狀態
            reward: 獎勵
            done: 是否結束
            info: 額外信息
        """
        # TODO: 實現環境step邏輯
        # 1. 執行動作，分配資源
        # 2. 計算獎勵
        # 3. 更新狀態
        # 4. 檢查是否結束
        pass
    
    def generate_request(self) -> Request:
        """生成隨機請求"""
        # TODO: 實現請求生成邏輯
        pass


def main():
    """主程式"""
    
    # 1. 初始化網路拓撲
    nodes = ['ROADM1', 'ROADM2', 'DC1', 'DC2']
    edges = [('ROADM1', 'ROADM2'), ('ROADM2', 'DC1'), ('ROADM1', 'DC2')]
    wavelengths = list(range(1, 81))  # 80個波長
    network = NetworkGraph(V=nodes, E=edges, A=wavelengths)
    
    # 2. 初始化Service Farms
    service_farms = [
        ServiceFarm('SF1', C_max=1000.0, M_max=10000.0),
        ServiceFarm('SF2', C_max=1500.0, M_max=15000.0)
    ]
    
    # 3. 初始化環境
    env = OpticalNetworkEnvironment(network, service_farms)
    
    # 4. 初始化Agent
    state_dim = 100  # TODO: 根據實際狀態維度設定
    action_dim = 50  # TODO: 根據實際動作空間設定
    agent = OpticalNetworkAgent(state_dim, action_dim)
    
    # 5. 訓練
    num_episodes = 1000
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        # TODO: 實現訓練迴圈
        # agent.train_episode(env)
    
    print("訓練完成！")


if __name__ == "__main__":
    main()