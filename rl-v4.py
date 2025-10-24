"""
光網路資源分配的深度強化學習系統 - 修正版
修正了所有AI審查報告中指出的重大錯誤和次要問題

主要修正：
1. PolicyNetwork 現在返回 logits 而非 probabilities
2. select_action 正確實現 masking（對 logits 做 mask）
3. 修正延遲計算（z4）
4. 保存和使用 log_probs
5. 實現 GAE（Generalized Advantage Estimation）
6. 移除 ServiceFarm 的自動 refill
7. 確保 edge tuple 方向一致性
8. 連續波長查找前先排序
9. 固定隨機種子以便重現
10. 改善狀態向量儲存（直接存 numpy array）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import math
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 修正 1: NetworkGraph - 確保 edge 方向一致性
# ============================================================================

@dataclass
class NetworkGraph:
    """網路拓撲圖 - 修正版：確保 edge tuple 方向一致性"""
    V: List[str]
    E: List[Tuple[str, str]]
    A: List[int]
    
    def __post_init__(self):
        self.num_nodes = len(self.V)
        self.num_edges = len(self.E)
        self.num_wavelengths = len(self.A)
        self.C_v = 100.0  # 單位：Gbps，每個波長的容量
        
        # 使用標準化的 edge tuples
        self.fiber_usage = {}
        self.edge_distances = {}
        
        # 創建標準化的 edge set
        self.normalized_edges = set()
        for u, v in self.E:
            self.normalized_edges.add(self._normalize_edge(u, v))
    
    def _normalize_edge(self, u: str, v: str) -> Tuple[str, str]:
        """標準化 edge tuple，確保一致性（總是較小的節點在前）"""
        return (u, v) if u < v else (v, u)
    
    def is_wavelength_available(self, e: Tuple[str, str], wavelength: int) -> bool:
        """檢查波長是否可用（修正：標準化 edge）"""
        e_norm = self._normalize_edge(e[0], e[1])
        return self.fiber_usage.get((e_norm, wavelength), 0) == 0
    
    def allocate_wavelength(self, e: Tuple[str, str], wavelength: int):
        """分配波長（修正：標準化 edge）"""
        e_norm = self._normalize_edge(e[0], e[1])
        self.fiber_usage[(e_norm, wavelength)] = 1
    
    def release_wavelength(self, e: Tuple[str, str], wavelength: int):
        """釋放波長（修正：標準化 edge）"""
        e_norm = self._normalize_edge(e[0], e[1])
        self.fiber_usage[(e_norm, wavelength)] = 0


@dataclass
class Request:
    """服務請求"""
    request_id: int
    o_i: str  # 來源節點
    d_i: str  # 目的節點
    b_i: float  # 頻寬需求（Gbps）
    tau_i: float  # 截止時間（秒）
    F_i: float  # 計算需求（FLOPS）
    M_i: float  # 記憶體需求（GB）
    p_i: int  # 優先級
    arrival_time: float  # 到達時間
    
    def required_wavelengths(self, C_v: float) -> int:
        """計算需要的波長數量"""
        return int(np.ceil(self.b_i / C_v))
    
    @property
    def release_time(self) -> float:
        """資源釋放時間"""
        return self.arrival_time + self.tau_i


@dataclass
class PathFeatures:
    """路徑特徵"""
    z1_k: float  # 可用波長總數
    z2_k: float  # 最大連續波長數
    z3_k: float  # 是否滿足連續性約束
    z4_k: float  # 路徑延遲
    z5_k: float  # 資源可用性
    
    @staticmethod
    def compute_z1(path_k: List, wavelength_set: Set, network: 'NetworkGraph') -> float:
        """計算路徑上可用的波長總數"""
        count = 0
        for wavelength in wavelength_set:
            available = all(
                network.is_wavelength_available(edge, wavelength)
                for edge in path_k
            )
            if available:
                count += 1
        return float(count)
    
    @staticmethod
    def compute_z2(path_k: List, wavelength_set: Set, network: 'NetworkGraph') -> float:
        """計算最大連續可用波長數"""
        max_contiguous = 0
        current_contiguous = 0
        
        for wavelength in sorted(wavelength_set):
            available = all(
                network.is_wavelength_available(edge, wavelength)
                for edge in path_k
            )
            
            if available:
                current_contiguous += 1
                max_contiguous = max(max_contiguous, current_contiguous)
            else:
                current_contiguous = 0
        
        return float(max_contiguous)
    
    @staticmethod
    def compute_z3(z2_value: float, n_r: int) -> float:
        """檢查是否滿足連續波長約束"""
        return 1.0 if z2_value >= n_r else 0.0
    
    @staticmethod
    def compute_z4(path_delay: float) -> float:
        """路徑延遲"""
        return path_delay
    
    @staticmethod
    def compute_z5(C_avail: float, F_i: float) -> float:
        """計算資源可用性比例"""
        return C_avail / F_i if F_i > 0 else 0.0


# ============================================================================
# 修正 2: ServiceFarm - 移除自動 refill
# ============================================================================

@dataclass
class ServiceFarm:
    """服務農場 - 修正版：移除瞬間 refill，資源不足時返回 False"""
    sf_id: str
    
    def __init__(self, sf_id: str, C_max: float, M_max: float):
        self.sf_id = sf_id
        self.C_max_sf = C_max  # 最大計算資源
        self.M_max_sf = M_max  # 最大記憶體
        self.C_avail_sf = C_max  # 當前可用計算資源
        self.M_avail_sf = M_max  # 當前可用記憶體
        self.total_allocations = 0  # 總分配次數
    
    def can_allocate(self, compute: float, memory: float) -> bool:
        """檢查是否有足夠資源"""
        return self.C_avail_sf >= compute and self.M_avail_sf >= memory
    
    def allocate_resources(self, compute: float, memory: float) -> bool:
        """
        分配資源（修正：移除自動 refill）
        
        Returns:
            True 如果成功分配，False 如果資源不足
        """
        if not self.can_allocate(compute, memory):
            logger.debug(f"{self.sf_id}: 資源不足 "
                        f"(需要 C={compute:.2f}, M={memory:.2f}, "
                        f"可用 C={self.C_avail_sf:.2f}, M={self.M_avail_sf:.2f})")
            return False
        
        self.C_avail_sf -= compute
        self.M_avail_sf -= memory
        self.total_allocations += 1
        
        return True
    
    def release_resources(self, compute: float, memory: float):
        """釋放資源（可選功能，用於時間窗口結束時）"""
        self.C_avail_sf = min(self.C_avail_sf + compute, self.C_max_sf)
        self.M_avail_sf = min(self.M_avail_sf + memory, self.M_max_sf)
    
    def get_utilization(self) -> Dict[str, float]:
        """獲取資源利用率"""
        return {
            'compute': 1.0 - (self.C_avail_sf / self.C_max_sf),
            'memory': 1.0 - (self.M_avail_sf / self.M_max_sf)
        }


@dataclass
class WavelengthAssignment:
    """波長分配記錄"""
    edge: Tuple[str, str]
    wavelength: int
    release_time: float


@dataclass
class AllocatedResources:
    """已分配的資源記錄"""
    request_id: int
    wavelength_assignments: List[WavelengthAssignment]
    sf_allocations: Dict[str, Tuple[float, float]]
    path_edges: List[Tuple[str, str]]


@dataclass
class Action:
    """動作：選擇路徑、服務農場和波長"""
    k_i: int  # 候選索引
    sf_i: str  # 服務農場 ID
    wavelength_start: int  # 起始波長
    wavelength_end: int  # 結束波長
    path_edges: List[Tuple[str, str]] = None
    sf_allocations: Dict[str, Tuple[float, float]] = None
    
    @property
    def contiguous_wavelength_set(self) -> Set[int]:
        """獲取連續波長集合"""
        return set(range(self.wavelength_start, self.wavelength_end + 1))
    
    def execute_action(self,
                      network: NetworkGraph,
                      request: Request,
                      sfs: List[ServiceFarm],
                      current_time: float) -> Tuple[bool, Optional[AllocatedResources]]:
        """執行動作並分配資源"""
        wavelength_set = self.contiguous_wavelength_set
        
        # 檢查 sf_allocations 是否有效
        if self.sf_allocations is None or len(self.sf_allocations) == 0:
            return False, None
        
        # 計算處理時間
        total_compute = sum(compute for compute, _ in self.sf_allocations.values())
        if total_compute == 0:
            return False, None
        
        processing_time = request.F_i / total_compute
        
        # 驗證截止時間約束
        if processing_time > request.tau_i:
            logger.debug(f"請求 {request.request_id}: 超過截止時間")
            return False, None
        
        # 驗證波長可用性
        for edge in self.path_edges:
            for wavelength in wavelength_set:
                if not network.is_wavelength_available(edge, wavelength):
                    return False, None
        
        # 驗證服務農場資源
        sf_dict = {sf.sf_id: sf for sf in sfs}
        for sf_id, (compute, memory) in self.sf_allocations.items():
            if sf_id not in sf_dict:
                return False, None
            if not sf_dict[sf_id].can_allocate(compute, memory):
                return False, None
        
        # 分配波長（基於時間的分配）
        release_time = current_time + processing_time
        wavelength_assignments = []
        
        for edge in self.path_edges:
            for wavelength in wavelength_set:
                network.allocate_wavelength(edge, wavelength)
                wavelength_assignments.append(
                    WavelengthAssignment(edge, wavelength, release_time)
                )
        
        # 分配服務農場資源（消耗性資源）
        for sf_id, (compute, memory) in self.sf_allocations.items():
            success = sf_dict[sf_id].allocate_resources(compute, memory)
            if not success:
                # 回滾波長分配
                for assignment in wavelength_assignments:
                    network.release_wavelength(assignment.edge, assignment.wavelength)
                return False, None
        
        # 創建分配記錄
        allocation = AllocatedResources(
            request_id=request.request_id,
            wavelength_assignments=wavelength_assignments,
            sf_allocations=self.sf_allocations.copy(),
            path_edges=self.path_edges.copy()
        )
        
        logger.debug(f"請求 {request.request_id}: 成功分配")
        return True, allocation


class RewardCalculator:
    """獎勵計算器"""
    
    def __init__(self,
                 R_succ: float = 100.0,  # 成功獎勵
                 R_block: float = 50.0,  # 阻塞懲罰
                 w_d: float = 1.0,  # 延遲權重
                 w_p: float = 1.0,  # 功率權重
                 w_f: float = 1.0,  # 碎片化權重
                 w_pri: float = 1.0,  # 優先級權重
                 upsilon: float = 0.5):  # 碎片化懲罰係數
        self.R_succ = R_succ
        self.R_block = R_block
        self.w_d = w_d
        self.w_p = w_p
        self.w_f = w_f
        self.w_pri = w_pri
        self.upsilon = upsilon
    
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
        """計算分配決策的獎勵"""
        if not success:
            return -self.R_block
        
        # 延遲效率
        n_delay = max(0.0, (tau_i - total_delay) / tau_i) if tau_i > 0 else 0.0
        
        # 功率效率
        n_power = 1.0 - (P_used / P_max) if P_max > 0 else 0.0
        
        # 碎片化懲罰
        delta_phi = phi_t - phi_t_minus
        n_frag = np.exp(-self.upsilon * max(0.0, delta_phi))
        
        # 優先級獎勵
        g_p = request.p_i
        
        reward = (self.R_succ +
                 self.w_d * n_delay +
                 self.w_p * n_power +
                 self.w_f * n_frag +
                 self.w_pri * g_p)
        
        return reward


@dataclass
class CandidateInfo:
    """候選方案資訊"""
    candidate_id: int
    path_edges: List[Tuple[str, str]]
    path_nodes: List[str]
    sf_combination: List[ServiceFarm]
    sf_allocations: Dict[str, Tuple[float, float]]
    total_delay: float
    processing_time: float
    power_consumption: float
    memory_usage: float
    cost: float
    path_features: PathFeatures


# ============================================================================
# 修正 3: BFSCandidateGenerator - 修正延遲計算（z4）
# ============================================================================

class BFSCandidateGenerator:
    """BFS 候選生成器 - 修正版"""
    
    # 常數定義（修正：明確定義物理常數）
    LIGHT_SPEED = 2e5  # km/s，光在光纖中的速度
    NODE_DELAY_PER_HOP = 0.0001  # 秒，每跳的節點處理延遲
    
    def __init__(self,
                 network: NetworkGraph,
                 service_farms: List[ServiceFarm],
                 topology_map: Dict,
                 k_candidates: int = 5):
        self.network = network
        self.service_farms = service_farms
        self.topology_map = topology_map
        self.k_candidates = k_candidates
    
    def generate_candidates(self, request: Request) -> List[CandidateInfo]:
        """生成候選方案"""
        # 過濾可用的服務農場
        available_sfs = [sf for sf in self.service_farms
                        if sf.C_avail_sf > 0 and sf.M_avail_sf > 0]
        
        if not available_sfs:
            logger.warning(f"請求 {request.request_id}: 沒有可用的服務農場")
            return []
        
        # 計算時間約束
        compute_min = min(sf.C_avail_sf for sf in available_sfs)
        T_compute = request.F_i / compute_min if compute_min > 0 else float('inf')
        
        if request.tau_i <= T_compute:
            logger.debug(f"請求 {request.request_id}: 截止時間太緊")
            return []
        
        T_limit = request.tau_i - T_compute
        
        # 計算初始延遲
        B = 50 * 1e9  # bps，頻寬
        gOSNR = 20  # dB，光訊噪比
        T_trans = request.b_i / (B * math.log2(1 + gOSNR))
        T_node = self.NODE_DELAY_PER_HOP
        initial_delay = T_trans + T_node
        
        # BFS 延遲擴展
        delay_dict = self._bfs_delay_expansion(request.o_i, T_limit, initial_delay)
        
        # 生成候選方案
        candidates = []
        candidate_id = 0
        
        for target_node in self.network.V:
            if target_node == request.o_i:
                continue
            
            delay, prev = delay_dict.get(target_node, (float('inf'), None))
            if delay > T_limit or delay == float('inf'):
                continue
            
            # 獲取目標節點的服務農場
            sfs_at_node = self._get_sfs_at_node(target_node)
            if not sfs_at_node:
                continue
            
            # 貪婪資源分配
            result = self._greedy_combine_sfs(sfs_at_node, request.F_i, request.M_i)
            if result is None:
                continue
            
            selected_sfs, sf_allocations, processing_time, power, memory = result
            
            # 重建路徑
            path_nodes = self._reconstruct_path(request.o_i, target_node, delay_dict)
            if len(path_nodes) < 2:
                logger.warning(f"路徑重建失敗: {request.o_i} -> {target_node}")
                continue
                
            path_edges = [(path_nodes[i], path_nodes[i+1])
                         for i in range(len(path_nodes)-1)]
            
            # 計算成本
            alpha, beta = 1.0, 0.01
            cost = alpha * (delay + processing_time) + beta * power
            
            # 計算路徑特徵
            path_features = self._compute_path_features(path_edges, request)
            
            # 檢查波長可用性
            if path_features.z3_k == 0:
                continue
            
            candidate = CandidateInfo(
                candidate_id=candidate_id,
                path_edges=path_edges,
                path_nodes=path_nodes,
                sf_combination=selected_sfs,
                sf_allocations=sf_allocations,
                total_delay=delay,
                processing_time=processing_time,
                power_consumption=power,
                memory_usage=memory,
                cost=cost,
                path_features=path_features
            )
            
            candidates.append(candidate)
            candidate_id += 1
        
        # 按成本排序並返回前 K 個
        candidates.sort(key=lambda c: c.cost)
        return candidates[:self.k_candidates]
    
    def _bfs_delay_expansion(self,
                            start_node: str,
                            T_limit: float,
                            initial_delay: float) -> Dict[str, Tuple[float, Optional[str]]]:
        """BFS 遍歷（帶延遲約束）"""
        delay_dict = {node: (float('inf'), None) for node in self.network.V}
        delay_dict[start_node] = (initial_delay, None)
        queue = [(start_node, initial_delay)]
        
        while queue:
            current_node, current_delay = queue.pop(0)
            neighbors = self.topology_map.get(current_node, {}).get('Neighbors', [])
            
            for neighbor in neighbors:
                if neighbor not in self.network.V:
                    continue
                
                edge = (current_node, neighbor)
                distance = self.network.edge_distances.get(edge, 200)
                T_prop = distance / self.LIGHT_SPEED
                total_delay = current_delay + T_prop + self.NODE_DELAY_PER_HOP
                
                if total_delay > T_limit:
                    continue
                
                if total_delay < delay_dict[neighbor][0]:
                    delay_dict[neighbor] = (total_delay, current_node)
                    queue.append((neighbor, total_delay))
        
        return delay_dict
    
    def _get_sfs_at_node(self, node: str) -> List[ServiceFarm]:
        """獲取節點上的服務農場"""
        connected = self.topology_map.get(node, {}).get('ConnectedSF', [])
        return [sf for sf in connected if isinstance(sf, ServiceFarm)]
    
    def _greedy_combine_sfs(self,
                           sfs: List[ServiceFarm],
                           F_i: float,
                           M_i: float) -> Optional[Tuple]:
        """貪婪組合服務農場"""
        selected = []
        allocations = {}
        total_compute = 0
        total_memory = 0
        total_power = 0
        
        # 按可用記憶體排序
        sorted_sfs = sorted(sfs, key=lambda sf: sf.M_avail_sf, reverse=True)
        
        for sf in sorted_sfs:
            compute_alloc = min(sf.C_avail_sf, max(0, F_i - total_compute))
            memory_alloc = min(sf.M_avail_sf, max(0, M_i - total_memory))
            
            if compute_alloc > 0 or memory_alloc > 0:
                selected.append(sf)
                allocations[sf.sf_id] = (compute_alloc, memory_alloc)
                total_compute += compute_alloc
                total_memory += memory_alloc
                total_power += 30  # 簡化的功率模型
            
            if total_compute >= F_i and total_memory >= M_i:
                break
        
        if total_compute < F_i or total_memory < M_i:
            return None
        
        processing_time = F_i / total_compute if total_compute > 0 else float('inf')
        return selected, allocations, processing_time, total_power, total_memory
    
    def _reconstruct_path(self,
                         start: str,
                         end: str,
                         delay_dict: Dict) -> List[str]:
        """重建路徑"""
        path = []
        current = end
        max_iterations = len(self.network.V) + 1  # 防止無限迴圈
        iterations = 0
        
        while current != start and current is not None and iterations < max_iterations:
            path.insert(0, current)
            _, current = delay_dict.get(current, (float('inf'), None))
            iterations += 1
        
        if current == start:
            path.insert(0, start)
        else:
            logger.warning(f"路徑重建失敗: {start} -> {end}")
            return []
        
        return path
    
    def _compute_path_features(self,
                              path_edges: List[Tuple[str, str]],
                              request: Request) -> PathFeatures:
        """
        計算路徑特徵（修正版）
        
        修正：z4 延遲計算現在正確計算傳播延遲和節點延遲
        """
        wavelength_set = set(self.network.A)
        
        z1 = PathFeatures.compute_z1(path_edges, wavelength_set, self.network)
        z2 = PathFeatures.compute_z2(path_edges, wavelength_set, self.network)
        n_r = request.required_wavelengths(self.network.C_v)
        z3 = PathFeatures.compute_z3(z2, n_r)
        
        # 修正：正確計算路徑延遲
        total_distance = sum(
            self.network.edge_distances.get(edge, 200)
            for edge in path_edges
        )
        
        # 傳播延遲 = 總距離 / 光速
        prop_delay = total_distance / self.LIGHT_SPEED
        
        # 節點處理延遲 = 跳數 * 每跳延遲
        node_delay = self.NODE_DELAY_PER_HOP * len(path_edges)
        
        # 總延遲
        z4 = prop_delay + node_delay
        
        # 資源可用性
        z5 = 1.0
        
        return PathFeatures(z1, z2, z3, z4, z5)


@dataclass
class State:
    """環境狀態"""
    u_t: Request
    candidates: List[CandidateInfo]
    network_state: Dict = None
    current_time: float = 0.0
    
    def get_state_vector(self,
                        K_max: int = 10,
                        all_nodes: List[str] = None) -> np.ndarray:
        """
        將狀態轉換為向量表示
        
        修正：使用固定的 node -> index 映射以確保一致性
        """
        # 節點編碼（使用固定映射）
        if all_nodes and self.u_t.o_i in all_nodes and self.u_t.d_i in all_nodes:
            o_idx = all_nodes.index(self.u_t.o_i) / len(all_nodes)
            d_idx = all_nodes.index(self.u_t.d_i) / len(all_nodes)
        else:
            # 備用方案：使用 hash（注意：跨 process 可能不一致）
            o_idx = hash(self.u_t.o_i) % 1000 / 1000
            d_idx = hash(self.u_t.d_i) % 1000 / 1000
        
        # 請求特徵（歸一化）
        request_vec = np.array([
            o_idx,
            d_idx,
            self.u_t.b_i / 200.0,  # 歸一化頻寬
            self.u_t.tau_i / 10.0,  # 歸一化延遲
            self.u_t.F_i / 1e9,  # 歸一化計算需求
            self.u_t.M_i / 200.0,  # 歸一化記憶體
            self.u_t.p_i / 10.0,  # 歸一化優先級
            self.u_t.arrival_time / 100.0  # 歸一化時間
        ])
        
        # 候選特徵（填充）
        candidate_vecs = []
        for i in range(K_max):
            if i < len(self.candidates):
                pf = self.candidates[i].path_features
                candidate_vecs.append([
                    pf.z1_k / 80.0,  # 歸一化
                    pf.z2_k / 80.0,
                    pf.z3_k,
                    pf.z4_k / 10.0,
                    pf.z5_k
                ])
            else:
                candidate_vecs.append([0.0] * 5)
        
        state_vec = np.concatenate([
            request_vec,
            np.array(candidate_vecs).flatten()
        ])
        
        return state_vec


class OpticalNetworkEnvironment:
    """光網路環境"""
    
    def __init__(self,
                 network: NetworkGraph,
                 service_farms: List[ServiceFarm],
                 topology_map: Dict,
                 max_requests: int = 100,
                 simulation_time: float = 1000.0,
                 mean_interarrival_time: float = 10.0,
                 seed: int = 42):
        self.network = network
        self.service_farms = service_farms
        self.topology_map = topology_map
        self.max_requests = max_requests
        self.simulation_time = simulation_time
        self.mean_interarrival_time = mean_interarrival_time
        
        # 修正：固定隨機種子
        self.seed = seed
        self._set_seed(seed)
        
        # 初始化組件
        self.bfs_generator = BFSCandidateGenerator(
            network, service_farms, topology_map, k_candidates=5
        )
        self.reward_calculator = RewardCalculator()
        
        # 狀態追踪
        self.current_time = 0.0
        self.request_queue = []
        self.current_request_idx = 0
        self.success_count = 0
        self.fail_count = 0
        self.phi_t_minus = 0.0
        
        # 資源追踪
        self.active_wavelength_assignments = []
        
        logger.info(f"環境初始化: max_requests={max_requests}, "
                   f"simulation_time={simulation_time}s, seed={seed}")
    
    def _set_seed(self, seed: int):
        """設置隨機種子以便重現"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def reset(self) -> State:
        """重置環境到初始狀態"""
        # 清除網路狀態
        self.network.fiber_usage.clear()
        
        # 重置服務農場
        for sf in self.service_farms:
            sf.C_avail_sf = sf.C_max_sf
            sf.M_avail_sf = sf.M_max_sf
            sf.total_allocations = 0
        
        # 重置指標
        self.current_time = 0.0
        self.current_request_idx = 0
        self.success_count = 0
        self.fail_count = 0
        self.phi_t_minus = 0.0
        self.active_wavelength_assignments.clear()
        
        # 生成請求佇列
        self.request_queue = self._generate_request_queue()
        
        if not self.request_queue:
            raise ValueError("無法生成請求")
        
        first_request = self.request_queue[0]
        self.current_time = first_request.arrival_time
        
        # 生成初始候選
        candidates = self.bfs_generator.generate_candidates(first_request)
        
        logger.info(f"環境重置: 生成 {len(self.request_queue)} 個請求")
        
        return State(
            u_t=first_request,
            candidates=candidates,
            current_time=self.current_time
        )
    
    def step(self,
             action_index: int,
             state: State) -> Tuple[State, float, bool, Dict]:
        """
        執行一步環境交互
        
        Returns:
            (next_state, reward, done, info)
        """
        request = state.u_t
        candidates = state.candidates
        self.current_time = request.arrival_time
        
        # 釋放過期的波長
        num_released = self._release_expired_wavelengths(self.current_time)
        
        # 執行動作
        if action_index < 0 or action_index >= len(candidates):
            # 無效動作或選擇阻塞
            reward = -self.reward_calculator.R_block
            self.fail_count += 1
            success = False
        else:
            selected = candidates[action_index]
            success, allocation = self._execute_candidate(selected, request)
            
            if success and allocation:
                self.active_wavelength_assignments.extend(
                    allocation.wavelength_assignments
                )
                
                # 計算獎勵
                phi_t = self._compute_fragmentation()
                reward = self.reward_calculator.compute_reward(
                    success=True,
                    request=request,
                    sf=selected.sf_combination[0] if selected.sf_combination else None,
                    tau_i=request.tau_i,
                    total_delay=selected.total_delay + selected.processing_time,
                    P_used=selected.power_consumption,
                    P_max=500.0,
                    phi_t=phi_t,
                    phi_t_minus=self.phi_t_minus
                )
                
                self.phi_t_minus = phi_t
                self.success_count += 1
            else:
                reward = -self.reward_calculator.R_block
                self.fail_count += 1
        
        # 前進到下一個請求
        self.current_request_idx += 1
        
        # 檢查是否完成
        if self.current_request_idx >= len(self.request_queue):
            done = True
            next_state = state
            info = self._get_info(success, num_released, 0)
            return next_state, reward, done, info
        
        # 生成下一個狀態
        next_request = self.request_queue[self.current_request_idx]
        self.current_time = next_request.arrival_time
        next_candidates = self.bfs_generator.generate_candidates(next_request)
        
        next_state = State(
            u_t=next_request,
            candidates=next_candidates,
            current_time=self.current_time
        )
        
        info = self._get_info(success, num_released, len(next_candidates))
        
        return next_state, reward, False, info
    
    def _execute_candidate(self,
                          candidate: CandidateInfo,
                          request: Request) -> Tuple[bool, Optional[AllocatedResources]]:
        """執行候選方案"""
        # 尋找連續波長
        n_r = request.required_wavelengths(self.network.C_v)
        wavelength_range = self._find_continuous_wavelengths(
            candidate.path_edges, n_r
        )
        
        if wavelength_range is None:
            return False, None
        
        # 創建並執行動作
        action = Action(
            k_i=candidate.candidate_id,
            sf_i=candidate.sf_combination[0].sf_id if candidate.sf_combination else None,
            wavelength_start=wavelength_range[0],
            wavelength_end=wavelength_range[1],
            path_edges=candidate.path_edges,
            sf_allocations=candidate.sf_allocations
        )
        
        return action.execute_action(
            self.network,
            request,
            self.service_farms,
            self.current_time
        )
    
    def _find_continuous_wavelengths(self,
                                    path_edges: List[Tuple[str, str]],
                                    n_r: int) -> Optional[Tuple[int, int]]:
        """
        尋找連續可用波長（修正版）
        
        修正：在檢查連續性前先排序
        """
        # 獲取可用波長
        available = []
        for w in self.network.A:
            if all(self.network.is_wavelength_available(edge, w)
                  for edge in path_edges):
                available.append(w)
        
        if len(available) < n_r:
            return None
        
        # 修正：排序以確保連續性檢查正確
        available = sorted(available)
        
        # 尋找連續段
        for i in range(len(available) - n_r + 1):
            is_contiguous = all(
                available[i+j+1] == available[i+j] + 1
                for j in range(n_r - 1)
            )
            if is_contiguous:
                return available[i], available[i + n_r - 1]
        
        return None
    
    def _release_expired_wavelengths(self, current_time: float) -> int:
        """釋放過期的波長"""
        to_remove = []
        count = 0
        
        for i, assignment in enumerate(self.active_wavelength_assignments):
            if current_time >= assignment.release_time:
                self.network.release_wavelength(assignment.edge, assignment.wavelength)
                to_remove.append(i)
                count += 1
        
        for i in reversed(to_remove):
            self.active_wavelength_assignments.pop(i)
        
        if count > 0:
            logger.debug(f"釋放 {count} 個波長 at t={current_time:.2f}s")
        
        return count
    
    def _compute_fragmentation(self) -> float:
        """計算頻譜碎片化程度"""
        total_segments = 0
        total_available = 0
        
        for edge in self.network.E:
            # 修正：排序以確保正確計算碎片
            available = sorted([
                w for w in self.network.A
                if self.network.is_wavelength_available(edge, w)
            ])
            
            if not available:
                continue
            
            segments = 1
            for i in range(len(available) - 1):
                if available[i+1] != available[i] + 1:
                    segments += 1
            
            total_segments += segments
            total_available += len(available)
        
        return total_segments / total_available if total_available > 0 else 0.0
    
    def _generate_request_queue(self) -> List[Request]:
        """生成請求佇列"""
        requests = []
        current_time = 0.0
        request_id = 0
        
        while current_time < self.simulation_time and request_id < self.max_requests:
            interarrival = np.random.exponential(self.mean_interarrival_time)
            current_time += interarrival
            
            if current_time >= self.simulation_time:
                break
            
            source = random.choice(self.network.V)
            destination = random.choice([n for n in self.network.V if n != source])
            
            request = Request(
                request_id=request_id,
                o_i=source,
                d_i=destination,
                b_i=max(10, np.random.exponential(50)),
                tau_i=max(0.5, np.random.exponential(2)),
                F_i=max(1e7, np.random.normal(1e8, 5e7)),
                M_i=max(10, np.random.normal(50, 20)),
                p_i=random.randint(1, 10),
                arrival_time=current_time
            )
            
            requests.append(request)
            request_id += 1
        
        return requests
    
    def _get_info(self, success: bool, released: int, num_candidates: int) -> Dict:
        """獲取環境資訊"""
        total = self.success_count + self.fail_count
        return {
            'success': success,
            'success_rate': self.success_count / total if total > 0 else 0.0,
            'num_candidates': num_candidates,
            'active_wavelengths': len(self.active_wavelength_assignments),
            'current_time': self.current_time,
            'wavelengths_released': released
        }


# ============================================================================
# 修正 4: PolicyNetwork - 返回 logits 而非 probabilities
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    策略網路（修正版）
    
    修正：forward 現在返回 logits 而非 probabilities
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)  # 返回 logits
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        修正：返回 logits（未歸一化的分數），不做 softmax
        
        Returns:
            logits: [batch_size, action_dim]
        """
        logits = self.network(state)
        return logits  # 修正：不做 softmax


class ValueNetwork(nn.Module):
    """價值網路"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


# ============================================================================
# 修正 5-7: OpticalNetworkAgent - 完整修正
# ============================================================================

class OpticalNetworkAgent:
    """
    光網路代理（修正版）
    
    主要修正：
    1. select_action 正確實現 masking（對 logits 做 mask）
    2. 保存並使用 log_probs
    3. 實現 GAE（Generalized Advantage Estimation）
    4. 直接儲存 state_vector 而非 State 物件
    5. 處理無候選情況（使用 REJECT action）
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 entropy_coef: float = 0.01,
                 seed: int = 42):
        self.state_dim = state_dim
        self.action_dim = action_dim  # 包含一個 REJECT action
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        
        # 修正：固定隨機種子
        self._set_seed(seed)
        
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=learning_rate
        )
        
        logger.info(f"代理初始化: state_dim={state_dim}, "
                   f"action_dim={action_dim}, lr={learning_rate}, seed={seed}")
    
    def _set_seed(self, seed: int):
        """設置隨機種子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def select_action(self,
                     state: State,
                     num_valid_candidates: int) -> Tuple[int, torch.Tensor]:
        """
        選擇動作（修正版）
        
        修正：
        1. 獲取 logits（而非 probabilities）
        2. 對 logits 做 mask（而非對 probs）
        3. 然後才做 softmax
        4. 正確處理無候選的情況
        
        Args:
            state: 當前狀態
            num_valid_candidates: 有效候選數量
        
        Returns:
            (action_index, log_prob)
        """
        state_vec = state.get_state_vector()
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)  # [1, state_dim]
        
        with torch.no_grad():
            # 修正：獲取 logits（未歸一化）
            logits = self.policy_net(state_tensor)  # [1, action_dim]
            
            # 處理無候選的情況
            if num_valid_candidates == 0:
                # 選擇 REJECT action（最後一個 action）
                action = self.action_dim - 1
                probs = F.softmax(logits, dim=-1)
                log_prob = torch.log(probs[0, action] + 1e-10)
                return action, log_prob
            
            # 修正：對 logits 做 masking（而非對 probs）
            if num_valid_candidates < self.action_dim - 1:  # -1 for REJECT action
                mask = torch.zeros_like(logits)
                # 將無效候選的 logits 設為很小的負數
                mask[0, num_valid_candidates:-1] = -1e9
                logits = logits + mask
            
            # 修正：在 masking 後才做 softmax
            probs = F.softmax(logits, dim=-1)  # [1, action_dim]
            
            # 使用 Categorical 分佈採樣
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob
    
    def update(self,
               states: List[np.ndarray],
               actions: List[int],
               log_probs: List[torch.Tensor],
               rewards: List[float],
               dones: List[bool]) -> Dict[str, float]:
        """
        更新策略和價值網路（修正版）
        
        修正：
        1. 使用儲存的 log_probs
        2. 實現 GAE
        3. 直接使用 state_vectors
        
        Args:
            states: 狀態向量列表（numpy arrays）
            actions: 動作列表
            log_probs: 儲存的 log 概率列表
            rewards: 獎勵列表
            dones: 完成標誌列表
        
        Returns:
            損失字典
        """
        if len(states) == 0:
            return {}
        
        # 轉換為 tensors（修正：直接使用 numpy arrays）
        states_t = torch.FloatTensor(np.array(states))  # [T, state_dim]
        actions_t = torch.LongTensor(actions)  # [T]
        rewards_t = torch.FloatTensor(rewards)  # [T]
        dones_t = torch.FloatTensor(dones)  # [T]
        
        # 修正：使用儲存的 log_probs（而非重新計算）
        log_probs_t = torch.stack(log_probs)  # [T]
        
        # 計算 values
        with torch.no_grad():
            values = self.value_net(states_t).squeeze()  # [T]
        
        # 修正：使用 GAE 計算 advantages
        advantages = torch.zeros_like(rewards_t)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards_t[t] + self.gamma * next_value * (1 - dones_t[t]) - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones_t[t]) * gae
            advantages[t] = gae
        
        # 歸一化 advantages（提升訓練穩定性）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 計算 returns
        returns = advantages + values
        
        # ============ 更新 Value Network ============
        values_pred = self.value_net(states_t).squeeze()
        value_loss = F.mse_loss(values_pred, returns.detach())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        # ============ 更新 Policy Network ============
        # 重新計算 logits 以獲取 entropy
        logits = self.policy_net(states_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # 修正：使用儲存的 log_probs 計算 policy loss
        policy_loss = -(log_probs_t * advantages.detach()).mean()
        
        # 添加熵正則化（鼓勵探索）
        entropy = dist.entropy().mean()
        policy_loss = policy_loss - self.entropy_coef * entropy
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def train_episode(self,
                     env: OpticalNetworkEnvironment,
                     max_steps: Optional[int] = None) -> Dict[str, float]:
        """
        訓練一個 episode（修正版）
        
        修正：
        1. 儲存 state_vector 而非 State 物件
        2. 儲存 log_probs
        3. 不使用 max(0, action)
        """
        state = env.reset()
        
        # 修正：直接儲存 state_vectors 和 log_probs
        states = []  # List[np.ndarray]
        actions = []
        log_probs = []  # 修正：儲存 log_probs
        rewards = []
        dones = []
        
        episode_reward = 0.0
        step = 0
        
        while True:
            num_candidates = len(state.candidates)
            
            # 選擇動作（修正：儲存 log_prob）
            action, log_prob = self.select_action(state, num_candidates)
            
            # 執行動作
            next_state, reward, done, info = env.step(action, state)
            
            # 修正：儲存 state_vector 和原始 action
            states.append(state.get_state_vector())
            actions.append(action)  # 修正：不使用 max(0, action)
            log_probs.append(log_prob)  # 修正：儲存 log_prob
            rewards.append(reward)
            dones.append(done)
            
            episode_reward += reward
            step += 1
            state = next_state
            
            if done or (max_steps and step >= max_steps):
                break
        
        # 更新網路（修正：傳入 log_probs）
        losses = self.update(states, actions, log_probs, rewards, dones)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': step,
            'success_rate': info.get('success_rate', 0.0),
            **losses
        }


# ============================================================================
# 主訓練流程
# ============================================================================

def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # 初始化網路拓撲
    logger.info("Init Tuple...")
    nodes = ['r1', 'r2', 'r3']
    edges = [('r1', 'r2'), ('r2', 'r1'), ('r2', 'r3'), ('r3', 'r2')]
    wavelengths = list(range(1, 21))  # 20 個波長
    
    network = NetworkGraph(V=nodes, E=edges, A=wavelengths)
    network.edge_distances = {
        ('r1', 'r2'): 200, ('r2', 'r1'): 200,
        ('r2', 'r3'): 300, ('r3', 'r2'): 300
    }
    
    # 初始化服務農場
    logger.info("Init SF...")
    sf1 = ServiceFarm('SF1', C_max=1e8, M_max=100)
    sf2 = ServiceFarm('SF2', C_max=2e8, M_max=150)
    sf3 = ServiceFarm('SF3', C_max=1.5e8, M_max=120)
    service_farms = [sf1, sf2, sf3]
    
    # 拓撲映射
    topology_map = {
        'r1': {'Neighbors': ['r2'], 'ConnectedSF': [sf1]},
        'r2': {'Neighbors': ['r1', 'r3'], 'ConnectedSF': [sf2]},
        'r3': {'Neighbors': ['r2'], 'ConnectedSF': [sf3]}
    }
    
    # 創建環境
    logger.info("Create Environmnt...")
    env = OpticalNetworkEnvironment(
        network=network,
        service_farms=service_farms,
        topology_map=topology_map,
        max_requests=30,
        simulation_time=300.0,
        mean_interarrival_time=10.0,
        seed=SEED
    )
    
    # 創建代理
    logger.info("Create Agent...")
    K_max = 10
    state_dim = 8 + K_max * 5
    action_dim = K_max + 1  # +1 for REJECT action
    agent = OpticalNetworkAgent(
        state_dim,
        action_dim,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        seed=SEED
    )
    
    # 訓練循環
    logger.info("="*70)
    logger.info("開始訓練...")
    logger.info("="*70)
    
    num_episodes = 10
    
    for episode in range(num_episodes):
        logger.info(f"\nEpisode {episode+1}/{num_episodes}")
        logger.info("-"*70)
        
        stats = agent.train_episode(env)
        
        logger.info(f"結果:")
        logger.info(f"  獎勵: {stats['episode_reward']:.2f}")
        logger.info(f"  成功率: {stats['success_rate']:.2%}")
        logger.info(f"  長度: {stats['episode_length']}")
        
        if 'policy_loss' in stats:
            logger.info(f"  策略損失: {stats['policy_loss']:.4f}")
            logger.info(f"  價值損失: {stats['value_loss']:.4f}")
            logger.info(f"  熵: {stats['entropy']:.4f}")
        
        # 資源統計
        logger.info("資源狀態:")
        for sf in service_farms:
            util = sf.get_utilization()
            logger.info(f"  {sf.sf_id}: 計算={util['compute']:.1%}, "
                       f"記憶體={util['memory']:.1%}")
    
    logger.info("="*70)
    logger.info("訓練完成")
    logger.info("="*70)

if __name__ == "__main__":
    main()