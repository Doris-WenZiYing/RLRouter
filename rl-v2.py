import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import math
import random

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
        self.edge_distances = {}  # {(u, v): distance_km}
        
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


@dataclass
class PathFeatures:
    """路徑屬性特徵 P_k"""
    z1_k: float  # 可用波長數量
    z2_k: float  # 最大連續可用波長數
    z3_k: float  # 是否滿足連續波長需求
    z4_k: float  # 路徑延遲
    z5_k: float  # SF 資源比率
    
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
class ServiceFarm:
    sf_id: str  # 特定sf
    
    def __init__(self, sf_id: str, C_max: float, M_max: float):
        self.sf_id = sf_id
        self.C_max_sf = C_max
        self.M_max_sf = M_max
        self.C_avail_sf = C_max
        self.M_avail_sf = M_max
        
    def compute_q_i(self, t: int) -> Dict[str, float]:
        C_tilde = self.C_avail_sf / self.C_max_sf if self.C_max_sf > 0 else 0.0
        M_tilde = self.M_avail_sf / self.M_max_sf if self.M_max_sf > 0 else 0.0
        W_tilde = 0.0  # TODO: 根據實際功耗模型計算
        
        return {
            'avail': 1.0 if self.C_avail_sf > 0 and self.M_avail_sf > 0 else 0.0,
            'C_ratio': C_tilde,
            'M_ratio': M_tilde,
            'W_power': W_tilde
        }
    
    def allocate_resources(self, compute: float, memory: float) -> bool:
        if self.C_avail_sf >= compute and self.M_avail_sf >= memory:
            self.C_avail_sf -= compute
            self.M_avail_sf -= memory
            return True
        return False
    
    def release_resources(self, compute: float, memory: float):
        self.C_avail_sf = min(self.C_avail_sf + compute, self.C_max_sf)
        self.M_avail_sf = min(self.M_avail_sf + memory, self.M_max_sf)


@dataclass
class AllocatedResources:
    # 已分配資源
    wavelengths: List[Tuple[Tuple[str, str], int, int]]  # [(edge, wavelength, time), ...]
    sfs: Dict[str, Tuple[float, float]]  # {sf_id: (compute_allocated, memory_allocated)}
    path_edges: List[Tuple[str, str]]
    release_time: float


@dataclass
class Action:
    k_i: int  # 候選路徑索引
    sf_i: str  # 在該路徑上的節點中選一個
    wavelength_start: int
    wavelength_end: int
    path_edges: List[Tuple[str, str]] = None  # 路徑邊集合
    sf_allocations: Dict[str, Tuple[float, float]] = None  # {sf_id: (compute, memory)}
    
    @property
    def contiguous_wavelength_set(self) -> Set[int]:
        return set(range(self.wavelength_start, self.wavelength_end + 1))
    
    def execute_action(self, 
                      network: NetworkGraph, 
                      request: Request, 
                      sfs: List[ServiceFarm], 
                      current_time: int) -> Tuple[bool, Optional[AllocatedResources]]:
        """
        Returns:
            (success, allocated_resources)
        """
        # 步驟 1: 驗證波長可用性
        wavelength_set = self.contiguous_wavelength_set
        for edge in self.path_edges:
            for wavelength in wavelength_set:
                if network.f_ea(edge, wavelength, current_time) == 1:
                    return False, None  # 波長已被佔用
        
        # 步驟 2: 驗證 SF 資源
        if self.sf_allocations is None:
            return False, None
        
        sf_dict = {sf.sf_id: sf for sf in sfs}
        for sf_id, (compute_needed, memory_needed) in self.sf_allocations.items():
            if sf_id not in sf_dict:
                return False, None
            sf = sf_dict[sf_id]
            if sf.C_avail_sf < compute_needed or sf.M_avail_sf < memory_needed:
                return False, None  # SF 資源不足
        
        # 步驟 3: 分配波長
        allocated_wavelengths = []
        for edge in self.path_edges:
            for wavelength in wavelength_set:
                network.fiber_usage[(edge, wavelength, current_time)] = 1
                allocated_wavelengths.append((edge, wavelength, current_time))
        
        # 步驟 4: 分配 SF 資源
        for sf_id, (compute_needed, memory_needed) in self.sf_allocations.items():
            sf = sf_dict[sf_id]
            sf.allocate_resources(compute_needed, memory_needed)
        
        # 步驟 5: 記錄分配的資源
        allocated = AllocatedResources(
            wavelengths=allocated_wavelengths,
            sfs=self.sf_allocations.copy(),
            path_edges=self.path_edges.copy(),
            release_time=current_time + request.tau_i
        )
        
        return True, allocated


class RewardCalculator:
    def __init__(self, 
                 R_succ: float = 100.0,
                 R_block: float = 50.0,
                 w_d: float = 1.0,
                 w_p: float = 1.0,
                 w_f: float = 1.0,
                 w_pri: float = 1.0,
                 upsilon: float = 0.5):
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
        if not success:
            return -self.R_block
            
        if sf is None or sf.C_avail_sf < 0 or sf.M_avail_sf < 0:
            return -self.R_block
            
        # 延遲效率
        n_delay = (tau_i - total_delay) / tau_i if tau_i > 0 and total_delay < tau_i else 0.0
        
        # 功耗效率
        n_power = 1.0 - (P_used / P_max) if P_max > 0 else 0.0
        
        # 碎片化效率
        delta_phi = phi_t - phi_t_minus
        n_frag = np.exp(-self.upsilon * max(0.0, delta_phi))
        
        g_p = request.p_i
        
        reward = self.R_succ + \
                self.w_d * n_delay + \
                self.w_p * n_power + \
                self.w_f * n_frag + \
                self.w_pri * g_p
        
        return reward

@dataclass
class CandidateInfo:
    # 候選路徑
    candidate_id: int
    path_edges: List[Tuple[str, str]]
    path_nodes: List[str]
    sf_combination: List[ServiceFarm]
    sf_allocations: Dict[str, Tuple[float, float]]  # {sf_id: (compute, memory)}
    total_delay: float
    processing_time: float
    power_consumption: float
    memory_usage: float
    cost: float
    path_features: PathFeatures


class BFSCandidateGenerator:
    
    def __init__(self, 
                 network: NetworkGraph, 
                 service_farms: List[ServiceFarm],
                 topology_map: Dict, 
                 k_candidates: int = 5):
        self.network = network
        self.service_farms = service_farms
        self.topology_map = topology_map
        self.k_candidates = k_candidates
        
    def generate_candidates(self, 
                          request: Request, 
                          current_time: int = 0) -> List[CandidateInfo]:
        
        # 步驟 1: 計算時間限制
        available_sfs = [sf for sf in self.service_farms 
                        if sf.C_avail_sf > 0 and sf.M_avail_sf > 0]
        
        if not available_sfs:
            return []
        
        compute_min = min(sf.C_avail_sf for sf in available_sfs)
        T_compute = request.F_i / compute_min if compute_min > 0 else float('inf')
        
        if request.tau_i <= T_compute:
            return []
        
        T_limit = request.tau_i - T_compute
        
        # 步驟 2: 計算初始延遲
        B = 50 * 1e9  # 50 Gbps
        gOSNR = 20
        T_trans = request.b_i / (B * math.log2(1 + gOSNR))
        T_node = 0.0001
        initial_delay = T_trans + T_node
        
        # 步驟 3: BFS 延遲擴展
        delay_dict = self._bfs_delay_expansion(request.o_i, T_limit, initial_delay)
        
        # 步驟 4: 構建候選列表
        candidates = []
        candidate_id = 0
        
        for target_node in self.network.V:
            if target_node == request.o_i:
                continue
                
            delay, prev = delay_dict[target_node]
            if delay > T_limit or delay == float('inf'):
                continue
            
            # 找該節點的 SF
            available_sfs_at_node = self._get_sfs_at_node(target_node)
            if not available_sfs_at_node:
                continue
            
            # 貪婪組合 SF（已修正）
            result = self._greedy_combine_sfs(available_sfs_at_node, request.F_i, request.M_i)
            if result is None:
                continue
            
            selected_sfs, sf_allocations, processing_time, power, memory = result
            
            # 重建路徑
            path_nodes = self._reconstruct_path(request.o_i, target_node, delay_dict)
            path_edges = [(path_nodes[i], path_nodes[i+1]) 
                         for i in range(len(path_nodes)-1)]
            
            # 計算成本
            alpha, beta = 1.0, 0.01
            cost = alpha * (delay + processing_time) + beta * power
            
            # 計算路徑特徵
            path_features = self._compute_path_features(path_edges, request, current_time)
            
            # 檢查是否有足夠連續波長
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
        
        # 排序並返回前 K 個
        candidates.sort(key=lambda c: c.cost)
        return candidates[:self.k_candidates]
    
    def _bfs_delay_expansion(self, 
                            start_node: str, 
                            T_limit: float, 
                            initial_delay: float) -> Dict[str, Tuple[float, str]]:
        delay_dict = {node: (float('inf'), None) for node in self.network.V}
        delay_dict[start_node] = (initial_delay, None)
        queue = [(start_node, initial_delay)]
        
        while queue:
            next_queue = []
            for current, current_delay in queue:
                neighbors = self.topology_map.get(current, {}).get('Neighbors', [])
                for neighbor in neighbors:
                    if neighbor not in self.network.V:
                        continue
                    
                    edge = (current, neighbor)
                    distance = self.network.edge_distances.get(edge, 200)
                    T_prop = distance / (2 * 10**5)
                    T_node = 0.0001
                    total_delay = current_delay + T_prop + T_node
                    
                    if total_delay > T_limit:
                        continue
                    
                    if total_delay < delay_dict[neighbor][0]:
                        delay_dict[neighbor] = (total_delay, current)
                        next_queue.append((neighbor, total_delay))
            
            queue = next_queue
        
        return delay_dict
    
    def _get_sfs_at_node(self, node: str) -> List[ServiceFarm]:
        # 取得節點上的 SF
        connected_sfs = self.topology_map.get(node, {}).get('ConnectedSF', [])
        return [sf for sf in connected_sfs if isinstance(sf, ServiceFarm)]
    
    def _greedy_combine_sfs(self, 
                           available_sfs: List[ServiceFarm], 
                           F_i: float, 
                           M_i: float) -> Optional[Tuple]:
        """
        Returns:
            (selected_sfs, sf_allocations, processing_time, power, memory) 或 None
        """
        selected = []
        sf_allocations = {}
        total_compute = 0
        total_memory = 0
        total_power = 0
        
        # 按記憶體排序
        sorted_sfs = sorted(available_sfs, 
                          key=lambda sf: sf.M_avail_sf, 
                          reverse=True)
        
        for sf in sorted_sfs:
            # 計算需要從這個 SF 分配多少資源
            compute_to_allocate = min(sf.C_avail_sf, max(0, F_i - total_compute))
            memory_to_allocate = min(sf.M_avail_sf, max(0, M_i - total_memory))
            
            if compute_to_allocate > 0 or memory_to_allocate > 0:
                selected.append(sf)
                sf_allocations[sf.sf_id] = (compute_to_allocate, memory_to_allocate)
                total_compute += compute_to_allocate
                total_memory += memory_to_allocate
                total_power += 30  # 假設每個 SF 30W
            
            # 檢查是否都滿足
            if total_compute >= F_i and total_memory >= M_i:
                break
        
        # 驗證是否滿足需求
        if total_compute < F_i or total_memory < M_i:
            return None
        
        processing_time = F_i / total_compute if total_compute > 0 else float('inf')
        return selected, sf_allocations, processing_time, total_power, total_memory
    
    def _reconstruct_path(self, 
                         start: str, 
                         end: str, 
                         delay_dict: Dict) -> List[str]:
        path = []
        current = end
        while current != start and current is not None:
            path.insert(0, current)
            current = delay_dict[current][1]
        path.insert(0, start)
        return path
    
    def _compute_path_features(self, 
                              path_edges: List[Tuple[str, str]], 
                              request: Request, 
                              current_time: int) -> PathFeatures:
        # 計算路徑特徵
        wavelength_set = set(self.network.A)
        
        # 構建 fiber_usage
        f_usage = {}
        for edge in path_edges:
            for w in self.network.A:
                f_usage[(edge, w)] = self.network.f_ea(edge, w, current_time)
        
        z1 = PathFeatures.compute_z1(path_edges, wavelength_set, f_usage)
        z2 = PathFeatures.compute_z2(path_edges, wavelength_set, f_usage)
        n_r = request.required_wavelengths(self.network.C_v)
        z3 = PathFeatures.compute_z3(z2, n_r)
        
        # z4: 路徑延遲
        total_distance = sum(self.network.edge_distances.get(edge, 200) 
                           for edge in path_edges)
        z4 = total_distance / (2 * 10**5) * len(path_edges)
        
        # z5: SF 資源比
        z5 = 1.0
        
        return PathFeatures(z1, z2, z3, z4, z5)

@dataclass
class State:
    u_t: Request
    candidates: List[CandidateInfo]
    network_state: Dict = None
    
    def get_state_vector(self, K_max: int = 10, all_nodes: List[str] = None) -> np.ndarray:
        """
        TODO: 可能需要優化節點編排方式
        """
        # 節點
        if all_nodes:
            try:
                o_idx = all_nodes.index(self.u_t.o_i) / len(all_nodes)
                d_idx = all_nodes.index(self.u_t.d_i) / len(all_nodes)
            except ValueError:
                o_idx, d_idx = 0.0, 0.0
        else:
            o_idx = hash(self.u_t.o_i) % 1000 / 1000
            d_idx = hash(self.u_t.d_i) % 1000 / 1000
        
        # 請求特徵
        request_vec = np.array([
            o_idx,
            d_idx,
            self.u_t.b_i / 200,
            self.u_t.tau_i / 10,
            self.u_t.F_i / 1e9,
            self.u_t.M_i / 200,
            self.u_t.p_i / 10
        ])
        
        # 候選特徵
        candidate_vecs = []
        for i in range(K_max):
            if i < len(self.candidates):
                pf = self.candidates[i].path_features
                candidate_vecs.append([
                    pf.z1_k / 80,
                    pf.z2_k / 80,
                    pf.z3_k,
                    pf.z4_k / 10,
                    pf.z5_k
                ])
            else:
                candidate_vecs.append([0, 0, 0, 0, 0])
        
        state_vec = np.concatenate([
            request_vec,
            np.array(candidate_vecs).flatten()
        ])
        
        return state_vec

class OpticalNetworkEnvironment:
    def __init__(self, 
                 network: NetworkGraph, 
                 service_farms: List[ServiceFarm],
                 topology_map: Dict, 
                 max_requests: int = 100):
        self.network = network
        self.service_farms = service_farms
        self.topology_map = topology_map
        self.max_requests = max_requests
        
        # BFS 候選生成器
        self.bfs_generator = BFSCandidateGenerator(
            network, service_farms, topology_map, k_candidates=5
        )
        
        # 獎勵計算器
        self.reward_calculator = RewardCalculator()
        
        # 狀態追蹤
        self.current_time = 0.0
        self.request_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.phi_t_minus = 0.0
        
        #  資源管理
        self.active_requests: List[AllocatedResources] = []
        
    def reset(self) -> State:
        # 重置網路
        self.network.fiber_usage.clear()
        
        # 重置 SF
        for sf in self.service_farms:
            sf.C_avail_sf = sf.C_max_sf
            sf.M_avail_sf = sf.M_max_sf
        
        # 重置計數
        self.current_time = 0.0
        self.request_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.phi_t_minus = 0.0
        self.active_requests.clear()
        
        # 生成第一個請求
        request = self._generate_request()
        
        # 生成候選
        candidates = self.bfs_generator.generate_candidates(request, int(self.current_time))
        
        state = State(u_t=request, candidates=candidates)
        return state
    
    def step(self, action_index: int, state: State) -> Tuple[State, float, bool, Dict]:
        request = state.u_t
        candidates = state.candidates
        
        # 釋放過期資源
        self._release_expired_resources()
        
        # 檢查動作有效性
        if action_index >= len(candidates) or action_index < 0:
            # 無效動作或拒絕
            reward = -self.reward_calculator.R_block
            self.fail_count += 1
            success = False
        else:
            # 執行動作
            selected = candidates[action_index]
            success, allocated = self._execute_candidate(selected, request)
            
            if success and allocated:
                # 記錄分配的資源
                self.active_requests.append(allocated)
                
                # 計算獎勵
                phi_t = self._compute_fragmentation()
                reward = self.reward_calculator.compute_reward(
                    success=True,
                    request=request,
                    sf=selected.sf_combination[0] if selected.sf_combination else None,
                    tau_i=request.tau_i,
                    total_delay=selected.total_delay + selected.processing_time,
                    P_used=selected.power_consumption,
                    P_max=500,
                    phi_t=phi_t,
                    phi_t_minus=self.phi_t_minus
                )
                self.phi_t_minus = phi_t
                self.success_count += 1
            else:
                reward = -self.reward_calculator.R_block
                self.fail_count += 1
        
        # 推進時間
        self.current_time += 0.1  # 每步推進 0.1 秒
        
        # 生成下一個請求
        self.request_count += 1
        next_request = self._generate_request()
        next_candidates = self.bfs_generator.generate_candidates(
            next_request, int(self.current_time)
        )
        next_state = State(u_t=next_request, candidates=next_candidates)
        
        # 檢查結束
        done = self.request_count >= self.max_requests
        
        info = {
            'success': success,
            'success_rate': self.success_count / self.request_count if self.request_count > 0 else 0,
            'num_candidates': len(next_candidates),
            'active_requests': len(self.active_requests)
        }
        
        return next_state, reward, done, info
    
    def _execute_candidate(self, 
                          candidate: CandidateInfo, 
                          request: Request) -> Tuple[bool, Optional[AllocatedResources]]:

        #  選擇連續波長
        n_r = request.required_wavelengths(self.network.C_v)
        wavelength_start, wavelength_end = self._find_continuous_wavelengths(
            candidate.path_edges, n_r
        )
        
        if wavelength_start is None:
            return False, None
        
        #  創建動作
        action = Action(
            k_i=candidate.candidate_id,
            sf_i=candidate.sf_combination[0].sf_id if candidate.sf_combination else None,
            wavelength_start=wavelength_start,
            wavelength_end=wavelength_end,
            path_edges=candidate.path_edges,
            sf_allocations=candidate.sf_allocations
        )
        
        #  執行動作
        success, allocated = action.execute_action(
            self.network, 
            request, 
            self.service_farms, 
            int(self.current_time)
        )
        
        return success, allocated
    
    def _find_continuous_wavelengths(self, 
                                    path_edges: List[Tuple[str, str]], 
                                    n_r: int) -> Tuple[Optional[int], Optional[int]]:
        """
         在路徑上找 n_r 個連續可用波長
        Returns:
            (wavelength_start, wavelength_end) 或 (None, None)
        """
        # 找所有在整條路徑上可用的波長
        available_wavelengths = []
        for w in self.network.A:
            is_available = True
            for edge in path_edges:
                if self.network.f_ea(edge, w, int(self.current_time)) == 1:
                    is_available = False
                    break
            if is_available:
                available_wavelengths.append(w)
        
        if len(available_wavelengths) < n_r:
            return None, None
        
        # 找連續片段
        for i in range(len(available_wavelengths) - n_r + 1):
            # 檢查是否連續
            is_continuous = True
            for j in range(n_r - 1):
                if available_wavelengths[i + j + 1] != available_wavelengths[i + j] + 1:
                    is_continuous = False
                    break
            
            if is_continuous:
                return available_wavelengths[i], available_wavelengths[i + n_r - 1]
        
        return None, None
    
    def _release_expired_resources(self):
        """釋放過期資源"""
        to_remove = []
        
        for i, allocated in enumerate(self.active_requests):
            if self.current_time >= allocated.release_time:
                # 釋放波長
                for edge, wavelength, time in allocated.wavelengths:
                    key = (edge, wavelength, time)
                    if key in self.network.fiber_usage:
                        del self.network.fiber_usage[key]
                
                # 釋放 SF 資源
                sf_dict = {sf.sf_id: sf for sf in self.service_farms}
                for sf_id, (compute, memory) in allocated.sfs.items():
                    if sf_id in sf_dict:
                        sf_dict[sf_id].release_resources(compute, memory)
                
                to_remove.append(i)
        
        # 移除已釋放的請求
        for i in reversed(to_remove):
            self.active_requests.pop(i)
    
    def _compute_fragmentation(self) -> float:
        """
        phi = 連續段數 / 總可用波長數
        """
        total_segments = 0
        total_available = 0
        
        for edge in self.network.E:
            available_wavelengths = []
            for w in self.network.A:
                if self.network.f_ea(edge, w, int(self.current_time)) == 0:
                    available_wavelengths.append(w)
            
            if not available_wavelengths:
                continue
            
            # 計算連續段數
            segments = 1
            for i in range(len(available_wavelengths) - 1):
                if available_wavelengths[i + 1] != available_wavelengths[i] + 1:
                    segments += 1
            
            total_segments += segments
            total_available += len(available_wavelengths)
        
        return total_segments / total_available if total_available > 0 else 0
    
    def _generate_request(self) -> Request:
        nodes = self.network.V
        o_i = random.choice(nodes)
        d_i = random.choice([n for n in nodes if n != o_i])
        
        return Request(
            request_id=self.request_count,
            o_i=o_i,
            d_i=d_i,
            b_i=max(10, np.random.exponential(50)),
            tau_i=max(0.5, np.random.exponential(2)),
            F_i=max(1e7, np.random.normal(1e8, 5e7)),
            M_i=max(10, np.random.normal(100, 50)),
            p_i=random.randint(1, 10)
        )

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        logits = self.net(state)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.net(state)


class OpticalNetworkAgent:
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=learning_rate
        )
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
    
    def select_action(self, state: State, num_valid_candidates: int):
        state_vec = state.get_state_vector()
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        
        with torch.no_grad():
            probs = self.policy_net(state_tensor)
        
        # Masking 無效動作
        mask = torch.zeros_like(probs)
        if num_valid_candidates < probs.shape[1]:
            mask[0, num_valid_candidates:] = float('-inf')
        probs = F.softmax(probs + mask, dim=-1)
        
        # 採樣
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def update(self, trajectories):
        """
        TODO: 實現網路更新邏輯
        
        需要實現:
        1. 計算 advantages (TD error 或 GAE)
        2. 計算 policy loss (含 entropy bonus)
        3. 計算 value loss
        4. 反向傳播
        5. 梯度裁剪
        6. 更新參數
        """
        states, actions, rewards, next_states, dones = trajectories
        
        # TODO: 轉換為 tensor
        states_t = torch.FloatTensor([s.get_state_vector() for s in states])
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor([s.get_state_vector() for s in next_states])
        dones_t = torch.FloatTensor(dones)
        
        # TODO: 計算 advantages
        with torch.no_grad():
            values = self.value_net(states_t).squeeze()
            next_values = self.value_net(next_states_t).squeeze()
            next_values = next_values * (1 - dones_t)
            advantages = rewards_t + self.gamma * next_values - values
        
        # TODO: Policy loss
        probs = self.policy_net(states_t)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()
        
        policy_loss = -(advantages.detach() * log_probs).mean()
        policy_loss -= self.entropy_coef * entropy
        
        # TODO: Value loss
        predicted_values = self.value_net(states_t).squeeze()
        target_values = rewards_t + self.gamma * next_values
        value_loss = F.mse_loss(predicted_values, target_values.detach())
        
        # TODO: 更新
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def train_episode(self, env, max_steps=100):
        """
        TODO: 實現訓練迴圈
        
        需要實現:
        1. 重置環境
        2. 收集軌跡 (states, actions, rewards)
        3. 調用 update()
        4. 返回統計資料
        """
        state = env.reset()
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        episode_reward = 0
        
        for step in range(max_steps):
            # 選擇動作
            num_candidates = len(state.candidates)
            if num_candidates == 0:
                action = -1  # 拒絕
                log_prob = torch.tensor(0.0)
            else:
                action, log_prob = self.select_action(state, num_candidates)
            
            # 執行
            next_state, reward, done, info = env.step(action, state)
            
            # 記錄
            states.append(state)
            actions.append(max(0, action))  # 確保非負
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # 更新
        if len(states) > 0:
            losses = self.update((states, actions, rewards, next_states, dones))
        else:
            losses = {}
        
        return {
            'episode_reward': episode_reward,
            'episode_length': len(states),
            'success_rate': info.get('success_rate', 0),
            **losses
        }


def main():
    # 1. 初始化網路
    nodes = ['r1', 'r2', 'r3']
    edges = [('r1', 'r2'), ('r2', 'r1'), ('r2', 'r3'), ('r3', 'r2')]
    wavelengths = list(range(1, 11))  # 10 個波長
    
    network = NetworkGraph(V=nodes, E=edges, A=wavelengths)
    network.edge_distances = {
        ('r1', 'r2'): 200,
        ('r2', 'r1'): 200,
        ('r2', 'r3'): 300,
        ('r3', 'r2'): 300,
    }
    
    # 2. 初始化 SF
    sf1 = ServiceFarm('SF1', C_max=1e8, M_max=100)
    sf2 = ServiceFarm('SF2', C_max=2e8, M_max=150)
    sf3 = ServiceFarm('SF3', C_max=1.5e8, M_max=120)
    service_farms = [sf1, sf2, sf3]
    
    # 3. 拓撲對應
    topology_map = {
        'r1': {
            'Neighbors': ['r2'],
            'ConnectedSF': [sf1]
        },
        'r2': {
            'Neighbors': ['r1', 'r3'],
            'ConnectedSF': [sf2]
        },
        'r3': {
            'Neighbors': ['r2'],
            'ConnectedSF': [sf3]
        }
    }
    
    # 4. 創建環境
    env = OpticalNetworkEnvironment(
        network, service_farms, topology_map, max_requests=50
    )
    
    # 5. 創建 Agent
    K_max = 10
    state_dim = 7 + K_max * 5
    action_dim = K_max
    agent = OpticalNetworkAgent(state_dim, action_dim, learning_rate=1e-4)
    
    # 6. 訓練
    print("\n開始訓練...")
    num_episodes = 20
    
    for episode in range(num_episodes):
        stats = agent.train_episode(env, max_steps=50)
        
        print(f"Episode {episode+1:3d} | "
              f"Reward: {stats['episode_reward']:7.2f} | "
              f"Success Rate: {stats['success_rate']:6.2%} | "
              f"Steps: {stats['episode_length']:3d}")
        
        if (episode + 1) % 5 == 0:
            print("-" * 60)
    
    print("\n✅ 訓練完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()