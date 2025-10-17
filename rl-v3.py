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


@dataclass
class NetworkGraph:
    V: List[str]
    E: List[Tuple[str, str]]
    A: List[int]
    
    def __post_init__(self):
        self.num_nodes = len(self.V)
        self.num_edges = len(self.E)
        self.num_wavelengths = len(self.A)
        self.C_v = 100.0
        self.fiber_usage = {}
        self.edge_distances = {}
    
    def is_wavelength_available(self, e: Tuple[str, str], wavelength: int) -> bool:
        """Check if wavelength is available on edge at current time."""
        return self.fiber_usage.get((e, wavelength), 0) == 0
    
    def allocate_wavelength(self, e: Tuple[str, str], wavelength: int):
        """Mark wavelength as occupied."""
        self.fiber_usage[(e, wavelength)] = 1
    
    def release_wavelength(self, e: Tuple[str, str], wavelength: int):
        """Mark wavelength as available."""
        self.fiber_usage[(e, wavelength)] = 0


@dataclass
class Request:
    request_id: int
    o_i: str
    d_i: str
    b_i: float
    tau_i: float
    F_i: float
    M_i: float
    p_i: int
    arrival_time: float
    
    def required_wavelengths(self, C_v: float) -> int:
        return int(np.ceil(self.b_i / C_v))
    
    @property
    def release_time(self) -> float:
        return self.arrival_time + self.tau_i


@dataclass
class PathFeatures:
    z1_k: float
    z2_k: float
    z3_k: float
    z4_k: float
    z5_k: float
    
    @staticmethod
    def compute_z1(path_k: List, wavelength_set: Set, network: 'NetworkGraph') -> float:
        """Count available wavelengths along path."""
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
        """Find maximum contiguous available wavelengths."""
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
        return 1.0 if z2_value >= n_r else 0.0
    
    @staticmethod
    def compute_z4(path_delay: float) -> float:
        return path_delay
    
    @staticmethod
    def compute_z5(C_avail: float, F_i: float) -> float:
        return C_avail / F_i if F_i > 0 else 0.0


@dataclass
class ServiceFarm:
    sf_id: str
    
    def __init__(self, sf_id: str, C_max: float, M_max: float):
        self.sf_id = sf_id
        self.C_max_sf = C_max
        self.M_max_sf = M_max
        self.C_avail_sf = C_max
        self.M_avail_sf = M_max
        self.refill_count = 0
        self.total_allocations = 0
    
    def can_allocate(self, compute: float, memory: float) -> bool:
        return self.C_avail_sf >= compute and self.M_avail_sf >= memory
    
    def allocate_resources(self, compute: float, memory: float) -> bool:
        if not self.can_allocate(compute, memory):
            return False
        
        self.C_avail_sf -= compute
        self.M_avail_sf -= memory
        self.total_allocations += 1
        
        # Automatic refill mechanism
        refilled = False
        if self.M_avail_sf <= 0:
            logger.info(f"{self.sf_id}: Memory depleted, triggering refill")
            self.M_avail_sf = self.M_max_sf
            refilled = True
        
        if self.C_avail_sf <= 0:
            logger.info(f"{self.sf_id}: Compute depleted, triggering refill")
            self.C_avail_sf = self.C_max_sf
            refilled = True
        
        if refilled:
            self.refill_count += 1
        
        return True
    
    def get_utilization(self) -> Dict[str, float]:
        return {
            'compute': 1.0 - (self.C_avail_sf / self.C_max_sf),
            'memory': 1.0 - (self.M_avail_sf / self.M_max_sf)
        }


@dataclass
class WavelengthAssignment:
    edge: Tuple[str, str]
    wavelength: int
    release_time: float


@dataclass
class AllocatedResources:
    request_id: int
    wavelength_assignments: List[WavelengthAssignment]
    sf_allocations: Dict[str, Tuple[float, float]]
    path_edges: List[Tuple[str, str]]


@dataclass
class Action:
    k_i: int
    sf_i: str
    wavelength_start: int
    wavelength_end: int
    path_edges: List[Tuple[str, str]] = None
    sf_allocations: Dict[str, Tuple[float, float]] = None
    
    @property
    def contiguous_wavelength_set(self) -> Set[int]:
        return set(range(self.wavelength_start, self.wavelength_end + 1))
    
    def execute_action(self,
                      network: NetworkGraph,
                      request: Request,
                      sfs: List[ServiceFarm],
                      current_time: float) -> Tuple[bool, Optional[AllocatedResources]]:
        wavelength_set = self.contiguous_wavelength_set
        
        # Calculate processing time
        total_compute = sum(compute for compute, _ in self.sf_allocations.values())
        if total_compute == 0:
            return False, None
        
        processing_time = request.F_i / total_compute
        
        # Verify deadline constraint
        if processing_time > request.tau_i:
            logger.debug(f"Request {request.request_id}: Exceeds deadline")
            return False, None
        
        # Verify wavelength availability
        for edge in self.path_edges:
            for wavelength in wavelength_set:
                if not network.is_wavelength_available(edge, wavelength):
                    return False, None
        
        # Verify service farm resources
        if self.sf_allocations is None:
            return False, None
        
        sf_dict = {sf.sf_id: sf for sf in sfs}
        for sf_id, (compute, memory) in self.sf_allocations.items():
            if sf_id not in sf_dict:
                return False, None
            if not sf_dict[sf_id].can_allocate(compute, memory):
                return False, None
        
        # Allocate wavelengths (time-based)
        release_time = current_time + processing_time
        wavelength_assignments = []
        
        for edge in self.path_edges:
            for wavelength in wavelength_set:
                network.allocate_wavelength(edge, wavelength)
                wavelength_assignments.append(
                    WavelengthAssignment(edge, wavelength, release_time)
                )
        
        # Allocate service farm resources (consumable)
        for sf_id, (compute, memory) in self.sf_allocations.items():
            sf_dict[sf_id].allocate_resources(compute, memory)
        
        # Create allocation record
        allocation = AllocatedResources(
            request_id=request.request_id,
            wavelength_assignments=wavelength_assignments,
            sf_allocations=self.sf_allocations.copy(),
            path_edges=self.path_edges.copy()
        )
        
        logger.debug(f"Request {request.request_id}: Successfully allocated")
        return True, allocation


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
        """Calculate reward for allocation decision."""
        if not success:
            return -self.R_block
        
        # Delay efficiency
        n_delay = max(0.0, (tau_i - total_delay) / tau_i) if tau_i > 0 else 0.0
        
        # Power efficiency
        n_power = 1.0 - (P_used / P_max) if P_max > 0 else 0.0
        
        # Fragmentation penalty
        delta_phi = phi_t - phi_t_minus
        n_frag = np.exp(-self.upsilon * max(0.0, delta_phi))
        
        # Priority bonus
        g_p = request.p_i
        
        reward = (self.R_succ +
                 self.w_d * n_delay +
                 self.w_p * n_power +
                 self.w_f * n_frag +
                 self.w_pri * g_p)
        
        return reward


@dataclass
class CandidateInfo:
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
    
    def generate_candidates(self, request: Request) -> List[CandidateInfo]:
        # Filter available service farms
        available_sfs = [sf for sf in self.service_farms
                        if sf.C_avail_sf > 0 and sf.M_avail_sf > 0]
        
        if not available_sfs:
            logger.warning(f"Request {request.request_id}: No available service farms")
            return []
        
        # Calculate time constraint
        compute_min = min(sf.C_avail_sf for sf in available_sfs)
        T_compute = request.F_i / compute_min if compute_min > 0 else float('inf')
        
        if request.tau_i <= T_compute:
            logger.debug(f"Request {request.request_id}: Deadline too tight")
            return []
        
        T_limit = request.tau_i - T_compute
        
        # Calculate initial delay
        B = 50 * 1e9
        gOSNR = 20
        T_trans = request.b_i / (B * math.log2(1 + gOSNR))
        T_node = 0.0001
        initial_delay = T_trans + T_node
        
        # BFS delay expansion
        delay_dict = self._bfs_delay_expansion(request.o_i, T_limit, initial_delay)
        
        # Generate candidates
        candidates = []
        candidate_id = 0
        
        for target_node in self.network.V:
            if target_node == request.o_i:
                continue
            
            delay, prev = delay_dict.get(target_node, (float('inf'), None))
            if delay > T_limit or delay == float('inf'):
                continue
            
            # Get service farms at target node
            sfs_at_node = self._get_sfs_at_node(target_node)
            if not sfs_at_node:
                continue
            
            # Greedy resource allocation
            result = self._greedy_combine_sfs(sfs_at_node, request.F_i, request.M_i)
            if result is None:
                continue
            
            selected_sfs, sf_allocations, processing_time, power, memory = result
            
            # Reconstruct path
            path_nodes = self._reconstruct_path(request.o_i, target_node, delay_dict)
            path_edges = [(path_nodes[i], path_nodes[i+1])
                         for i in range(len(path_nodes)-1)]
            
            # Calculate cost
            alpha, beta = 1.0, 0.01
            cost = alpha * (delay + processing_time) + beta * power
            
            # Compute path features
            path_features = self._compute_path_features(path_edges, request)
            
            # Check wavelength availability
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
        
        # Sort by cost and return top K
        candidates.sort(key=lambda c: c.cost)
        return candidates[:self.k_candidates]
    
    def _bfs_delay_expansion(self,
                            start_node: str,
                            T_limit: float,
                            initial_delay: float) -> Dict[str, Tuple[float, Optional[str]]]:
        """BFS traversal with delay constraint."""
        LIGHT_SPEED = 2e5
        NODE_DELAY = 0.0001
        
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
                T_prop = distance / LIGHT_SPEED
                total_delay = current_delay + T_prop + NODE_DELAY
                
                if total_delay > T_limit:
                    continue
                
                if total_delay < delay_dict[neighbor][0]:
                    delay_dict[neighbor] = (total_delay, current_node)
                    queue.append((neighbor, total_delay))
        
        return delay_dict
    
    def _get_sfs_at_node(self, node: str) -> List[ServiceFarm]:
        connected = self.topology_map.get(node, {}).get('ConnectedSF', [])
        return [sf for sf in connected if isinstance(sf, ServiceFarm)]
    
    def _greedy_combine_sfs(self,
                           sfs: List[ServiceFarm],
                           F_i: float,
                           M_i: float) -> Optional[Tuple]:
        selected = []
        allocations = {}
        total_compute = 0
        total_memory = 0
        total_power = 0
        
        # Sort by available memory
        sorted_sfs = sorted(sfs, key=lambda sf: sf.M_avail_sf, reverse=True)
        
        for sf in sorted_sfs:
            compute_alloc = min(sf.C_avail_sf, max(0, F_i - total_compute))
            memory_alloc = min(sf.M_avail_sf, max(0, M_i - total_memory))
            
            if compute_alloc > 0 or memory_alloc > 0:
                selected.append(sf)
                allocations[sf.sf_id] = (compute_alloc, memory_alloc)
                total_compute += compute_alloc
                total_memory += memory_alloc
                total_power += 30
            
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
        path = []
        current = end
        
        while current != start and current is not None:
            path.insert(0, current)
            _, current = delay_dict.get(current, (float('inf'), None))
        
        path.insert(0, start)
        return path
    
    def _compute_path_features(self,
                              path_edges: List[Tuple[str, str]],
                              request: Request) -> PathFeatures:
        wavelength_set = set(self.network.A)
        
        z1 = PathFeatures.compute_z1(path_edges, wavelength_set, self.network)
        z2 = PathFeatures.compute_z2(path_edges, wavelength_set, self.network)
        n_r = request.required_wavelengths(self.network.C_v)
        z3 = PathFeatures.compute_z3(z2, n_r)
        
        # Path delay
        total_distance = sum(
            self.network.edge_distances.get(edge, 200)
            for edge in path_edges
        )
        z4 = total_distance / (2 * 10**5) * len(path_edges)
        
        # Resource availability
        z5 = 1.0
        
        return PathFeatures(z1, z2, z3, z4, z5)


@dataclass
class State:
    u_t: Request
    candidates: List[CandidateInfo]
    network_state: Dict = None
    current_time: float = 0.0
    
    def get_state_vector(self,
                        K_max: int = 10,
                        all_nodes: List[str] = None) -> np.ndarray:
        # Node encoding
        if all_nodes and self.u_t.o_i in all_nodes:
            o_idx = all_nodes.index(self.u_t.o_i) / len(all_nodes)
            d_idx = all_nodes.index(self.u_t.d_i) / len(all_nodes)
        else:
            o_idx = hash(self.u_t.o_i) % 1000 / 1000
            d_idx = hash(self.u_t.d_i) % 1000 / 1000
        
        # Request features (normalized)
        request_vec = np.array([
            o_idx,
            d_idx,
            self.u_t.b_i / 200.0,
            self.u_t.tau_i / 10.0,
            self.u_t.F_i / 1e9,
            self.u_t.M_i / 200.0,
            self.u_t.p_i / 10.0,
            self.u_t.arrival_time / 100.0
        ])
        
        # Candidate features (padded)
        candidate_vecs = []
        for i in range(K_max):
            if i < len(self.candidates):
                pf = self.candidates[i].path_features
                candidate_vecs.append([
                    pf.z1_k / 80.0,
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
    
    def __init__(self,
                 network: NetworkGraph,
                 service_farms: List[ServiceFarm],
                 topology_map: Dict,
                 max_requests: int = 100,
                 simulation_time: float = 1000.0,
                 mean_interarrival_time: float = 10.0):
        self.network = network
        self.service_farms = service_farms
        self.topology_map = topology_map
        self.max_requests = max_requests
        self.simulation_time = simulation_time
        self.mean_interarrival_time = mean_interarrival_time
        
        # Initialize components
        self.bfs_generator = BFSCandidateGenerator(
            network, service_farms, topology_map, k_candidates=5
        )
        self.reward_calculator = RewardCalculator()
        
        # State tracking
        self.current_time = 0.0
        self.request_queue = []
        self.current_request_idx = 0
        self.success_count = 0
        self.fail_count = 0
        self.phi_t_minus = 0.0
        
        # Resource tracking
        self.active_wavelength_assignments = []
        
        logger.info(f"Environment initialized: max_requests={max_requests}, "
                   f"simulation_time={simulation_time}s")
    
    def reset(self) -> State:
        """Reset environment to initial state."""
        # Clear network state
        self.network.fiber_usage.clear()
        
        # Reset service farms
        for sf in self.service_farms:
            sf.C_avail_sf = sf.C_max_sf
            sf.M_avail_sf = sf.M_max_sf
            sf.refill_count = 0
            sf.total_allocations = 0
        
        # Reset metrics
        self.current_time = 0.0
        self.current_request_idx = 0
        self.success_count = 0
        self.fail_count = 0
        self.phi_t_minus = 0.0
        self.active_wavelength_assignments.clear()
        
        # Generate request queue
        self.request_queue = self._generate_request_queue()
        
        if not self.request_queue:
            raise ValueError("Failed to generate requests")
        
        first_request = self.request_queue[0]
        self.current_time = first_request.arrival_time
        
        # Generate initial candidates
        candidates = self.bfs_generator.generate_candidates(first_request)
        
        logger.info(f"Environment reset: {len(self.request_queue)} requests generated")
        
        return State(
            u_t=first_request,
            candidates=candidates,
            current_time=self.current_time
        )
    
    def step(self,
             action_index: int,
             state: State) -> Tuple[State, float, bool, Dict]:
        """
        Execute one environment step.
        
        Returns:
            (next_state, reward, done, info)
        """
        request = state.u_t
        candidates = state.candidates
        self.current_time = request.arrival_time
        
        # Release expired wavelengths
        num_released = self._release_expired_wavelengths(self.current_time)
        
        # Execute action
        if action_index < 0 or action_index >= len(candidates):
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
                
                # Calculate reward
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
        
        # Advance to next request
        self.current_request_idx += 1
        
        # Check if done
        if self.current_request_idx >= len(self.request_queue):
            done = True
            next_state = state
            info = self._get_info(success, num_released, 0)
            return next_state, reward, done, info
        
        # Generate next state
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
        # Find contiguous wavelengths
        n_r = request.required_wavelengths(self.network.C_v)
        wavelength_range = self._find_continuous_wavelengths(
            candidate.path_edges, n_r
        )
        
        if wavelength_range is None:
            return False, None
        
        # Create and execute action
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
        """Find contiguous available wavelengths."""
        # Get available wavelengths
        available = []
        for w in self.network.A:
            if all(self.network.is_wavelength_available(edge, w)
                  for edge in path_edges):
                available.append(w)
        
        if len(available) < n_r:
            return None
        
        # Find contiguous segment
        for i in range(len(available) - n_r + 1):
            is_contiguous = all(
                available[i+j+1] == available[i+j] + 1
                for j in range(n_r - 1)
            )
            if is_contiguous:
                return available[i], available[i + n_r - 1]
        
        return None
    
    def _release_expired_wavelengths(self, current_time: float) -> int:
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
            logger.debug(f"Released {count} wavelengths at t={current_time:.2f}s")
        
        return count
    
    def _compute_fragmentation(self) -> float:
        total_segments = 0
        total_available = 0
        
        for edge in self.network.E:
            available = [
                w for w in self.network.A
                if self.network.is_wavelength_available(edge, w)
            ]
            
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
        total = self.success_count + self.fail_count
        return {
            'success': success,
            'success_rate': self.success_count / total if total > 0 else 0.0,
            'num_candidates': num_candidates,
            'active_wavelengths': len(self.active_wavelength_assignments),
            'current_time': self.current_time,
            'wavelengths_released': released
        }


class PolicyNetwork(nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.network(state)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    
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
        
        logger.info(f"Agent initialized: state_dim={state_dim}, "
                   f"action_dim={action_dim}, lr={learning_rate}")
    
    def select_action(self,
                     state: State,
                     num_valid_candidates: int) -> Tuple[int, torch.Tensor]:

        state_vec = state.get_state_vector()
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        
        with torch.no_grad():
            probs = self.policy_net(state_tensor)
        
        # Mask invalid actions
        if num_valid_candidates < probs.shape[1]:
            mask = torch.zeros_like(probs)
            mask[0, num_valid_candidates:] = float('-inf')
            probs = F.softmax(probs + mask, dim=-1)
        
        # Sample action
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def update(self, trajectory: Tuple) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = trajectory
        
        if len(states) == 0:
            return {}
        
        # Convert to tensors
        states_t = torch.FloatTensor([s.get_state_vector() for s in states])
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor([s.get_state_vector() for s in next_states])
        dones_t = torch.FloatTensor(dones)
        
        # Calculate advantages
        with torch.no_grad():
            values = self.value_net(states_t).squeeze()
            next_values = self.value_net(next_states_t).squeeze()
            next_values = next_values * (1 - dones_t)
            advantages = rewards_t + self.gamma * next_values - values
        
        # Policy loss
        probs = self.policy_net(states_t)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()
        
        policy_loss = -(advantages.detach() * log_probs).mean()
        policy_loss -= self.entropy_coef * entropy
        
        # Value loss
        predicted_values = self.value_net(states_t).squeeze()
        target_values = rewards_t + self.gamma * next_values
        value_loss = F.mse_loss(predicted_values, target_values.detach())
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def train_episode(self,
                     env: OpticalNetworkEnvironment,
                     max_steps: Optional[int] = None) -> Dict[str, float]:
        """Execute one training episode."""
        state = env.reset()
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        episode_reward = 0.0
        step = 0
        
        while True:
            num_candidates = len(state.candidates)
            if num_candidates == 0:
                action = -1
            else:
                action, _ = self.select_action(state, num_candidates)
            
            next_state, reward, done, info = env.step(action, state)
            
            states.append(state)
            actions.append(max(0, action))
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            episode_reward += reward
            step += 1
            state = next_state
            
            if done or (max_steps and step >= max_steps):
                break
        
        losses = self.update((states, actions, rewards, next_states, dones))
        
        return {
            'episode_reward': episode_reward,
            'episode_length': step,
            'success_rate': info.get('success_rate', 0.0),
            **losses
        }


def main():
    """Main training procedure."""
    logger.info("="*70)
    logger.info("Deep RL for Optical Network Resource Allocation")
    logger.info("="*70)
    
    # Initialize network topology
    logger.info("Initializing network topology...")
    nodes = ['r1', 'r2', 'r3']
    edges = [('r1', 'r2'), ('r2', 'r1'), ('r2', 'r3'), ('r3', 'r2')]
    wavelengths = list(range(1, 21))
    
    network = NetworkGraph(V=nodes, E=edges, A=wavelengths)
    network.edge_distances = {
        ('r1', 'r2'): 200, ('r2', 'r1'): 200,
        ('r2', 'r3'): 300, ('r3', 'r2'): 300
    }
    
    # Initialize service farms
    logger.info("Initializing service farms...")
    sf1 = ServiceFarm('SF1', C_max=1e8, M_max=100)
    sf2 = ServiceFarm('SF2', C_max=2e8, M_max=150)
    sf3 = ServiceFarm('SF3', C_max=1.5e8, M_max=120)
    service_farms = [sf1, sf2, sf3]
    
    # Topology mapping
    topology_map = {
        'r1': {'Neighbors': ['r2'], 'ConnectedSF': [sf1]},
        'r2': {'Neighbors': ['r1', 'r3'], 'ConnectedSF': [sf2]},
        'r3': {'Neighbors': ['r2'], 'ConnectedSF': [sf3]}
    }
    
    # Create environment
    logger.info("Creating environment...")
    env = OpticalNetworkEnvironment(
        network=network,
        service_farms=service_farms,
        topology_map=topology_map,
        max_requests=30,
        simulation_time=300.0,
        mean_interarrival_time=10.0
    )
    
    # Create agent
    logger.info("Creating agent...")
    K_max = 10
    state_dim = 8 + K_max * 5
    action_dim = K_max
    agent = OpticalNetworkAgent(state_dim, action_dim, learning_rate=1e-4)
    
    # Training loop
    logger.info("="*70)
    logger.info("Starting training...")
    logger.info("="*70)
    
    num_episodes = 10
    
    for episode in range(num_episodes):
        logger.info(f"\nEpisode {episode+1}/{num_episodes}")
        logger.info("-"*70)
        
        stats = agent.train_episode(env)
        
        logger.info(f"Results:")
        logger.info(f"  Reward: {stats['episode_reward']:.2f}")
        logger.info(f"  Success Rate: {stats['success_rate']:.2%}")
        logger.info(f"  Length: {stats['episode_length']}")
        
        if 'policy_loss' in stats:
            logger.info(f"  Policy Loss: {stats['policy_loss']:.4f}")
            logger.info(f"  Value Loss: {stats['value_loss']:.4f}")
            logger.info(f"  Entropy: {stats['entropy']:.4f}")
        
        # Resource statistics
        logger.info("Resource Status:")
        for sf in service_farms:
            util = sf.get_utilization()
            logger.info(f"  {sf.sf_id}: Compute={util['compute']:.1%}, "
                       f"Memory={util['memory']:.1%}, Refills={sf.refill_count}")
    
    logger.info("="*70)
    logger.info("Training completed")
    logger.info("="*70)


if __name__ == "__main__":
    main()