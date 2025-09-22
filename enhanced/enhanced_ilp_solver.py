#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from pulp import *
import pickle
from collections import Counter
from networkx.algorithms.simple_paths import all_simple_paths
from pathlib import Path

if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul 2>&1')

# ====== Alpha/Beta係數設定 (與強化學習一致) ======
ALPHA = 0.6  # 延遲權重 - 可調整
BETA = 0.4   # 功率權重 - 可調整

print(f"BFS Alpha/Beta Weighting: α={ALPHA}, β={BETA}")

# ====== 基本參數 ======
N = 8
K_PATHS = 3
MAX_WAVELENGTH = 10
DEBUG = False  # 關閉詳細debug
CUTOFF_LEN = 6

# 功率參數
POWER_BASE = 10.0
POWER_PER_HOP = 5.0
POWER_PER_WAVELENGTH = 2.0

# 路徑設定
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "tms_8nodes_enhanced"
OUTPUT_CSV = BASE_DIR / "bfs_alpha_beta_dataset_all_topologies.csv"
LABEL_MAP_FILE = BASE_DIR / "bfs_alpha_beta_label_mappings.pkl"
TOPOLOGY_FILE = BASE_DIR / "topology_info.pkl"

# ====== 載入拓撲資訊 ======
try:
    with open(str(TOPOLOGY_FILE), 'rb') as f:
        topology_info = pickle.load(f)
    topology_names = sorted(topology_info.keys())
    topo_encoding = {name: idx for idx, name in enumerate(topology_names)}
    print(f"Loaded {len(topology_names)} topologies: {list(topology_names)}")
except FileNotFoundError:
    print("Error: Run enhanced_datamake.py first")
    sys.exit(1)

# ====== 功率計算函數 ======
def calculate_path_power(path, G, wavelength):
    if len(path) <= 1:
        return POWER_BASE
    hop_count = len(path) - 1
    return POWER_BASE + (hop_count * POWER_PER_HOP) + (wavelength * POWER_PER_WAVELENGTH)

def calculate_total_power_consumption(solution_vars, paths, G):
    total_power = 0
    for (s, d), path_list in paths.items():
        for p_id, path in enumerate(path_list):
            for w in range(MAX_WAVELENGTH):
                if solution_vars.get((s, d, p_id, w)) and solution_vars[(s, d, p_id, w)].varValue == 1:
                    total_power += calculate_path_power(path, G, w)
    return total_power

def calculate_total_delay_cost(solution_vars, paths, G, tm):
    total_delay_cost = 0
    for (s, d), path_list in paths.items():
        for p_id, path in enumerate(path_list):
            for w in range(MAX_WAVELENGTH):
                if solution_vars.get((s, d, p_id, w)) and solution_vars[(s, d, p_id, w)].varValue == 1:
                    path_delay_cost = sum(G[u][v]['length'] for u, v in zip(path, path[1:]))
                    wavelength_delay_cost = tm[s][d] * path_delay_cost * ((w + 1)**2)
                    total_delay_cost += wavelength_delay_cost
    return total_delay_cost

# ====== 標準分配解編碼 ======
def encode_solution(solution_vars, paths, demand_list):
    solution_encoding = []
    for s, d in demand_list:
        assigned_path = -1
        assigned_wavelength = -1
        assigned_power = 0.0
        
        if (s, d) in paths:
            path_list = paths[(s, d)]
            for p_id in range(len(path_list)):
                for w in range(MAX_WAVELENGTH):
                    if (s, d, p_id, w) in solution_vars and solution_vars[(s, d, p_id, w)].varValue == 1:
                        assigned_path = p_id
                        assigned_wavelength = w
                        path = path_list[p_id]
                        assigned_power = calculate_path_power(path, None, w)
                        break
                if assigned_path != -1:
                    break
        
        solution_encoding.extend([assigned_path, assigned_wavelength, assigned_power])
    return np.array(solution_encoding)

# ====== 路徑生成 ======
def k_shortest_paths(graph, source, target, k, cutoff=CUTOFF_LEN):
    try:
        all_paths = list(all_simple_paths(graph, source, target, cutoff=cutoff))
        all_paths = sorted(all_paths, key=lambda p: sum(graph[u][v]['length'] for u, v in zip(p, p[1:])))
        return all_paths[:k]
    except:
        return []

def generate_paths_for_topology(G):
    paths = {}
    demand_list = []
    for s in range(N):
        for d in range(N):
            if s != d:
                path_list = k_shortest_paths(G, s, d, K_PATHS)
                paths[(s, d)] = path_list
                if path_list:
                    demand_list.append((s, d))
    return paths, demand_list

def path_cost(path, G):
    return sum(G[u][v]['length'] for u, v in zip(path, path[1:]))

# ====== 主要處理迴圈 ======
all_X_data, all_Y_data = [], []
global_solution_stats = Counter()
topo_stats = {}

for topo_name, topo_data in topology_info.items():
    print(f"\nProcessing {topo_name.upper()}...")
    
    G = topo_data['graph']
    topo_code = topo_encoding[topo_name]
    paths, demand_list = generate_paths_for_topology(G)
    
    topo_dir = INPUT_DIR / topo_name
    if not topo_dir.exists():
        continue
    
    topo_files = sorted([f for f in topo_dir.iterdir() if f.suffix == '.csv'])
    infeasible_count = 0
    processed_count = 0
    
    for idx, tm_file in enumerate(topo_files):
        try:
            tm = pd.read_csv(str(tm_file), header=None, encoding='utf-8').values
            flat_tm = tm.flatten()
        except:
            continue
        
        # ILP模型
        model = LpProblem(f"BFS_AlphaBeta_RWA_ILP_{topo_name}", LpMinimize)
        x = {}
        
        # 決策變數
        for (s, d), path_list in paths.items():
            for p_id, path in enumerate(path_list):
                for w in range(MAX_WAVELENGTH):
                    x[(s, d, p_id, w)] = LpVariable(f"x_{s}_{d}_{p_id}_{w}", 0, 1, LpBinary)
        
        # Alpha/Beta加權目標函數
        delay_cost_terms = []
        power_cost_terms = []
        
        for (s, d), path_list in paths.items():
            if not path_list:
                continue
            for p_id, path in enumerate(path_list):
                for w in range(MAX_WAVELENGTH):
                    delay_cost = tm[s][d] * path_cost(path, G) * ((w + 1)**2)
                    power_cost = tm[s][d] * calculate_path_power(path, G, w)
                    delay_cost_terms.append(delay_cost * x[(s, d, p_id, w)])
                    power_cost_terms.append(power_cost * x[(s, d, p_id, w)])
        
        if delay_cost_terms and power_cost_terms:
            model += ALPHA * lpSum(delay_cost_terms) + BETA * lpSum(power_cost_terms)
        elif delay_cost_terms:
            model += lpSum(delay_cost_terms)
        else:
            continue
        
        # 約束條件
        for (s, d), path_list in paths.items():
            if not path_list:
                continue
            if tm[s][d] > 0:
                model += lpSum(x[(s, d, p_id, w)] for p_id in range(len(path_list)) for w in range(MAX_WAVELENGTH)) == 1
            else:
                for p_id in range(len(path_list)):
                    for w in range(MAX_WAVELENGTH):
                        model += x[(s, d, p_id, w)] == 0
        
        for u, v in G.edges():
            for w in range(MAX_WAVELENGTH):
                conflict_vars = []
                for (s, d), path_list in paths.items():
                    for p_id, path in enumerate(path_list):
                        if len(path) > 1 and (u, v) in zip(path, path[1:]):
                            conflict_vars.append(x[(s, d, p_id, w)])
                if conflict_vars:
                    model += lpSum(conflict_vars) <= 1
        
        # 求解
        try:
            model.solve(PULP_CBC_CMD(msg=0))
        except:
            continue
        
        # 處理結果
        if LpStatus[model.status] == 'Optimal':
            solution_label = encode_solution(x, paths, demand_list)
            total_power = calculate_total_power_consumption(x, paths, G)
            total_delay_cost = calculate_total_delay_cost(x, paths, G, tm)
            alpha_beta_cost = ALPHA * total_delay_cost + BETA * total_power
            
            # Label-as-Feature: 提取標籤統計特徵
            used_wavelengths = set()
            total_path_hops = 0
            num_assigned_demands = 0
            avg_wavelength_per_demand = 0
            
            for (s, d), path_list in paths.items():
                for p_id, path in enumerate(path_list):
                    for w in range(MAX_WAVELENGTH):
                        if x.get((s, d, p_id, w)) and x[(s, d, p_id, w)].varValue == 1:
                            used_wavelengths.add(w)
                            total_path_hops += len(path) - 1 if len(path) > 1 else 0
                            num_assigned_demands += 1
                            avg_wavelength_per_demand += w
            
            num_wavelengths_used = len(used_wavelengths)
            avg_path_length = total_path_hops / num_assigned_demands if num_assigned_demands > 0 else 0
            avg_wavelength_index = avg_wavelength_per_demand / num_assigned_demands if num_assigned_demands > 0 else 0
            demand_coverage_ratio = num_assigned_demands / len(demand_list) if demand_list else 0
            
            # 71維特徵向量 (Label-as-Feature)
            enhanced_features = np.concatenate([
                flat_tm,                      # 64維TM
                [topo_code],                  # 1維拓撲
                [alpha_beta_cost],            # 1維α/β成本
                [num_wavelengths_used],       # 標籤衍生特徵
                [avg_path_length],
                [avg_wavelength_index],
                [demand_coverage_ratio],
                [total_power / 1000.0]
            ])
            
            all_X_data.append(enhanced_features)
            all_Y_data.append(solution_label)
            processed_count += 1
            
        else:
            infeasible_count += 1
    
    topo_stats[topo_name] = {
        'processed': processed_count,
        'infeasible': infeasible_count,
        'total_files': len(topo_files)
    }
    
    print(f"  Processed: {processed_count}, Infeasible: {infeasible_count}")

if not all_X_data:
    print("Error: No valid data processed")
    sys.exit(1)

# ====== 儲存結果 ======
try:
    df = pd.DataFrame(all_X_data)
    solution_dim = len(all_Y_data[0]) if all_Y_data else 0
    demand_count = solution_dim // 3
    
    label_columns = []
    for d in range(demand_count):
        label_columns.extend([f'demand_{d}_path', f'demand_{d}_wavelength', f'demand_{d}_power'])
    
    for i, col_name in enumerate(label_columns):
        if i < solution_dim:
            df[col_name] = [label[i] for label in all_Y_data]
    
    df.to_csv(str(OUTPUT_CSV), index=False, encoding='utf-8')
    print(f"\nDataset saved: {OUTPUT_CSV}")
    print(f"Samples: {len(all_Y_data)}, Features: 71D, Labels: {solution_dim}D")
except Exception as e:
    print(f"Save error: {e}")
    sys.exit(1)

# 儲存標籤映射
bfs_alpha_beta_label_info = {
    'label_type': 'bfs_alpha_beta_weighted_solution',
    'solution_encoding': 'path_wavelength_power_triplets_per_demand',
    'solution_dimensions': solution_dim,
    'demand_count': demand_count,
    'dimensions_per_demand': 3,
    'topology_encoding': topo_encoding,
    'feature_dimensions': {'tm_size': 64, 'topo_size': 1, 'alpha_beta_cost_size': 1, 'label_derived_size': 5},
    'power_parameters': {
        'power_base': POWER_BASE,
        'power_per_hop': POWER_PER_HOP,
        'power_per_wavelength': POWER_PER_WAVELENGTH
    },
    'bfs_weighting_coefficients': {
        'alpha': ALPHA,
        'beta': BETA
    },
    'objective_function': f'{ALPHA} × delay_cost + {BETA} × power_cost',
    'total_samples': len(all_Y_data),
    'unique_solutions': len(global_solution_stats),
    'topology_names': topology_names
}

with open(str(LABEL_MAP_FILE), 'wb') as f:
    pickle.dump(bfs_alpha_beta_label_info, f)

print(f"Label mapping saved: {LABEL_MAP_FILE}")
print("BFS Alpha/Beta + Label-as-Feature system ready!")
print(f"Alpha={ALPHA}, Beta={BETA} - Use same coefficients for RL comparison")