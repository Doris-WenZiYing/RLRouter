#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import joblib
import sys
import os
from pathlib import Path
from tensorflow.keras.models import load_model
import networkx as nx

if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul 2>&1')

# ====== 配置 ======
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "bfs_alpha_beta_training_results" / "bfs_alpha_beta_dnn_model.keras"
SCALER_PATH = BASE_DIR / "bfs_alpha_beta_scaler.pkl"
LABEL_MAP_PATH = BASE_DIR / "bfs_alpha_beta_label_mappings.pkl"
TOPOLOGY_INFO_PATH = BASE_DIR / "topology_info.pkl"
RESULTS_DIR = BASE_DIR / "bfs_alpha_beta_prediction_results"

RESULTS_DIR.mkdir(exist_ok=True)

class BFSAlphaBetaRWAPredictor:
    """BFS Alpha/Beta 加權的路由波長分配預測器"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_info = None
        self.topology_info = None
        self.topo_encoding = {}
        self.reverse_topo = {}
        self.alpha = 0.6
        self.beta = 0.4
        
    def load_model_components(self):
        """載入所有模型組件"""
        try:
            # 載入模型
            self.model = load_model(str(MODEL_PATH))
            
            # 載入特徵標準化器
            self.scaler = joblib.load(str(SCALER_PATH))
            
            # 載入標籤映射
            with open(str(LABEL_MAP_PATH), 'rb') as f:
                self.label_info = pickle.load(f)
            
            self.topo_encoding = self.label_info['topology_encoding']
            self.reverse_topo = {code: name for name, code in self.topo_encoding.items()}
            self.alpha = self.label_info['bfs_weighting_coefficients']['alpha']
            self.beta = self.label_info['bfs_weighting_coefficients']['beta']
            
            # 載入拓撲資訊
            with open(str(TOPOLOGY_INFO_PATH), 'rb') as f:
                self.topology_info = pickle.load(f)
            
            print(f"Model loaded: Alpha={self.alpha}, Beta={self.beta}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def calculate_path_power(self, path, G, wavelength):
        """計算路徑功率消耗"""
        power_params = self.label_info['power_parameters']
        base_power = power_params['power_base']
        power_per_hop = power_params['power_per_hop']
        power_per_wavelength = power_params['power_per_wavelength']
        
        if len(path) <= 1:
            return base_power
        
        hop_count = len(path) - 1
        return base_power + (hop_count * power_per_hop) + (wavelength * power_per_wavelength)
    
    def generate_paths_for_topology(self, topology_name):
        """為指定拓撲生成路徑"""
        if topology_name not in self.topology_info:
            raise ValueError(f"Unknown topology: {topology_name}")
        
        G = self.topology_info[topology_name]['graph']
        paths = {}
        demand_list = []
        
        from networkx.algorithms.simple_paths import all_simple_paths
        
        for s in range(8):
            for d in range(8):
                if s != d:
                    try:
                        all_paths = list(all_simple_paths(G, s, d, cutoff=6))
                        all_paths = sorted(all_paths, key=lambda p: sum(G[u][v]['length'] for u, v in zip(p, p[1:])))
                        path_list = all_paths[:3]
                        
                        if path_list:
                            paths[(s, d)] = path_list
                            demand_list.append((s, d))
                    except:
                        continue
        
        return paths, demand_list
    
    def create_bfs_alpha_beta_features(self, tm, topology_name):
        """建立BFS Alpha/Beta特徵向量 (71維)"""
        flat_tm = tm.flatten()
        
        if topology_name not in self.topo_encoding:
            raise ValueError(f"Unknown topology: {topology_name}")
        
        topo_code = self.topo_encoding[topology_name]
        
        # 估算Alpha/Beta加權成本
        G = self.topology_info[topology_name]['graph']
        estimated_alpha_beta_cost = self.estimate_alpha_beta_cost(tm, G)
        
        # 估算標籤衍生特徵
        estimated_wavelengths = self.estimate_feature_stats(tm, G, 'wavelengths')
        estimated_path_length = self.estimate_feature_stats(tm, G, 'path_length')
        estimated_wavelength_idx = self.estimate_feature_stats(tm, G, 'wavelength_idx')
        estimated_coverage = self.estimate_feature_stats(tm, G, 'coverage')
        estimated_power = self.estimate_feature_stats(tm, G, 'power')
        
        # 71維特徵：64TM + 1拓撲 + 1α/β成本 + 5標籤衍生
        enhanced_features = np.concatenate([
            flat_tm,                      # 64維TM
            [topo_code],                  # 1維拓撲
            [estimated_alpha_beta_cost],  # 1維α/β成本
            [estimated_wavelengths],      # 標籤衍生特徵
            [estimated_path_length],
            [estimated_wavelength_idx],
            [estimated_coverage],
            [estimated_power]
        ])
        
        return enhanced_features.reshape(1, -1)
    
    def estimate_alpha_beta_cost(self, tm, G):
        """估算Alpha/Beta加權成本"""
        total_cost = 0
        for s in range(8):
            for d in range(8):
                if s != d and tm[s][d] > 0:
                    try:
                        shortest_path_length = nx.shortest_path_length(G, s, d, weight='length')
                        delay_cost = tm[s][d] * shortest_path_length * 4  # 假設波長2
                        power_cost = tm[s][d] * (10 + shortest_path_length * 5 + 2 * 2)  # 基本功率估算
                        total_cost += self.alpha * delay_cost + self.beta * power_cost
                    except:
                        continue
        return total_cost
    
    def estimate_feature_stats(self, tm, G, stat_type):
        """估算標籤衍生特徵"""
        if stat_type == 'wavelengths':
            return 5.0  # 估算使用5個波長
        elif stat_type == 'path_length':
            return 2.5  # 估算平均路徑長度
        elif stat_type == 'wavelength_idx':
            return 2.0  # 估算平均波長索引
        elif stat_type == 'coverage':
            return 0.8  # 估算80%覆蓋率
        elif stat_type == 'power':
            return 500.0  # 估算總功率
        return 0
    
    def predict_rwa_solution(self, tm, topology_name):
        """預測路由波長分配解決方案"""
        # 建立71維特徵
        enhanced_features = self.create_bfs_alpha_beta_features(tm, topology_name)
        
        # 特徵標準化 (與訓練時一致)
        tm_features = enhanced_features[:, :64]
        topo_features = enhanced_features[:, 64:65]
        alpha_beta_cost = enhanced_features[:, 65:66]
        label_derived_features = enhanced_features[:, 66:71]
        
        # 標準化數值型特徵
        X_to_scale = np.concatenate([tm_features, alpha_beta_cost, label_derived_features], axis=1)
        X_scaled_part = self.scaler.transform(X_to_scale)
        
        # 重新組合
        X_scaled = np.concatenate([
            X_scaled_part[:, :64],      # 標準化TM
            topo_features,              # 原始拓撲
            X_scaled_part[:, 64:65],    # 標準化α/β成本
            X_scaled_part[:, 65:70]     # 標準化標籤衍生特徵
        ], axis=1)
        
        # 預測
        raw_prediction = self.model.predict(X_scaled, verbose=0)[0]
        
        # 解碼預測結果
        decoded_solution = self.decode_prediction(raw_prediction, topology_name)
        
        return {
            'raw_prediction': raw_prediction,
            'decoded_solution': decoded_solution,
            'input_topology': topology_name,
            'weighting': {'alpha': self.alpha, 'beta': self.beta}
        }
    
    def decode_prediction(self, raw_prediction, topology_name):
        """解碼預測結果為標準分配解 (path, wavelength, power 三元組)"""
        paths, demand_list = self.generate_paths_for_topology(topology_name)
        
        # 每個demand有3個值：path_id, wavelength, power
        demand_count = len(demand_list)
        expected_dim = demand_count * 3
        
        if len(raw_prediction) != expected_dim:
            print(f"Warning: Prediction dimension mismatch. Expected {expected_dim}, got {len(raw_prediction)}")
            return None
        
        prediction_triplets = raw_prediction.reshape(demand_count, 3)
        
        assignments = {}
        total_power = 0
        max_wavelength = -1
        
        G = self.topology_info[topology_name]['graph']
        
        for i, (s, d) in enumerate(demand_list):
            if (s, d) in paths:
                path_list = paths[(s, d)]
                
                # 解碼：path_id, wavelength, power
                pred_path_id = max(0, min(len(path_list)-1, int(round(prediction_triplets[i, 0]))))
                pred_wavelength = max(0, min(9, int(round(prediction_triplets[i, 1]))))
                pred_power = max(0, prediction_triplets[i, 2])
                
                if pred_path_id < len(path_list):
                    assigned_path = path_list[pred_path_id]
                    
                    assignments[(s, d)] = {
                        'path': assigned_path,
                        'wavelength': pred_wavelength,
                        'path_id': pred_path_id,
                        'predicted_power': pred_power
                    }
                    
                    # 實際功率計算
                    actual_power = self.calculate_path_power(assigned_path, G, pred_wavelength)
                    total_power += actual_power
                    
                    if pred_wavelength > max_wavelength:
                        max_wavelength = pred_wavelength
        
        return {
            'assignments': assignments,
            'total_power': total_power,
            'max_wavelength': max_wavelength,
            'demand_count': len(assignments),
            'total_demands': len(demand_list)
        }
    
    def validate_solution(self, decoded_solution, topology_name):
        """驗證解決方案可行性"""
        if not decoded_solution:
            return {'valid': False, 'conflicts': ['Invalid solution']}
        
        assignments = decoded_solution['assignments']
        edge_wavelength_usage = {}
        conflicts = []
        
        for (s, d), assignment in assignments.items():
            path = assignment['path']
            wavelength = assignment['wavelength']
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                
                if (u, v) not in edge_wavelength_usage:
                    edge_wavelength_usage[(u, v)] = {}
                
                if wavelength in edge_wavelength_usage[(u, v)]:
                    conflicts.append(f"Wavelength conflict: edge({u},{v}) wavelength {wavelength}")
                else:
                    edge_wavelength_usage[(u, v)][wavelength] = (s, d)
        
        return {
            'valid': len(conflicts) == 0,
            'conflicts': conflicts,
            'wavelength_usage': decoded_solution['max_wavelength'] + 1 if 'max_wavelength' in decoded_solution else 0
        }
    
    def evaluate_solution_quality(self, tm, decoded_solution, topology_name):
        """評估解決方案質量"""
        if not decoded_solution:
            return {'quality_score': 0, 'error': 'Invalid solution'}
        
        validation = self.validate_solution(decoded_solution, topology_name)
        if not validation['valid']:
            return {'quality_score': 0, 'error': f"Conflicts: {len(validation['conflicts'])}"}
        
        assignments = decoded_solution['assignments']
        G = self.topology_info[topology_name]['graph']
        
        total_delay_cost = 0
        total_power_cost = 0
        
        for (s, d), assignment in assignments.items():
            demand = tm[s][d]
            path = assignment['path']
            wavelength = assignment['wavelength']
            
            # 延遲成本
            path_cost = sum(G[u][v]['length'] for u, v in zip(path, path[1:]))
            delay_cost = demand * path_cost * ((wavelength + 1)**2)
            total_delay_cost += delay_cost
            
            # 功率成本
            power_cost = demand * self.calculate_path_power(path, G, wavelength)
            total_power_cost += power_cost
        
        # BFS Alpha/Beta加權總成本
        weighted_cost = self.alpha * total_delay_cost + self.beta * total_power_cost
        quality_score = 1.0 / (1.0 + weighted_cost / 1000)
        
        return {
            'quality_score': quality_score,
            'weighted_cost': weighted_cost,
            'delay_cost': total_delay_cost,
            'power_cost': total_power_cost,
            'coverage': decoded_solution['demand_count'] / decoded_solution['total_demands']
        }
    
    def predict_from_file(self, tm_file_path, topology_name):
        """從TM文件預測RWA解決方案"""
        print(f"Predicting: {tm_file_path} on {topology_name.upper()}")
        
        # 載入TM
        tm_path = Path(tm_file_path)
        if not tm_path.exists():
            raise FileNotFoundError(f"TM file not found: {tm_path}")
        
        try:
            tm = pd.read_csv(str(tm_path), header=None, encoding='utf-8').values
        except:
            tm = pd.read_csv(str(tm_path), header=None, encoding='latin1').values
        
        # 預測
        result = self.predict_rwa_solution(tm, topology_name)
        
        # 驗證和評估
        decoded = result['decoded_solution']
        if decoded:
            validation = self.validate_solution(decoded, topology_name)
            quality = self.evaluate_solution_quality(tm, decoded, topology_name)
            
            print(f"Result: Valid={validation['valid']}, Quality={quality['quality_score']:.3f}, "
                  f"Power={decoded['total_power']:.1f}mW, MaxWL={decoded['max_wavelength']}")
            
            result.update({
                'validation': validation,
                'quality': quality
            })
        else:
            print(f"Error: Failed to decode solution")
        
        return result
    
    def test_multiple_samples(self, topology_name, num_samples=10):
        """測試多個樣本"""
        print(f"\nTesting {topology_name.upper()} topology ({num_samples} samples):")
        
        base_dir = BASE_DIR / "tms_8nodes_enhanced" / topology_name
        if not base_dir.exists():
            print(f"Error: Directory not found: {base_dir}")
            return None
        
        results = []
        for i in range(num_samples):
            tm_path = base_dir / f"tm_{i:04d}.csv"
            if tm_path.exists():
                try:
                    result = self.predict_from_file(tm_path, topology_name)
                    if result['decoded_solution']:
                        results.append(result)
                except Exception as e:
                    print(f"Error testing {tm_path}: {e}")
        
        if results:
            valid_results = [r for r in results if r.get('validation', {}).get('valid', False)]
            avg_quality = np.mean([r['quality']['quality_score'] for r in valid_results]) if valid_results else 0
            validity_rate = len(valid_results) / len(results)
            avg_power = np.mean([r['decoded_solution']['total_power'] for r in valid_results]) if valid_results else 0
            
            print(f"Summary: {len(valid_results)}/{len(results)} valid, "
                  f"Avg Quality: {avg_quality:.3f}, Avg Power: {avg_power:.1f}mW")
            
            return {
                'tested': len(results),
                'valid': len(valid_results),
                'validity_rate': validity_rate,
                'avg_quality': avg_quality,
                'avg_power': avg_power
            }
        else:
            print("No valid results")
            return None

def main():
    """主要執行函數"""
    print("BFS Alpha/Beta RWA Predictor")
    print("="*40)
    
    predictor = BFSAlphaBetaRWAPredictor()
    
    if not predictor.load_model_components():
        print("Error: Failed to load model components")
        return
    
    # 測試所有拓撲
    print(f"\nTesting all topologies:")
    overall_results = {}
    
    for topo_name in predictor.topo_encoding.keys():
        result = predictor.test_multiple_samples(topo_name, num_samples=5)
        if result:
            overall_results[topo_name] = result
    
    # 總結
    if overall_results:
        print(f"\nOverall Summary:")
        total_valid = sum(r['valid'] for r in overall_results.values())
        total_tested = sum(r['tested'] for r in overall_results.values())
        overall_validity = total_valid / total_tested if total_tested > 0 else 0
        
        print(f"Total validity: {overall_validity:.2%} ({total_valid}/{total_tested})")
        print(f"Alpha={predictor.alpha}, Beta={predictor.beta}")
        print(f"Supported topologies: {list(predictor.topo_encoding.keys())}")
        
        # 儲存結果
        with open(str(RESULTS_DIR / 'bfs_alpha_beta_test_results.pkl'), 'wb') as f:
            pickle.dump(overall_results, f)
        
        print(f"Results saved to: {RESULTS_DIR}/")
    else:
        print("No valid test results")

if __name__ == "__main__":
    main()