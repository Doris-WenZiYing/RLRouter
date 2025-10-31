#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
from itertools import combinations

import numpy as np
import pandas as pd
import pickle
import joblib
import networkx as nx
from pathlib import Path
import time

try:
    from tensorflow.keras.models import load_model  # optional
except Exception:
    def load_model(_):  # 沒 TF/模型也不會掛
        return None

# Windows console UTF-8
if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul 2>&1')

# ====== 路徑與預設 ======
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "bfs_alpha_beta_training_results" / "bfs_alpha_beta_dnn_model.keras"
SCALER_PATH = BASE_DIR / "bfs_alpha_beta_scaler.pkl"
LABEL_MAP_PATH = BASE_DIR / "bfs_alpha_beta_label_mappings.pkl"
TOPOLOGY_INFO_PATH = BASE_DIR / "topology_info.pkl"
RESULTS_DIR = BASE_DIR / "bfs_alpha_beta_prediction_results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_ALPHA = 1.0
DEFAULT_BETA  = 0.01

# ====== 與 ILP 一致的物理常數，給 proxies 用 ======
LIGHT_SPEED_IN_FIBER = 2e8   # m/s
NODE_DELAY = 1e-4            # s per hop
LAMBDA_PROXY = 2             # wavelength proxy（與 ILP 側一致）
POWER_BASE = 10.0
POWER_PER_HOP = 5.0
POWER_PER_WAVELENGTH = 2.0

class BFSAlphaBetaRWAPredictor:
    """BFS Alpha/Beta 加權的 RWA 規劃器（外部拓樸 + row-mode DNN gating + 時間量測 + 剪枝統計）"""

    def __init__(self, use_dnn_gating=True):
        # DNN（可有可無）
        self.model = None
        self.scaler = None
        self.label_info = None

        # topology 容器（改由外部注入）
        self.topology_info = {}
        self.topo_encoding = {}
        self.reverse_topo = {}

        # 權重 α/β
        self.alpha = DEFAULT_ALPHA
        self.beta  = DEFAULT_BETA

        # λ 參數
        self.max_wl = 40

        # 若沒有 label_map 就用這組 power 預設
        self.power_params = {
            'power_base': POWER_BASE,
            'power_per_hop': POWER_PER_HOP,
            'power_per_wavelength': POWER_PER_WAVELENGTH
        }

        # 內部 ROADM 名稱 <-> index 映射 & 鄰接與距離
        self.linear_roadm2idx = {}
        self.linear_idx2roadm = {}
        self.Topology = {}     # {'r1': {'Neighbors': ['r2', ...]}, ...}
        self.linear_dist = {}  # {('r1','r2'): km, ...}

        # gating 開關（可被環境變數覆蓋）
        disable_env = os.environ.get("PREDICTOR_DISABLE_DNN", "0") == "1"
        self.use_dnn_gating = use_dnn_gating and (not disable_env)

    # ---------- 載入（可選） ----------
    def load_model_components(self):
        """盡力載入 DNN/scaler/label_map；缺了也無妨。"""
        try:
            if MODEL_PATH.exists():
                self.model = load_model(str(MODEL_PATH))
        except Exception as e:
            print(f"[warn] load_model failed: {e}")
            self.model = None

        try:
            if SCALER_PATH.exists():
                self.scaler = joblib.load(str(SCALER_PATH))
        except Exception as e:
            print(f"[warn] load scaler failed: {e}")
            self.scaler = None

        try:
            if LABEL_MAP_PATH.exists():
                with open(str(LABEL_MAP_PATH), 'rb') as f:
                    self.label_info = pickle.load(f)
                # 覆蓋 α/β 與功率參數（若存在）
                if 'bfs_weighting_coefficients' in self.label_info:
                    self.alpha = float(self.label_info['bfs_weighting_coefficients'].get('alpha', self.alpha))
                    self.beta  = float(self.label_info['bfs_weighting_coefficients'].get('beta',  self.beta))
                if 'power_parameters' in self.label_info:
                    self.power_params.update(self.label_info['power_parameters'])
        except Exception as e:
            print(f"[warn] load label_info failed: {e}")
            self.label_info = None

        print(f"[predictor] Ready. Alpha={self.alpha}, Beta={self.beta}, gating={'ON' if self.use_dnn_gating else 'OFF'}")
        return True

    # ---------- 外部拓樸注入 ----------
    def load_external_topology(self, Topology_dict: dict, ROADM_list: dict, topo_name='external'):
        """
        Topology_dict: { 'r1': {'Neighbors':[...], 'ConnectedHosts':[...], 'ConnectedSF':[...]} , ... }
        ROADM_list:    {('r1','r2'): distance_km, ...}
        """
        # 建 ROADM 名稱 -> index
        roadms = list(Topology_dict.keys())
        roadms.sort()  # 確保穩定順序
        self.linear_roadm2idx = {r: i for i, r in enumerate(roadms)}
        self.linear_idx2roadm = {i: r for r, i in self.linear_roadm2idx.items()}

        # 鄰接/距離
        self.Topology = {r: {'Neighbors': list(Topology_dict[r].get('Neighbors', []))}
                         for r in roadms}
        self.linear_dist = dict(ROADM_list)  # km

        # networkx 圖（邊長用 km）
        G = nx.Graph()
        for i in range(len(roadms)):
            G.add_node(i)
        for (ru, rv), L in ROADM_list.items():
            if ru in self.linear_roadm2idx and rv in self.linear_roadm2idx:
                u = self.linear_roadm2idx[ru]; v = self.linear_roadm2idx[rv]
                G.add_edge(u, v, length=L)

        self.topology_info = {topo_name: {'graph': G}}
        self.topo_encoding = {topo_name: 0}
        self.reverse_topo = {0: topo_name}

    def roadm_sfs_from_topology(self, Topology_dict: dict):
        """把 Topology_dict['ConnectedSF'] 轉成 {idx: [(name, compute, power, mem, threads), ...]}"""
        mapping = {}
        for rname, info in Topology_dict.items():
            idx = self.linear_roadm2idx.get(rname)
            if idx is None:
                continue
            sfs = info.get('ConnectedSF', [])
            norm = []
            for sf in sfs:
                name, compute, power, mem, threads = sf
                norm.append((name, float(compute), float(power), float(mem), int(threads)))
            mapping[idx] = norm
        return mapping

    def host_to_roadm_idx(self, host_id: str, Topology_dict: dict):
        """給 'h3' → 回傳其對應的 ROADM index。"""
        for rname, info in Topology_dict.items():
            if host_id in info.get('ConnectedHosts', []):
                return self.linear_roadm2roadm_idx(rname)
        return None

    def linear_roadm2roadm_idx(self, rname):
        return self.linear_roadm2idx.get(rname)

    # ---------- 與 ILP 一致：proxy 特徵計算 ----------
    def _shortest_path_info(self, G, s, d):
        try:
            p = nx.shortest_path(G, s, d, weight='length')
            hop = len(p) - 1
            dist_m = 0.0
            for u, v in zip(p, p[1:]):
                # length: km → m
                dist_m += G[u][v]['length'] * 1000.0
            return p, hop, dist_m
        except Exception:
            return None, None, None

    def _build_proxy_features(self, tm, G, alpha, beta):
        """
        與 ILP 同版：回傳
        (alpha_beta_proxy, tnet_mean, hop_mean, edge_load_max, edge_load_mean, power_proxy_mean)
        """
        N = tm.shape[0]
        total_traffic = tm.sum() + 1e-9

        tnet_sum = 0.0
        hop_sum = 0.0
        power_proxy_sum = 0.0
        edge_load = {(u, v): 0.0 for u, v in G.edges()}

        for s in range(N):
            for d in range(N):
                if s == d or tm[s][d] <= 0:
                    continue
                p, hop, dist_m = self._shortest_path_info(G, s, d)
                if p is None:
                    continue

                Tprop = dist_m / LIGHT_SPEED_IN_FIBER + hop * NODE_DELAY
                tnet_sum += tm[s][d] * Tprop
                hop_sum += tm[s][d] * hop

                power_proxy = POWER_BASE + hop * POWER_PER_HOP + LAMBDA_PROXY * POWER_PER_WAVELENGTH
                power_proxy_sum += tm[s][d] * power_proxy

                for u, v in zip(p, p[1:]):
                    if (u, v) in edge_load:
                        edge_load[(u, v)] += tm[s][d]
                    elif (v, u) in edge_load:
                        edge_load[(v, u)] += tm[s][d]

        tnet_mean = tnet_sum / total_traffic
        hop_mean = hop_sum / total_traffic
        power_mean = power_proxy_sum / total_traffic

        if edge_load:
            edge_vals = list(edge_load.values())
            edge_load_max = max(edge_vals)
            edge_load_mean = sum(edge_vals) / len(edge_vals)
        else:
            edge_load_max = 0.0
            edge_load_mean = 0.0

        alpha_beta_proxy = alpha * tnet_mean + beta * power_mean
        return alpha_beta_proxy, tnet_mean, hop_mean, edge_load_max, edge_load_mean, power_mean

    # ---------- DNN 32 維（row-mode） ----------
    def _tm_size_from_scaler(self):
        """
        從 scaler 反推 TM 邊長 N：
        訓練時進 scaler 的是 [TM + proxyAB + 5 proxies] → 維度 = N*N + 6
        （topology code 沒進 scaler）
        """
        try:
            n_in = int(getattr(self.scaler, 'n_features_in_', 0))
            if n_in >= 7:
                tm_flat = n_in - 6
                N = int(round(math.sqrt(tm_flat)))
                if N * N == tm_flat:
                    return N
        except Exception:
            pass
        return 5  # 預設 5x5

    def _prepare_row_feature(self, tm_full, topo_code, G, src):
        """
        做成「row-mode」單筆 32D 特徵：
          [TM(N*N, 只留第 src 列) | topo(1, raw) | proxyAB(1) | 5 proxies]
        topology 那一維不縮放；其餘（TM + proxyAB + 5 proxies）用 scaler。
        """
        N = self._tm_size_from_scaler()
        tm_full = np.asarray(tm_full, dtype=float)
        # 尺寸對齊 N×N
        if tm_full.shape != (N, N):
            fixed = np.zeros((N, N), dtype=float)
            r = min(N, tm_full.shape[0]); c = min(N, tm_full.shape[1])
            fixed[:r, :c] = tm_full[:r, :c]
            tm_full = fixed

        # 只保留第 src 列
        tm_row = np.zeros_like(tm_full, dtype=float)
        tm_row[src, :] = tm_full[src, :]
        np.fill_diagonal(tm_row, 0.0)

        flat_tm = tm_row.flatten()  # N*N

        alpha = float(self.alpha); beta = float(self.beta)
        proxy_ab, tnet_mean, hop_mean, edge_load_max, edge_load_mean, power_proxy_mean = \
            self._build_proxy_features(tm_row, G, alpha=alpha, beta=beta)

        # 原始 32 維（尚未 scaler 拼裝）
        feat_raw = np.concatenate([
            flat_tm,                         # N*N
            np.array([topo_code]),           # 1（RAW，不縮放）
            np.array([proxy_ab]),            # 1
            np.array([tnet_mean, hop_mean, edge_load_max, edge_load_mean, power_proxy_mean])  # 5
        ]).reshape(1, -1)                    # (1, 32)

        # scaler：TM + proxyAB + 5 proxies（共 N*N + 6）→ topology 那 1 維不縮放
        tm_len   = N * N
        topo_col = tm_len
        to_scale = np.concatenate([feat_raw[:, :tm_len], feat_raw[:, topo_col+1:]], axis=1)  # (1, 31)
        scaled   = self.scaler.transform(to_scale)

        feat32 = np.concatenate([
            scaled[:, :tm_len],              # TM_scaled
            feat_raw[:, topo_col:topo_col+1],# topo_raw
            scaled[:, tm_len:]               # proxies_scaled
        ], axis=1)                           # (1, 32)

        return feat32.astype(np.float32)

    def _rowmode_scores_for_src(self, tm_full, topology_name, src):
        """
        回傳 (dst_list, scores)：
          - dst_list = [所有 d≠src 的索引，固定順序]
          - scores   = model 輸出的長度 N-1 向量（與 dst_list 一一對應）
        分數意義：score = α*Tprop(index) + β*power（越小越好）
        """
        if (self.model is None) or (self.scaler is None) or (topology_name not in self.topology_info):
            return [], np.array([])
        N = self._tm_size_from_scaler()
        if not (0 <= src < N):
            return [], np.array([])

        G = self.topology_info[topology_name]['graph']
        topo_code = float(self.topo_encoding.get(topology_name, 0))
        X_row = self._prepare_row_feature(tm_full, topo_code, G, src)  # (1, 32)
        y_hat = self.model.predict(X_row, verbose=0).ravel()           # (N-1,)

        dst_list = [d for d in range(N) if d != src]
        return dst_list, y_hat

    def _rank_destinations_by_dnn(self, source_idx, dests, tm_full, topology_name='external', topk=3):
        """
        正確 row-mode：一次 forward 取得 src 的 (N-1) 分數，對齊 d，挑前 K。
        注意：分數越小越好 → 取最小 K 個。
        """
        dst_list, scores = self._rowmode_scores_for_src(tm_full, topology_name, source_idx)
        if len(dst_list) == 0:
            return []

        # 只保留目前 BFS 可達的候選 dests
        mask = [i for i, d in enumerate(dst_list) if d in dests]
        if not mask:
            return []
        sub_scores = scores[mask]
        sub_dsts   = [dst_list[i] for i in mask]

        order = np.argsort(sub_scores)[:max(1, topk)]
        return [sub_dsts[i] for i in order]

    # ---------- 物理解延遲 ----------
    @staticmethod
    def _tx_delay(Di_bits, B=50e9, gOSNR=20):
        return Di_bits / (B * math.log2(1 + gOSNR))

    @staticmethod
    def _prop_delay(distance_m, node_delay=NODE_DELAY):
        # fiber 約 2e8 m/s + per-hop node delay
        return distance_m / (2e8) + node_delay

    # ---------- BFS 延遲擴展 ----------
    def _bfs_delay_expansion(self, src_roadm, T_limit, initial_delay):
        delay_dict = {r: (math.inf, None) for r in self.Topology}
        delay_dict[src_roadm] = (initial_delay, None)
        q = [(src_roadm, initial_delay)]
        while q:
            nxt = []
            for cur, cur_t in q:
                for nb in self.Topology[cur]['Neighbors']:
                    dist_km = self.linear_dist.get((cur, nb))
                    if dist_km is None:
                        continue
                    tot = cur_t + self._prop_delay(dist_km * 1000.0)
                    if tot > T_limit:
                        continue
                    if tot <= delay_dict[nb][0]:
                        delay_dict[nb] = (tot, cur)
                        nxt.append((nb, tot))
            q = nxt
        return delay_dict

    @staticmethod
    def _restore_roadm_path(src, dst, delay_dict):
        path, cur = [], dst
        while cur is not None and cur != src:
            path.insert(0, cur)
            cur = delay_dict[cur][1]
            if cur is None:
                break
        if cur == src:
            path.insert(0, src)
        return path

    def _roadm_path_to_idx(self, roadm_path):
        return [self.linear_roadm2idx[r] for r in roadm_path if r in self.linear_roadm2idx]

    # ---------- 功率（連結功率；SF power 另外算） ----------
    def calculate_path_power(self, path, G, wavelength):
        base_power = self.power_params.get('power_base', POWER_BASE)
        p_hop      = self.power_params.get('power_per_hop', POWER_PER_HOP)
        p_wl       = self.power_params.get('power_per_wavelength', POWER_PER_WAVELENGTH)
        if not path or len(path) <= 1:
            return base_power
        return base_power + (len(path) - 1) * p_hop + wavelength * p_wl

    # ---------- 核心選擇：目的地 + SF 子集合 ----------
    def choose_best_destination(
        self,
        source,                 # source ROADM index
        dests,                  # 候選目的地 index list
        demand_value,           # 保留簽名（這版不用）
        topology_name,          # 'external'
        Di_task, Fi, Mi, Tmax,  # 資料量 Di、指令量 Fi、記憶體上限 Mi、時限 Tmax
        roadm_to_sfs: dict,     # {dest_idx: [(name, compute, power, mem, threads), ...]}
        alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA
    ):
        """回傳 (best, timings, enum_stats)；timings: {'t_bfs','t_gating','t_sf_enum','t_total_core'} (秒)"""
        self.alpha, self.beta = float(alpha), float(beta)
        timings = {'t_bfs': 0.0, 't_gating': 0.0, 't_sf_enum': 0.0, 't_total_core': 0.0}
        enum_stats = {
            'dest_before': 0, 'dest_after': 0,
            'subsets_possible_before': 0, 'subsets_possible_after': 0,
            'subsets_evaluated': 0, 'subsets_avoided_by_prune': 0,
            'per_dest': []
        }
        t_core0 = time.perf_counter()

        src_roadm = self.linear_idx2roadm.get(source)
        if src_roadm is None:
            return None, timings, enum_stats

        # 初始傳輸延遲 + 節點 delay
        Ttrans = self._tx_delay(Di_task)
        initial_delay = Ttrans + NODE_DELAY

        # 用「最樂觀 compute」推 BFS 可達性界限
        all_sfs = [sf for lst in roadm_to_sfs.values() for sf in lst]
        if not all_sfs:
            return None, timings, enum_stats
        optimistic_cap = sum(sf[1] / max(1, sf[4]) for sf in all_sfs)
        if optimistic_cap <= 0:
            return None, timings, enum_stats
        Tmin_proc = Fi / optimistic_cap
        T_limit = Tmax - Tmin_proc
        if T_limit <= 0:
            return None, timings, enum_stats

        # --- BFS timing ---
        t0 = time.perf_counter()
        delay_dict = self._bfs_delay_expansion(src_roadm, T_limit, initial_delay)
        timings['t_bfs'] = time.perf_counter() - t0

        # 可達目的地（且該點有 SF）
        dest_candidates = []
        for d in dests:
            d_r = self.linear_idx2roadm.get(d)
            if d_r is None:
                continue
            net_delay, _ = delay_dict.get(d_r, (math.inf, None))
            if math.isfinite(net_delay) and roadm_to_sfs.get(d, []):
                dest_candidates.append((d, net_delay))
        dest_candidates.sort(key=lambda x: x[1])

        # === 剪枝前統計 ===
        enum_stats['dest_before'] = len(dest_candidates)
        for d_idx, _ in dest_candidates:
            m = len(roadm_to_sfs.get(d_idx, []))
            enum_stats['subsets_possible_before'] += max(0, (1 << m) - 1)

        # ====== DNN gating（量時間；可關） ======
        if self.use_dnn_gating and (self.model is not None) and (self.scaler is not None):
            t0 = time.perf_counter()
            reachable_dests = [d for d, _ in dest_candidates]
            N = self._tm_size_from_scaler()
            tm_full = np.zeros((N, N), dtype=float)  # 線上若有實際 TM，可替換
            use_topk = 3
            dnn_top = self._rank_destinations_by_dnn(
                source_idx=source,
                dests=reachable_dests,
                tm_full=tm_full,
                topology_name=topology_name,
                topk=use_topk
            )
            timings['t_gating'] = time.perf_counter() - t0
            if dnn_top:
                before = len(dest_candidates)
                dest_candidates = [item for item in dest_candidates if item[0] in dnn_top]
                print(f"[gating] DNN top-{use_topk} for src={source}: {dnn_top} from {before} candidates")

        # ====== 無 DNN 時可在此加入解析式 gating（若你想強制關 DNN 可加在此） ======

        # === 剪枝後統計 ===
        enum_stats['dest_after'] = len(dest_candidates)
        for d_idx, _ in dest_candidates:
            m = len(roadm_to_sfs.get(d_idx, []))
            enum_stats['subsets_possible_after'] += max(0, (1 << m) - 1)
        enum_stats['subsets_avoided_by_prune'] = max(
            0, enum_stats['subsets_possible_before'] - enum_stats['subsets_possible_after']
        )

        # --- SF 枚舉 timing ---
        t0 = time.perf_counter()
        best = None
        for d_idx, net_delay in dest_candidates:
            time_budget = Tmax - net_delay
            if time_budget <= 0:
                continue

            sfs = roadm_to_sfs.get(d_idx, [])
            n = len(sfs)
            local_best = None

            # 每個目的地的枚舉統計
            per_dest_eval = 0
            t_dest0 = time.perf_counter()

            # 枚舉非空子集合（記憶體 ≤ Mi）
            for r in range(1, n + 1):
                for subset in combinations(range(n), r):
                    per_dest_eval += 1
                    enum_stats['subsets_evaluated'] += 1

                    mem_sum = 0.0
                    cap_sum = 0.0
                    pwr_sum = 0.0
                    chosen  = []

                    feasible = True
                    for i in subset:
                        name, compute, power, mem, threads = sfs[i]
                        mem_sum += mem
                        if mem_sum > Mi:
                            feasible = False
                            break
                        cap_sum += compute / max(1, threads)
                        pwr_sum += power
                        chosen.append((name, compute, power, mem, threads))
                    if not feasible or cap_sum <= 0:
                        continue

                    Tproc = Fi / cap_sum
                    if not (Tproc < time_budget):  # 嚴格小於
                        continue

                    roadm_path = self._restore_roadm_path(src_roadm, self.linear_idx2roadm[d_idx], delay_dict)
                    idx_path = self._roadm_path_to_idx(roadm_path)

                    total_cost = self.alpha * (net_delay + Tproc) + self.beta * (pwr_sum)
                    cand = {
                        'd': d_idx,
                        'path': idx_path,
                        'wavelength': -1,  # λ 之後全域分配
                        'processing_time': Tproc,
                        'net_delay': net_delay,
                        'sf_power': pwr_sum,
                        'link_power': 0.0,
                        'total_cost': total_cost,
                        'sfs': chosen
                    }
                    if (local_best is None) or (cand['total_cost'] < local_best['total_cost']):
                        local_best = cand

            # 收 per-dest 統計
            t_dest = time.perf_counter() - t_dest0
            enum_stats['per_dest'].append({
                'd': d_idx,
                'm': n,
                'subsets_possible': max(0, (1 << n) - 1),
                'subsets_eval': per_dest_eval,
                't_enum_dest': t_dest
            })

            if local_best is not None:
                if (best is None) or (local_best['total_cost'] < best['total_cost']):
                    best = local_best

        timings['t_sf_enum'] = time.perf_counter() - t0
        timings['t_total_core'] = time.perf_counter() - t_core0
        return best, timings, enum_stats

# ---------- 直接執行：拓樸與任務 ----------
def main():
    t0 = time.perf_counter()
    print("BFS Alpha/Beta RWA Predictor (External Topology)")
    print("=" * 60)

    GPU_catalog = {
        'A100': [1e8, 40],
        'H100': [2e8, 80],
    }
    SF_list = [
        ('SF1', GPU_catalog['A100'][0], 30, GPU_catalog['A100'][1], 1),  # r1
        ('SF2', GPU_catalog['H100'][0], 30, GPU_catalog['H100'][1], 1),  # r2
        ('SF3', GPU_catalog['A100'][0], 20, GPU_catalog['A100'][1], 1),  # r3
        ('SF4', GPU_catalog['A100'][0], 30, GPU_catalog['A100'][1], 1),  # r1
        ('SF5', GPU_catalog['H100'][0], 20, GPU_catalog['H100'][1], 1),  # r4
        ('SF6', GPU_catalog['A100'][0], 25, GPU_catalog['A100'][1], 1),  # r3
        ('SF7', GPU_catalog['H100'][0], 15, GPU_catalog['H100'][1], 1),  # r5
    ]
    Tmax_list = [1.5, 3, 5, 3.5, 4]
    Task_list = [
        (1, 'h1',  8e6, 1e8,  40),
        (2, 'h2', 16e6, 2e8,  80),
        (3, 'h3', 80e6, 4e8, 120),
        (4, 'h4', 32e6, 3e8, 100),
        (5, 'h5', 64e6, 5e8, 160),
    ]
    ROADM_list = {
        ('r1','r2'):100, ('r2','r1'):100,
        ('r2','r3'):150, ('r3','r2'):150,
        ('r3','r4'):200, ('r4','r3'):200,
        ('r3','r5'):250, ('r5','r3'):250,
    }
    Topology = {
        'r1': {'Neighbors':['r2'],             'ConnectedHosts':['h1'], 'ConnectedSF':[SF_list[0], SF_list[3]]},
        'r2': {'Neighbors':['r1','r3'],        'ConnectedHosts':['h2'], 'ConnectedSF':[SF_list[1]]},
        'r3': {'Neighbors':['r2','r4','r5'],   'ConnectedHosts':['h3'], 'ConnectedSF':[SF_list[2], SF_list[5]]},
        'r4': {'Neighbors':['r3'],             'ConnectedHosts':['h4'], 'ConnectedSF':[SF_list[4]]},
        'r5': {'Neighbors':['r3'],             'ConnectedHosts':['h5'], 'ConnectedSF':[SF_list[6]]},
    }

    # 開/關 DNN gating：也可用環境變數 PREDICTOR_DISABLE_DNN=1 關掉
    pred = BFSAlphaBetaRWAPredictor(use_dnn_gating=True)
    pred.load_model_components()                       # 可有可無（載不到也不影響）
    pred.load_external_topology(Topology, ROADM_list)  # 用你的拓樸
    roadm_to_sfs = pred.roadm_sfs_from_topology(Topology)

    # 把 host 轉成 source index，組合任務 [(tid, s_idx, Di, Fi, Mi, Tmax), ...]
    tasks = []
    for i, (tid, host_id, Di, Fi, Mi) in enumerate(Task_list):
        s_idx = None
        for rname, info in Topology.items():
            if host_id in info.get('ConnectedHosts', []):
                s_idx = pred.linear_roadm2roadm_idx(rname)
                break
        if s_idx is None:
            print(f"[warn] host {host_id} 找不到 ROADM，略過該任務")
            continue
        tasks.append((tid, s_idx, Di, Fi, Mi, Tmax_list[i]))

    # 全域 λ 使用狀態（避免邊上 λ 衝突）
    edge_wl_usage = {}  # {(min(u,v),max(u,v)): set(wl)}
    def alloc_lambda(idx_path):
        for wl in range(pred.max_wl):
            ok = True
            for u, v in zip(idx_path, idx_path[1:]):
                ek = (min(u,v), max(u,v))
                if wl in edge_wl_usage.get(ek, set()):
                    ok = False; break
            if ok:
                for u, v in zip(idx_path, idx_path[1:]):
                    ek = (min(u,v), max(u,v))
                    edge_wl_usage.setdefault(ek, set()).add(wl)
                return wl
        return None

    alpha, beta = DEFAULT_ALPHA, DEFAULT_BETA
    Success, Fail = 0, 0

    # 追加：時間統計
    agg = {'t_bfs':0.0, 't_gating':0.0, 't_sf_enum':0.0, 't_total_core':0.0}

    for tid, s, Di, Fi, Mi, Tmax in tasks:
        dests = [i for i in range(len(pred.linear_idx2roadm)) if i != s]
        best, tstat, est = pred.choose_best_destination(
            source=s, dests=dests, demand_value=0,
            topology_name='external',
            Di_task=Di, Fi=Fi, Mi=Mi, Tmax=Tmax,
            roadm_to_sfs=roadm_to_sfs,
            alpha=alpha, beta=beta
        )
        # 累計時間
        for k in agg: agg[k] += tstat.get(k, 0.0)

        print(f"\n[Task {tid}] source={s}")
        print(f"  timings: BFS={tstat['t_bfs']*1000:.3f} ms, "
              f"DNN_gating={tstat['t_gating']*1000:.3f} ms, "
              f"SF_enum={tstat['t_sf_enum']*1000:.3f} ms, "
              f"core_total={tstat['t_total_core']*1000:.3f} ms")

        # 剪枝統計（凸顯 gating 幫忙多少）
        print(f"  pruning stats: "
              f"dest_before={est['dest_before']}, dest_after={est['dest_after']}, "
              f"subsets_before={est['subsets_possible_before']}, "
              f"subsets_after={est['subsets_possible_after']}, "
              f"subsets_eval={est['subsets_evaluated']}, "
              f"avoided_by_prune={est['subsets_avoided_by_prune']}")
        if est['subsets_evaluated'] > 0:
            avg_time_per_subset = (tstat['t_sf_enum'] / est['subsets_evaluated'])
            est_no_prune = avg_time_per_subset * est['subsets_possible_before']
            print(f"  est avg time/subset = {avg_time_per_subset*1e6:.2f} µs "
                  f"→ est SF_enum w/o prune ≈ {est_no_prune*1000:.3f} ms")

        if best is None:
            print("  -> 無可行目的地（延遲/算力/記憶體/BFS 約束）")
            Fail += 1
            continue

        path = best['path']
        wl = alloc_lambda(path)
        if wl is None:
            print("  -> 這條路徑沒有剩餘 λ")
            Fail += 1
            continue

        d = best['d']
        print(f"  chosen dest={d}, path={path}, λ={wl}")
        print(f"  net_delay={best['net_delay']:.6f}, proc_time={best['processing_time']:.6f}, "
              f"sf_power={best['sf_power']:.1f}, total_cost={best['total_cost']:.6f}")
        print(f"  SF used at dest {d}: {[sf[0] for sf in best['sfs']]}")
        Success += 1

    total = Success + Fail
    rate = (Success / total * 100) if total > 0 else 0.0
    print("\n📊 統計結果：")
    print(f"  ✔ 成功任務數：{Success}")
    print(f"  ❌ 失敗任務數：{Fail}")
    print(f"  ✅ 成功率：{rate:.2f}%")

    if total > 0:
        print("\n⏱️  核心時間（不含載入模型/拓樸）：")
        print(f"  平均/任務 BFS：       {agg['t_bfs']/total*1000:.3f} ms")
        print(f"  平均/任務 DNN gating：{agg['t_gating']/total*1000:.3f} ms  "
              f"(use_dnn_gating={'ON' if pred.use_dnn_gating else 'OFF'})")
        print(f"  平均/任務 SF 枚舉：   {agg['t_sf_enum']/total*1000:.3f} ms")
        print(f"  平均/任務 核心總計：  {agg['t_total_core']/total*1000:.3f} ms")

    dt = time.perf_counter() - t0
    print(f"[TOTAL (包含載入與 I/O)] elapsed: {dt*1000:.2f} ms")


if __name__ == "__main__":
    main()
