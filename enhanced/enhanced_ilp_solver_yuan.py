#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========= Row-Score Dataset Maker (no ILP; 71D features + (N-1) scores) =========
# - TM 必須是 N×N（這裡 N=5），且「只會有一列有數字、其餘列為 0」。
# - 每一列 (src=s) 產生一筆樣本：features=71D、labels=(N-1) 個 score。
# - score(s->d) = ALPHA * |s-d| * DELAY_UNIT + BETA * TM[s,d]，把 TM 當 power。
# - 拓撲特徵與 5 個 proxy 完全沿用你的舊管線（維度不變）。
# - 產生的 CSV：前 71 欄是特徵，後面依序是 score_0..score_{N-2}（固定順序）。

import os
import sys
import time as timemod
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import networkx as nx

# tqdm 可選；沒有就用簡單 for
try:
    from tqdm.auto import tqdm as _tqdm_base
    TQDM_AVAILABLE = True
    def _tqdm(x, **k):
        return _tqdm_base(x, **k)
except Exception:
    TQDM_AVAILABLE = False
    def _tqdm(x, **k):
        return x

# Windows console UTF-8
if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul 2>&1')

# ====== 參數 ======
ALPHA = 1.0   # 延遲權重
BETA  = 0.01  # 功率權重
DELAY_UNIT = 1e-3  # index-based delay 係數：Tprop_index = |s-d| * DELAY_UNIT

# 注意：你要的 N=5 / TM 資料夾也要對應 5 節點
N = 5

# 路徑
BASE_DIR       = Path(__file__).parent
INPUT_DIR      = BASE_DIR / "tms_19nodes_enhanced" # 這裡要放 5×5 的 TM
TOPOLOGY_FILE  = BASE_DIR / "topology_info.pkl"
OUTPUT_CSV     = BASE_DIR / "bfs_alpha_beta_dataset_all_topologies.csv"
LABEL_MAP_FILE = BASE_DIR / "bfs_alpha_beta_label_mappings.pkl"

# 物理常數（只供 proxy 用，維持你原本實作）
LIGHT_SPEED_IN_FIBER = 2e8  # m/s
NODE_DELAY           = 1e-4 # s/hop
LAMBDA_PROXY         = 2

# Power 代理參數（供 proxy 用；不影響你的 score 定義）
POWER_BASE           = 10.0
POWER_PER_HOP        = 5.0
POWER_PER_WAVELENGTH = 2.0

print(f"[RowScore] Alpha={ALPHA}, Beta={BETA}, N={N}, DELAY_UNIT={DELAY_UNIT}")

# ====== 載入拓撲 ======
try:
    with open(str(TOPOLOGY_FILE), 'rb') as f:
        topology_info = pickle.load(f)
    topology_names = sorted(topology_info.keys())
    topo_encoding  = {name: idx for idx, name in enumerate(topology_names)}
    print(f"Loaded {len(topology_names)} topologies: {topology_names}")
except FileNotFoundError:
    print("Error: topology_info.pkl not found. Please prepare it first.")
    sys.exit(1)

# ====== 工具函式 ======
def _postfix(it, text: str):
    """
    安全地更新 tqdm 的尾註。如果不是 tqdm 物件（例如 list），就忽略。
    先嘗試 set_postfix_str，不存在就退而求其次用 set_postfix。
    """
    fn = getattr(it, "set_postfix_str", None)
    if callable(fn):
        try:
            fn(text)
            return
        except Exception:
            pass
    fn2 = getattr(it, "set_postfix", None)
    if callable(fn2):
        try:
            fn2({"info": text})
        except Exception:
            pass

def prop_delay_index(s: int, d: int) -> float:
    """用 index 距離近似的傳播延遲：|s-d| * DELAY_UNIT"""
    return abs(s - d) * DELAY_UNIT

def shortest_path_info(G: nx.Graph, s: int, d: int):
    """最短路徑資訊（供 proxy 計算）"""
    try:
        p = nx.shortest_path(G, s, d, weight='length')
        hop = len(p) - 1
        dist_m = 0.0
        for u, v in zip(p, p[1:]):
            dist_m += G[u][v]['length'] * 1000.0
        return p, hop, dist_m
    except Exception:
        return None, None, None

def build_proxy_features(tm: np.ndarray, G: nx.Graph, alpha=ALPHA, beta=BETA):
    """
    與你舊管線一致：回傳 1+5 個 proxy（alpha_beta_proxy + 5 個統計）
    注意：這裡把 tm 當 traffic 權重；你實際 TM 只有一列有值，所以能對應到「該 src 列」的 proxy。
    """
    N = tm.shape[0]
    total = tm.sum() + 1e-9
    tnet_sum = 0.0
    hop_sum  = 0.0
    power_proxy_sum = 0.0
    edge_load = {(u, v): 0.0 for u, v in G.edges()}

    for s in range(N):
        for d in range(N):
            if s == d or tm[s, d] <= 0:
                continue
            p, hop, dist_m = shortest_path_info(G, s, d)
            if p is None:
                continue
            Tprop = dist_m / LIGHT_SPEED_IN_FIBER + hop * NODE_DELAY
            tnet_sum += tm[s, d] * Tprop
            hop_sum  += tm[s, d] * hop

            power_proxy = POWER_BASE + hop * POWER_PER_HOP + LAMBDA_PROXY * POWER_PER_WAVELENGTH
            power_proxy_sum += tm[s, d] * power_proxy

            for u, v in zip(p, p[1:]):
                if (u, v) in edge_load:
                    edge_load[(u, v)] += tm[s, d]
                elif (v, u) in edge_load:
                    edge_load[(v, u)] += tm[s, d]

    tnet_mean  = tnet_sum / total
    hop_mean   = hop_sum  / total
    power_mean = power_proxy_sum / total

    if edge_load:
        vals = list(edge_load.values())
        edge_load_max  = max(vals)
        edge_load_mean = sum(vals) / len(vals)
    else:
        edge_load_max = 0.0
        edge_load_mean = 0.0

    alpha_beta_proxy = alpha * tnet_mean + beta * power_mean
    return alpha_beta_proxy, tnet_mean, hop_mean, edge_load_max, edge_load_mean, power_mean

# ====== 主流程（逐列做樣本） ======
all_X, all_Y = [], []

topo_bar = _tqdm(topology_names, desc="Topologies", unit="topo")
for topo_name in topo_bar:
    G = topology_info[topo_name]['graph']
    topo_code = topo_encoding[topo_name]

    topo_dir = INPUT_DIR / topo_name
    if not topo_dir.exists():
        print(f"[warn] TM folder missing: {topo_dir}")
        continue

    tm_files = sorted([f for f in topo_dir.iterdir() if f.suffix == '.csv'])
    tm_bar = _tqdm(tm_files, desc=f"{topo_name} TMs", unit="tm", leave=False)

    processed = 0
    for tm_path in tm_bar:
        try:
            tm_full = pd.read_csv(str(tm_path), header=None, encoding='utf-8').values
            if tm_full.shape != (N, N):
                # 若 TM 尺寸跟 N 不符，這裡簡單截/補 0 到 N×N
                fixed = np.zeros((N, N), dtype=float)
                r = min(N, tm_full.shape[0]); c = min(N, tm_full.shape[1])
                fixed[:r, :c] = tm_full[:r, :c]
                tm_full = fixed
        except Exception as e:
            # ★ 修正：把 iterator (tm_bar) 傳給 _postfix
            _postfix(tm_bar, f"read_err:{e}")
            continue

        # 每一列（src）做一筆
        for s in range(N):
            # 只保留第 s 列（其他列清 0）
            tm_row = np.zeros_like(tm_full, dtype=float)
            tm_row[s, :] = tm_full[s, :]
            np.fill_diagonal(tm_row, 0.0)

            # 71D 特徵
            flat_tm = tm_row.flatten()
            ab_proxy, tnet_mean, hop_mean, edge_max, edge_mean, pwr_mean = \
                build_proxy_features(tm_row, G, alpha=ALPHA, beta=BETA)

            features_71 = np.concatenate([
                flat_tm,                # N*N
                [topo_code],            # 1
                [ab_proxy],             # 1
                [tnet_mean], [hop_mean], [edge_max], [edge_mean], [pwr_mean]  # 5
            ])

            # (N-1) 維的 score 向量：固定順序 d=0..N-1, d≠s
            scores = []
            for d in range(N):
                if d == s:
                    continue
                Psd = float(tm_full[s, d])  # TM 值當 power
                Tprop = prop_delay_index(s, d)
                score = ALPHA * Tprop + BETA * Psd
                scores.append(score)

            all_X.append(features_71)
            all_Y.append(scores)
            processed += 1

        # ★ 修正：同上，把 iterator (tm_bar) 傳給 _postfix
        _postfix(tm_bar, f"proc={processed}")

    print(f"[{topo_name}] samples: {processed}")

# ====== 存檔 ======
try:
    if not all_X or not all_Y:
        raise RuntimeError("No samples generated. Check INPUT_DIR, N, and TM content.")

    X = np.array(all_X)               # [num_samples, 71]
    Y = np.array(all_Y)               # [num_samples, N-1]
    df = pd.DataFrame(X)              # 先放 71 維特徵

    # 追加 (N-1) 還有 score_* label 欄位
    for k in range(Y.shape[1]):
        df[f"score_{k}"] = Y[:, k]

    df.to_csv(str(OUTPUT_CSV), index=False, encoding='utf-8')
    print(f"\nDataset saved: {OUTPUT_CSV}")
    print(f"Samples: {len(all_Y)}, Feature cols: 71, Label cols: {Y.shape[1]}")
except Exception as e:
    print(f"Save error: {e}")
    sys.exit(1)

# ====== 標籤映射（提供 DNN 使用的 meta） ======
label_map = {
    'label_type': 'row_destination_scores',
    'tm_row_mode': True,                 # 一筆 = 單一 src 的整列 TM
    'row_output_dim': int(Y.shape[1]),   # = N-1
    'solution_dimensions': int(Y.shape[1]),
    'topology_encoding': topo_encoding,
    'feature_dimensions': {
        'tm_size': N * N,                # 這  N*N 會進 scaler
        'topo_size': 1,                  # 這 1 維（拓撲碼）不進 scaler
        'alpha_beta_cost_size': 1,       # 這 1 維會進 scaler
        'label_derived_size': 5          # 這 5 維會進 scaler
    },
    'bfs_weighting_coefficients': {'alpha': ALPHA, 'beta': BETA},
    'N': N,
    'notes': 'labels are alpha*|s-d|*DELAY_UNIT + beta*TM[s,d] for d!=s in fixed order'
}
with open(str(LABEL_MAP_FILE), 'wb') as f:
    pickle.dump(label_map, f)
print(f"Label mapping saved: {LABEL_MAP_FILE}")
