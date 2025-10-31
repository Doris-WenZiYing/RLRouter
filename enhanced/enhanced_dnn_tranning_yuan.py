#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import joblib

# Windows console UTF-8（可留可去）
if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul 2>&1')

# ====== 路徑設定 ======
BASE_DIR = Path(__file__).parent
DATASET_FILE = BASE_DIR / "bfs_alpha_beta_dataset_all_topologies.csv"
LABEL_MAP_FILE = BASE_DIR / "bfs_alpha_beta_label_mappings.pkl"
RESULTS_DIR = BASE_DIR / "bfs_alpha_beta_training_results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_NAME = RESULTS_DIR / "bfs_alpha_beta_dnn_model.keras"
SCALER_NAME = BASE_DIR / "bfs_alpha_beta_scaler.pkl"

# ====== 載入數據與 label_map ======
print("Loading Row-Mode Gating dataset...")
try:
    df = pd.read_csv(str(DATASET_FILE), encoding='utf-8')
    with open(str(LABEL_MAP_FILE), 'rb') as f:
        label_info = pickle.load(f)
except FileNotFoundError:
    print("Error: run ILP row-score generator first to create dataset and label_map.")
    sys.exit(1)

# ====== 從 label_map 動態讀取維度 ======
# tm_size = N*N；新的 label_map 仍維持 feature_dimensions.tm_size
tm_size = int(label_info.get('feature_dimensions', {}).get('tm_size', 25))  # 5x5 預設 25
# 特徵欄位位置（0-based）
TOPO_IDX        = tm_size                 # 拓撲碼
ALPHABETA_IDX   = tm_size + 1             # proxy α/β 成本
DERIVED_START   = tm_size + 2             # 5 個 proxy 特徵起點
DERIVED_END     = tm_size + 7             # 結束(不含)
NUM_FEATURES    = tm_size + 7             # TM + topo + proxy_ab + 5 proxies

alpha = float(label_info['bfs_weighting_coefficients']['alpha'])
beta  = float(label_info['bfs_weighting_coefficients']['beta'])
N     = int(label_info.get('N', int(np.sqrt(tm_size))))  # 方便顯示/驗證

print(f"Dataset columns = {df.shape[1]}")
print(f"Interpreted N={N}, tm_size={tm_size}, NUM_FEATURES={NUM_FEATURES}")
print(f"Alpha={alpha}, Beta={beta}")

# ====== 切出特徵 / 標籤 ======
feature_cols = df.columns[:NUM_FEATURES].tolist()                 # 0..NUM_FEATURES-1
label_cols   = [c for c in df.columns if str(c).startswith('score_')]  # ★ 新的 row-score 標籤

if len(label_cols) == 0:
    print("Error: no 'score_*' columns found. Confirm ILP exporter wrote row-mode labels.")
    sys.exit(1)

X = df[feature_cols].values
y = df[label_cols].values  # shape = [num_samples, N-1]

print(f"Data shapes: X={X.shape}, y={y.shape} (y is row score vector of length N-1)")

# ====== 特徵標準化（拓撲碼不縮放）======
tm_features            = X[:, :tm_size]
topo_features          = X[:, TOPO_IDX:TOPO_IDX+1]               # 保留原值（不要縮放）
proxy_alpha_beta_cost  = X[:, ALPHABETA_IDX:ALPHABETA_IDX+1]
proxy_derived_features = X[:, DERIVED_START:DERIVED_END]         # 5 維

X_to_scale = np.concatenate([tm_features, proxy_alpha_beta_cost, proxy_derived_features], axis=1)
scaler = StandardScaler()
X_scaled_part = scaler.fit_transform(X_to_scale)

# 重新組合： [TM_scaled | topo_raw | proxyAB_scaled | 5proxies_scaled]
X_scaled = np.concatenate([
    X_scaled_part[:, :tm_size],
    topo_features,                                  # raw topo code
    X_scaled_part[:, tm_size:tm_size+1],            # proxyAB
    X_scaled_part[:, tm_size+1:]                    # 5 proxies
], axis=1)

joblib.dump(scaler, str(SCALER_NAME))

# ====== 拓撲分佈（展示用途）======
topo_encoding = label_info['topology_encoding']                   # {'linear':0, ...}
reverse_topo = {code: name for name, code in topo_encoding.items()}

print("\nTopology distribution:")
for code, name in reverse_topo.items():
    cnt = int(np.sum(X[:, TOPO_IDX] == code))
    pct = cnt / len(X) * 100 if len(X) else 0
    print(f"  {name.upper()}: {cnt} samples ({pct:.1f}%)")

# ====== train/test split（以拓撲碼分層）======
def _stratify_safe(arr):
    """保證 stratify 的 y 為 1D 並且沒有 nan"""
    arr = np.asarray(arr).ravel()
    mask = ~np.isnan(arr)
    return arr[mask], mask

stratify_target, mask_strat = _stratify_safe(X_scaled[:, TOPO_IDX])
if mask_strat.sum() < len(X_scaled):
    # 萬一 topo 欄位有 NaN，就一起濾掉
    X_scaled, y = X_scaled[mask_strat], y[mask_strat]

min_samples_per_topo = min(int(np.sum(X_scaled[:, TOPO_IDX] == code)) for code in reverse_topo.keys())
try:
    if min_samples_per_topo >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=X_scaled[:, TOPO_IDX].ravel()
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
except ValueError:
    # 再保險一次
    mask = ~np.isnan(X_scaled[:, TOPO_IDX].ravel())
    X_scaled, y = X_scaled[mask], y[mask]
    if min(int(np.sum(X_scaled[:, TOPO_IDX] == code)) for code in reverse_topo.keys()) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=X_scaled[:, TOPO_IDX].ravel()
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ====== 建模（多輸出迴歸：輸出維度 = N-1）======
def create_row_score_model(input_dim, output_dim, complexity='medium'):
    if complexity == 'simple':
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(), Dropout(0.30),
            Dense(128, activation='relu'),
            BatchNormalization(), Dropout(0.30),
            Dense(output_dim, activation='linear')
        ])
    elif complexity == 'complex':
        model = Sequential([
            Dense(1024, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(), Dropout(0.40),
            Dense(512, activation='relu'),
            BatchNormalization(), Dropout(0.40),
            Dense(256, activation='relu'),
            BatchNormalization(), Dropout(0.40),
            Dense(128, activation='relu'),
            BatchNormalization(), Dropout(0.30),
            Dense(output_dim, activation='linear')
        ])
    else:
        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(), Dropout(0.40),
            Dense(256, activation='relu'),
            BatchNormalization(), Dropout(0.40),
            Dense(128, activation='relu'),
            BatchNormalization(), Dropout(0.30),
            Dense(output_dim, activation='linear')
        ])
    return model

total_samples = len(X_train)
output_dim = y.shape[1]  # = N-1

if total_samples < 1000:
    model_complexity = 'simple';  epochs, patience_lr, patience_stop = 200, 15, 25
elif total_samples > 5000:
    model_complexity = 'complex'; epochs, patience_lr, patience_stop = 500, 25, 35
else:
    model_complexity = 'medium';  epochs, patience_lr, patience_stop = 400, 20, 30

print(f"\nBuilding {model_complexity.upper()} model...")
print(f"Input: {X_scaled.shape[1]}D, Output: {output_dim}D (row scores)")

model = create_row_score_model(X_scaled.shape[1], output_dim, model_complexity)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_lr, min_lr=1e-6, verbose=1),
    ModelCheckpoint(str(MODEL_NAME), monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=patience_stop, restore_best_weights=True, verbose=1)
]

print(f"\nTraining for {epochs} epochs...")
history = model.fit(
    X_train, y_train,
    epochs=epochs, batch_size=32,
    validation_split=0.15,
    callbacks=callbacks, verbose=1
)

# ====== 評估（回歸）======
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test, verbose=0)
print(f"\nTest Results: MSE={test_loss:.6f}, MAE={test_mae:.6f}")

# 按拓撲評估（回歸指標）
topo_results = {}
for topo_code, topo_name in reverse_topo.items():
    topo_mask = (X_test[:, TOPO_IDX] == topo_code)
    cnt = int(np.sum(topo_mask))
    if cnt == 0:
        continue
    yt = y_test[topo_mask]; yp = y_pred[topo_mask]
    mse = float(np.mean((yt - yp) ** 2))
    mae = float(np.mean(np.abs(yt - yp)))
    topo_results[topo_name] = {
        'mse': mse,
        'mae': mae,
        'samples': cnt
    }
    status = "OK" if mae < 1.0 else "Warning"
    print(f"{status}: {topo_name.upper()}: MAE={mae:.4f}, MSE={mse:.6f}, samples={cnt}")

# （可選）Top-K 命中率：把真實/預測分數各自取 top-K 目的地，比對重疊率
def topk_hit_rate(y_true, y_pred, K=2):
    assert y_true.shape == y_pred.shape
    n = y_true.shape[0]
    hits = 0
    for i in range(n):
        true_idx = np.argsort(-y_true[i])[:K]
        pred_idx = np.argsort(-y_pred[i])[:K]
        # 集合交集大小 / K
        hits += len(set(true_idx.tolist()) & set(pred_idx.tolist())) / K
    return hits / n if n > 0 else 0.0

for K in (1, 2, 3):
    h = topk_hit_rate(y_test, y_pred, K=K)
    print(f"Top-{K} hit rate: {h:.3f}")

# ====== 視覺化（簡易）======
def plot_results():
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Val Loss')
        ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

        if 'mae' in history.history:
            ax2.plot(history.history['mae'], label='Train MAE')
            ax2.plot(history.history['val_mae'], label='Val MAE')
            ax2.set_title('Training MAE'); ax2.legend(); ax2.grid(True, alpha=0.3)

        if topo_results:
            topos = list(topo_results.keys())
            maes  = [topo_results[t]['mae'] for t in topos]
            ax3.bar([t.upper() for t in topos], maes, alpha=0.8)
            ax3.set_title('MAE by Topology')
            ax3.axhline(y=1.0, color='red', linestyle='--', label='MAE=1.0')
            ax3.legend(); ax3.grid(True, alpha=0.3); plt.setp(ax3.get_xticklabels(), rotation=45)

        # 取部分點畫 真值 vs. 預測 的散點
        sample_idx = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)
        ax4.scatter(y_test[sample_idx].flatten(), y_pred[sample_idx].flatten(), alpha=0.5, s=1)
        lo, hi = y_test.min(), y_test.max()
        ax4.plot([lo, hi], [lo, hi], 'r--')
        ax4.set_title('Prediction vs Actual (All Row Scores)')
        ax4.set_xlabel('Actual'); ax4.set_ylabel('Predicted'); ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(RESULTS_DIR / 'training_results.png'), dpi=300, bbox_inches='tight')
        if not sys.platform.startswith('win') or 'DISPLAY' in os.environ:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"Plot error: {e}")

plot_results()

# ====== 儲存 summary ======
row_output_dim = int(label_info.get('row_output_dim', y.shape[1]))  # = N-1

results_summary = {
    'overall_test_mse': float(test_loss),
    'overall_test_mae': float(test_mae),
    'topology_results': topo_results,
    'topk_hit_rate': {k: float(topk_hit_rate(y_test, y_pred, K=k)) for k in (1, 2, 3)},
    'model_type': f'{model_complexity}_row_score_regression',
    'bfs_weighting_coefficients': {'alpha': alpha, 'beta': beta},
    'topology_encoding': topo_encoding,
    'solution_dimensions': int(y.shape[1]),
    'row_output_dim': row_output_dim,
    'tm_size': tm_size,
    'feature_dim': NUM_FEATURES,
    'notes': 'Outputs are row-mode scores for all d ≠ s, aligned with ILP exporter ordering.'
}
with open(str(RESULTS_DIR / 'results_summary.pkl'), 'wb') as f:
    pickle.dump(results_summary, f)

print(f"\n=== Training Complete ===")
print(f"Model: {model_complexity.upper()} ({model.count_params():,} params)")
print(f"Alpha/Beta: {alpha}/{beta}")
print(f"Features: {NUM_FEATURES}D (TM={tm_size}+Topo+ProxyAB+5Proxies)")
print(f"Output(dim): {y.shape[1]} (should be N-1={N-1})")
print(f"Files saved:\n  - {RESULTS_DIR}/bfs_alpha_beta_dnn_model.keras\n  - {SCALER_NAME}\n  - {RESULTS_DIR}/training_results.png\n  - {RESULTS_DIR}/results_summary.pkl")
