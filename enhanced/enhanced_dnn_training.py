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

if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul 2>&1')

# ====== 路徑設定 ======
BASE_DIR = Path(__file__).parent
DATASET_FILE = BASE_DIR / "bfs_alpha_beta_dataset_all_topologies.csv"
LABEL_MAP_FILE = BASE_DIR / "bfs_alpha_beta_label_mappings.pkl"
MODEL_NAME = "bfs_alpha_beta_dnn_model.keras"
SCALER_NAME = BASE_DIR / "bfs_alpha_beta_scaler.pkl"
RESULTS_DIR = BASE_DIR / "bfs_alpha_beta_training_results"

RESULTS_DIR.mkdir(exist_ok=True)

# ====== 載入數據 ======
print("Loading BFS Alpha/Beta dataset...")
try:
    df = pd.read_csv(str(DATASET_FILE), encoding='utf-8')
    with open(str(LABEL_MAP_FILE), 'rb') as f:
        label_info = pickle.load(f)
except FileNotFoundError:
    print("Error: Run enhanced_ilp_solver_with_power.py first")
    sys.exit(1)

# 解析數據結構
feature_cols = df.columns[:71].tolist()  # 71維特徵
label_cols = [col for col in df.columns if col.startswith('demand_')]

X = df[feature_cols].values  # 71維特徵
y = df[label_cols].values    # 標準分配解標籤

alpha = label_info['bfs_weighting_coefficients']['alpha']
beta = label_info['bfs_weighting_coefficients']['beta']

print(f"Data: X={X.shape}, y={y.shape}")
print(f"BFS Weighting: Alpha={alpha}, Beta={beta}")
print(f"Features: 71D (64TM + 1Topo + 1AlphaBeta + 5LabelDerived)")

# ====== 特徵標準化 ======
tm_features = X[:, :64]                    # TM特徵
topo_features = X[:, 64:65]                # 拓撲特徵
alpha_beta_cost = X[:, 65:66]              # Alpha/Beta成本特徵
label_derived_features = X[:, 66:71]       # 標籤衍生特徵

# 標準化數值型特徵
X_to_scale = np.concatenate([tm_features, alpha_beta_cost, label_derived_features], axis=1)
scaler = StandardScaler()
X_scaled_part = scaler.fit_transform(X_to_scale)

# 重新組合
X_scaled = np.concatenate([
    X_scaled_part[:, :64],      # 標準化TM
    topo_features,              # 原始拓撲
    X_scaled_part[:, 64:65],    # 標準化Alpha/Beta成本
    X_scaled_part[:, 65:70]     # 標準化標籤衍生特徵
], axis=1)

joblib.dump(scaler, str(SCALER_NAME))

# ====== 拓撲分析 ======
topo_encoding = label_info['topology_encoding']
reverse_topo = {code: name for name, code in topo_encoding.items()}

print("\nTopology distribution:")
for topo_code, topo_name in reverse_topo.items():
    topo_samples = np.sum(X[:, 64] == topo_code)
    percentage = topo_samples / len(X) * 100
    print(f"  {topo_name.upper()}: {topo_samples} samples ({percentage:.1f}%)")

# ====== 數據切分 ======
min_samples_per_topo = min(np.sum(X[:, 64] == code) for code in reverse_topo.keys())

if min_samples_per_topo >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=X_scaled[:, 64])
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ====== 模型建立 ======
def create_bfs_alpha_beta_model(input_dim, output_dim, complexity='medium'):
    if complexity == 'simple':
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(output_dim, activation='linear')
        ])
    elif complexity == 'complex':
        model = Sequential([
            Dense(1024, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(output_dim, activation='linear')
        ])
    else:  # medium
        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(output_dim, activation='linear')
        ])
    return model

total_samples = len(X_train)
output_dim = y.shape[1]

if total_samples < 1000:
    model_complexity = 'simple'
    epochs, patience_lr, patience_stop = 200, 15, 25
elif total_samples > 5000:
    model_complexity = 'complex'
    epochs, patience_lr, patience_stop = 500, 25, 35
else:
    model_complexity = 'medium'
    epochs, patience_lr, patience_stop = 400, 20, 30

print(f"\nBuilding {model_complexity.upper()} model...")
print(f"Input: 71D, Output: {output_dim}D")

model = create_bfs_alpha_beta_model(X_scaled.shape[1], output_dim, model_complexity)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# ====== 訓練設定 ======
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_lr, min_lr=1e-6, verbose=1),
    ModelCheckpoint(str(RESULTS_DIR / MODEL_NAME), monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=patience_stop, restore_best_weights=True, verbose=1)
]

# ====== 訓練 ======
print(f"\nTraining for {epochs} epochs...")
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=32,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# ====== 評估 ======
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test, verbose=0)

print(f"\nTest Results: MSE={test_loss:.6f}, MAE={test_mae:.6f}")

# 按拓撲評估
topo_results = {}
for topo_code, topo_name in reverse_topo.items():
    topo_test_mask = X_test[:, 64] == topo_code
    topo_sample_count = np.sum(topo_test_mask)
    
    if topo_sample_count > 0:
        topo_y_test = y_test[topo_test_mask]
        topo_y_pred = y_pred[topo_test_mask]
        
        topo_mse = np.mean((topo_y_test - topo_y_pred) ** 2)
        topo_mae = np.mean(np.abs(topo_y_test - topo_y_pred))
        
        topo_y_pred_rounded = np.round(topo_y_pred)
        path_matches = np.mean(topo_y_test[:, ::3] == topo_y_pred_rounded[:, ::3])
        wavelength_matches = np.mean(topo_y_test[:, 1::3] == topo_y_pred_rounded[:, 1::3])
        
        topo_results[topo_name] = {
            'mse': topo_mse,
            'mae': topo_mae,
            'path_match_rate': path_matches,
            'wavelength_match_rate': wavelength_matches,
            'samples': topo_sample_count
        }
        
        status = "OK" if topo_mae < 1.0 else "Warning"
        print(f"{status}: {topo_name.upper()}: MAE={topo_mae:.4f}, Path={path_matches:.2%}, WL={wavelength_matches:.2%}")

# ====== 視覺化 ======
def plot_results():
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 訓練曲線
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Val Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE曲線
        if 'mae' in history.history:
            ax2.plot(history.history['mae'], label='Train MAE')
            ax2.plot(history.history['val_mae'], label='Val MAE')
            ax2.set_title('Training MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 拓撲MAE對比
        if topo_results:
            topos = list(topo_results.keys())
            maes = [topo_results[t]['mae'] for t in topos]
            colors = ['green' if mae < 1.0 else 'orange' for mae in maes]
            
            ax3.bar([t.upper() for t in topos], maes, color=colors, alpha=0.8)
            ax3.set_title('MAE by Topology')
            ax3.axhline(y=1.0, color='red', linestyle='--', label='Target=1.0')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # 預測vs實際
        sample_idx = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)
        ax4.scatter(y_test[sample_idx].flatten(), y_pred[sample_idx].flatten(), alpha=0.5, s=1)
        ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax4.set_title('Prediction vs Actual')
        ax4.set_xlabel('Actual')
        ax4.set_ylabel('Predicted')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(RESULTS_DIR / 'training_results.png'), dpi=300, bbox_inches='tight')
        
        if not sys.platform.startswith('win') or 'DISPLAY' in os.environ:
            plt.show()
        plt.close()
        
    except Exception as e:
        print(f"Plot error: {e}")

plot_results()

# ====== 儲存結果 ======
results_summary = {
    'overall_test_mse': test_loss,
    'overall_test_mae': test_mae,
    'topology_results': topo_results,
    'model_type': f'{model_complexity}_bfs_alpha_beta_regression',
    'bfs_weighting_coefficients': {'alpha': alpha, 'beta': beta},
    'topology_encoding': topo_encoding,
    'solution_dimensions': output_dim,
    'demand_count': label_info['demand_count']
}

with open(str(RESULTS_DIR / 'results_summary.pkl'), 'wb') as f:
    pickle.dump(results_summary, f)

# ====== 最終報告 ======
good_performers = sum(1 for result in topo_results.values() if result['mae'] < 1.0)
total_topos = len(topo_results)

print(f"\n=== Training Complete ===")
print(f"Model: {model_complexity.upper()} ({model.count_params():,} params)")
print(f"Alpha/Beta: {alpha}/{beta}")
print(f"Features: 71D (TM+Topo+AlphaBeta+LabelDerived)")
print(f"Output: {output_dim}D standard solutions")
print(f"Performance: {good_performers}/{total_topos} topologies with MAE<1.0")
print(f"Files: {RESULTS_DIR}/{MODEL_NAME}, {SCALER_NAME}")
print("Ready for RL comparison with same Alpha/Beta coefficients!")