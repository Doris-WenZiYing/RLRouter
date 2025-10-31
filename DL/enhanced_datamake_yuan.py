import numpy as np
import pandas as pd
import os
import sys
import networkx as nx
import pickle
from pathlib import Path

# 設定控制台編碼為UTF-8（如果可能）
if sys.platform.startswith('win'):
    try:
        import locale
        if locale.getpreferredencoding().lower() != 'utf-8':
            print("Note: Console encoding may not support Unicode characters")
    except:
        pass

# ====== 基本參數設定 ======
N_NODES = 5            
N_SAMPLES = 20         
VARIATION = 0.5               

# 使用pathlib確保跨平台路徑兼容性
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "tms_5nodes_enhanced"
TOPOLOGY_FILE = BASE_DIR / "topology_info.pkl"

# ====== 建立輸出目錄 ======
OUTPUT_DIR.mkdir(exist_ok=True)

# ====== 拓撲定義區域 (可以自由修改！) ======

def create_full_mesh(n):
    """全連接拓撲 - 所有節點互相連接"""
    G = nx.complete_graph(n, create_using=nx.DiGraph())
    for u, v in G.edges():
        G[u][v]['length'] = 1
    return G

def create_ring_topology(n):
    """環形拓撲 - 1-2-3-4-5-6-7-8-1"""
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        G.add_edge(i, (i+1) % n, length=1)
        G.add_edge((i+1) % n, i, length=1)  # 雙向
    return G

def create_linear_topology(n):
    """線性拓撲 - 1-2-3-4-5-6-7-8 (沒有環)"""
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n-1):
        G.add_edge(i, i+1, length=1)
        G.add_edge(i+1, i, length=1)  # 雙向
    return G

def create_star_topology(n):
    """星形拓撲 - 節點0在中心，連接所有其他節點"""
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(1, n):
        G.add_edge(0, i, length=1)
        G.add_edge(i, 0, length=1)  # 雙向
    return G

def create_custom_topology_1(n):
    """
    自定義拓撲1 - 你可以在這裡定義自己的拓撲！
    範例：環形 + 一些額外連接
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    # 先建立環形基礎
    for i in range(n):
        G.add_edge(i, (i+1) % n, length=1)
        G.add_edge((i+1) % n, i, length=1)
    
    # 添加一些對角連接 (自定義部分)
    extra_connections = [
        (0, 3), (1, 4), (2, 5), (3, 6)  # 你可以修改這些連接
    ]
    
    for u, v in extra_connections:
        if u < n and v < n:  # 確保節點存在
            G.add_edge(u, v, length=1)
            G.add_edge(v, u, length=1)
    
    return G

def create_custom_topology_2(n):
    """
    自定義拓撲2 - 更複雜的自定義拓撲
    範例：兩個環相連
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    # 前半部分形成一個環 (0-3)
    for i in range(4):
        G.add_edge(i, (i+1) % 4, length=1)
        G.add_edge((i+1) % 4, i, length=1)
    
    # 後半部分形成另一個環 (4-7)
    for i in range(4, 8):
        next_node = 4 + ((i-4+1) % 4)
        G.add_edge(i, next_node, length=1)
        G.add_edge(next_node, i, length=1)
    
    # 兩個環之間的連接
    bridge_connections = [(1, 5), (3, 7)]  # 連接兩個環
    for u, v in bridge_connections:
        G.add_edge(u, v, length=1)
        G.add_edge(v, u, length=1)
    
    return G

# ====== 在這裡選擇你要的拓撲！ ======
topologies = {
    'full_mesh': create_full_mesh(N_NODES),
    'ring': create_ring_topology(N_NODES),
    'linear': create_linear_topology(N_NODES),
    'star': create_star_topology(N_NODES)
    # 'custom1': create_custom_topology_1(N_NODES),
    # 'custom2': create_custom_topology_2(N_NODES)
}

# ====== 快速自定義拓撲選項 ======
def create_path_topology(n, path_edges):
    """
    根據指定的路徑創建拓撲
    path_edges: 邊的列表，例如 [(0,1), (1,2), (2,3), ...]
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    for u, v in path_edges:
        if u < n and v < n:
            G.add_edge(u, v, length=1)
            G.add_edge(v, u, length=1)  # 雙向
    
    return G

# 可以取消註解來添加自定義拓撲
# ring_edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,0)]
# topologies['my_custom_ring'] = create_path_topology(N_NODES, ring_edges)

# ====== 驗證拓撲連通性 ======
def validate_topology(G, name):
    """檢查拓撲是否強連通"""
    if not nx.is_strongly_connected(G):
        print(f"Warning: {name} topology is not strongly connected!")
        
        # 列出不連通的節點對
        disconnected_pairs = []
        for s in range(N_NODES):
            for d in range(N_NODES):
                if s != d and not nx.has_path(G, s, d):
                    disconnected_pairs.append((s, d))
        
        if disconnected_pairs:
            print(f"   Disconnected pairs: {disconnected_pairs[:5]}{'...' if len(disconnected_pairs) > 5 else ''}")
        
        return False
    else:
        print(f"OK: {name} topology is strongly connected")
        return True

print("Checking topology connectivity...")
valid_topologies = {}
for name, G in topologies.items():
    if validate_topology(G, name):
        valid_topologies[name] = G
    else:
        print(f"   Skipping {name} topology (not connected)")

topologies = valid_topologies

# ====== 儲存拓撲資訊 ======
topology_info = {}
for topo_name, G in topologies.items():
    topology_info[topo_name] = {
        'graph': G,
        'edge_count': G.number_of_edges(),
        'density': nx.density(G),
        'diameter': nx.diameter(G.to_undirected()) if nx.is_connected(G.to_undirected()) else float('inf')
    }

try:
    with open(str(TOPOLOGY_FILE), 'wb') as f:
        pickle.dump(topology_info, f)
    print(f"Topology info saved to: {TOPOLOGY_FILE}")
except Exception as e:
    print(f"Error saving topology file: {e}")
    sys.exit(1)

# ====== 生成TM數據 ======
print(f"\nGenerating TM data...")

# 建立基準流量矩陣
np.random.seed(42)  
canonical_tm = np.random.randint(20, 120, size=(N_NODES, N_NODES))
np.fill_diagonal(canonical_tm, 0)  

print(f"Base traffic matrix:")
print(canonical_tm)

# 為每種拓撲生成TM數據
for topo_name, G in topologies.items():
    print(f"\nGenerating TM data for {topo_name.upper()} topology...")
    
    # 使用pathlib確保跨平台路徑
    topo_dir = OUTPUT_DIR / topo_name
    topo_dir.mkdir(exist_ok=True)
    
    # 重設隨機種子確保每種拓撲使用相同的TM模式
    np.random.seed(42)
    
    for i in range(N_SAMPLES):
        noise = np.random.normal(loc=0.0, scale=VARIATION, size=(N_NODES, N_NODES))
        noisy_tm = canonical_tm * (1 + noise)
        noisy_tm = np.clip(noisy_tm, 0, None).astype(int)
        
        # 對於不連通的節點對，將流量設為0
        for s in range(N_NODES):
            for d in range(N_NODES):
                if s != d and not nx.has_path(G, s, d):
                    noisy_tm[s][d] = 0
        
        df = pd.DataFrame(noisy_tm)
        csv_file = topo_dir / f"tm_{i:04d}.csv"
        try:
            df.to_csv(str(csv_file), index=False, header=False, encoding='utf-8')
        except Exception as e:
            print(f"Error writing CSV file {csv_file}: {e}")
            # 嘗試使用其他編碼
            try:
                df.to_csv(str(csv_file), index=False, header=False, encoding='latin1')
            except Exception as e2:
                print(f"Error with fallback encoding: {e2}")
                continue
        
        if i % 200 == 0:
            print(f"  Generated {i} / {N_SAMPLES} TMs")
    
    print(f"Completed: {topo_name} topology")

print("All Traffic Matrices generated successfully!")

# ====== 拓撲統計資訊 ======
print("\n" + "="*60)
print("Topology Structure Statistics")
print("="*60)

for topo_name, G in topologies.items():
    print(f"\n{topo_name.upper()}:")
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   Density: {nx.density(G):.3f}")
    print(f"   Strongly connected: {nx.is_strongly_connected(G)}")
    
    # 顯示部分邊的連接
    edges_sample = list(G.edges())[:10]
    print(f"   Sample edges: {edges_sample}{'...' if len(list(G.edges())) > 10 else ''}")
    
    # 計算連通性統計
    if nx.is_connected(G.to_undirected()):
        diameter = nx.diameter(G.to_undirected())
        avg_path = nx.average_shortest_path_length(G.to_undirected())
        print(f"   Diameter: {diameter}")
        print(f"   Average shortest path: {avg_path:.2f}")

# ====== 使用說明 ======
print(f"\n" + "="*60)
print("Custom Topology Usage Guide")
print("="*60)

print(f"""
How to customize your topology:

1. Modify existing topology functions:
   - Edit create_custom_topology_1() or create_custom_topology_2()
   - Add your desired edge connections

2. Use quick path functionality:
   # Example: create 1-2-3-4-5-6-7-8-1 ring
   ring_edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,0)]
   topologies['my_ring'] = create_path_topology(N_NODES, ring_edges)

3. Select specific topologies:
   # If you only want certain topologies, modify the topologies dict
   topologies = {{
       'ring': create_ring_topology(N_NODES),
       'custom1': create_custom_topology_1(N_NODES)
   }}

4. Verify results:
   - Program automatically checks topology connectivity
   - Only strongly connected topologies are kept

Output files:
   - {OUTPUT_DIR}/[topology_name]/tm_XXXX.csv
   - {TOPOLOGY_FILE}

Generated topologies: {list(topologies.keys())}
""")

print(f"Enhanced DataMake completed!")
print(f"   Generated {len(topologies)} topologies")
print(f"   Each topology has {N_SAMPLES} TM samples")
print(f"   Total training samples: {len(topologies) * N_SAMPLES}")

# ====== Windows特殊處理 ======
if sys.platform.startswith('win'):
    print(f"\nWindows compatibility notes:")
    print(f"- All file paths use cross-platform format")
    print(f"- CSV files saved with UTF-8 encoding")
    print(f"- Path separators automatically handled")
    print(f"- Console output may show simplified characters if Unicode not supported")