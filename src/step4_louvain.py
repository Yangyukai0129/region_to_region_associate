"""
步驟4：Louvain社群檢測（多次運行）
"""

import numpy as np
import networkit as nk
from tqdm import tqdm
import sys
sys.path.append('.')

from config.config import Config
from src.utils import (
    load_numpy, save_pickle,
    print_step_header, get_timestamp
)


def create_network_from_similarity(similarity_matrix, min_similarity=0.1):
    """
    從相似度矩陣建立網絡圖
    
    Parameters:
    -----------
    similarity_matrix : np.ndarray
        相似度矩陣，形狀(n_locations, n_locations)
        對稱矩陣，值範圍[0,1]
    min_similarity : float
        建立邊的最小相似度閾值
        只有相似度 >= min_similarity 的地點對才建立邊
    
    Returns:
    --------
    G : nk.Graph
        NetworkKit圖對象，加權無向圖
        節點數 = n_locations
        邊權重 = 相似度
    
    Notes:
    ------
    - 過濾低相似度的連接可以減少噪音
    - min_similarity建議設為0.05-0.15
    - 邊權重用於Louvain的模組度計算
    """
    n_locations = similarity_matrix.shape[0]
    
    print(f"\n【步驟4.1】建立網絡圖")
    print(f"  節點數: {n_locations}")
    print(f"  相似度閾值: {min_similarity}")
    
    # 建立加權無向圖
    G = nk.Graph(n_locations, weighted=True, directed=False)
    
    # 添加邊
    n_edges = 0
    total_weight = 0.0
    
    for i in range(n_locations):
        for j in range(i+1, n_locations):
            if similarity_matrix[i, j] >= min_similarity:
                G.addEdge(i, j, similarity_matrix[i, j])
                n_edges += 1
                total_weight += similarity_matrix[i, j]
    
    print(f"  邊數: {n_edges}")
    print(f"  密度: {n_edges / (n_locations * (n_locations - 1) / 2):.4f}")
    print(f"  平均邊權重: {total_weight / n_edges:.4f}")
    
    return G


def run_louvain_multiple_times(G, num_runs=100, gamma=1.5):
    """
    多次運行Louvain社群檢測
    
    Parameters:
    -----------
    G : nk.Graph
        NetworkKit圖對象
    num_runs : int
        運行次數
    gamma : float
        解析度參數，控制社群大小
        - gamma < 1: 傾向大社群
        - gamma = 1: 標準模組度
        - gamma > 1: 傾向小社群
    
    Returns:
    --------
    clustering_results : list
        [(modularity, labels), ...]
        每個元素包含：
        - modularity: 模組度分數 (float)
        - labels: 分群標籤數組 (np.ndarray, shape=(n_locations,))
    
    Notes:
    ------
    - Louvain算法有隨機性，多次運行結果可能不同
    - 保留所有結果以便後續建立共識矩陣
    - 使用tqdm顯示進度條
    """
    n_locations = G.numberOfNodes()
    clustering_results = []
    
    print(f"\n【步驟4.2】執行Louvain社群檢測")
    print(f"  運行次數: {num_runs}")
    print(f"  gamma參數: {gamma}")
    print(f"  開始時間: {get_timestamp()}")
    
    for run in tqdm(range(num_runs), desc="Louvain進度"):
        # 執行PLM (Parallel Louvain Method)
        plm = nk.community.PLM(G, refine=True, gamma=gamma)
        plm.run()
        partition = plm.getPartition()
        
        # 計算模組度
        modularity = nk.community.Modularity().getQuality(partition, G)
        
        # 將Partition轉為numpy數組（便於序列化）
        labels = np.array([partition[i] for i in range(n_locations)], dtype=np.int32)
        
        # 保存結果
        clustering_results.append((modularity, labels))
    
    print(f"  完成時間: {get_timestamp()}")
    
    # 統計模組度
    modularities = [m for m, _ in clustering_results]
    print(f"\n模組度統計:")
    print(f"  範圍: {min(modularities):.4f} ~ {max(modularities):.4f}")
    print(f"  平均: {np.mean(modularities):.4f}")
    print(f"  標準差: {np.std(modularities):.4f}")
    
    # 統計社群數量
    n_communities = [len(np.unique(labels)) for _, labels in clustering_results]
    print(f"\n社群數量統計:")
    print(f"  範圍: {min(n_communities)} ~ {max(n_communities)}")
    print(f"  平均: {np.mean(n_communities):.1f}")
    
    return clustering_results


def save_louvain_results(clustering_results, period_name, output_dir):
    """
    保存Louvain結果
    
    Parameters:
    -----------
    clustering_results : list
        分群結果列表
    period_name : str
        時期名稱
    output_dir : Path
        輸出目錄
    """
    filepath = output_dir / 'louvain_results' / f'louvain_results_{period_name}.pkl'
    save_pickle(clustering_results, filepath)


def run_louvain_clustering(similarity_matrix=None, period_name=None,
                           config=None, output_dir=None):
    """
    執行Louvain社群檢測的主函數
    
    Parameters:
    -----------
    similarity_matrix : np.ndarray, optional
        相似度矩陣，形狀(n_locations, n_locations)
        如果為None，自動從文件載入
    period_name : str, optional
        時期名稱，例如'1965-1974'
    config : Config, optional
        配置對象，默認使用Config
    output_dir : Path, optional
        輸出目錄，默認使用config.PROCESSED_DATA_DIR
    
    Returns:
    --------
    clustering_results : list
        分群結果列表
        [(modularity, labels), ...]
    
    Notes:
    ------
    - 支援自動載入相似度矩陣
    - 運行時間取決於網絡大小和運行次數
    - 800個節點×100次約需5-10分鐘
    """
    if config is None:
        config = Config
    
    if output_dir is None:
        output_dir = config.PROCESSED_DATA_DIR
    
    print_step_header(4, "Louvain社群檢測", period_name)
    
    # 自動載入相似度矩陣（如果未提供）
    if similarity_matrix is None:
        if period_name is None:
            raise ValueError("必須提供 similarity_matrix 或 period_name")
        
        print("\n自動載入相似度矩陣...")
        filepath = output_dir / 'similarity_matrices' / f'similarity_{period_name}.npy'
        similarity_matrix = load_numpy(filepath)
    
    # 建立網絡
    G = create_network_from_similarity(
        similarity_matrix,
        min_similarity=config.MIN_SIMILARITY
    )
    
    # 執行Louvain
    clustering_results = run_louvain_multiple_times(
        G,
        num_runs=config.NUM_LOUVAIN_RUNS,
        gamma=config.LOUVAIN_GAMMA
    )
    
    # 保存結果
    save_louvain_results(clustering_results, period_name, output_dir)
    
    print(f"\n✓ 步驟4完成")
    
    return clustering_results