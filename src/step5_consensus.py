"""
步驟5：建立共識矩陣
"""

import numpy as np
import sys
sys.path.append('.')

from config.config import Config
from src.utils import (
    load_pickle, save_numpy,
    print_step_header, print_matrix_statistics
)


def build_consensus_matrix(clustering_results, n_locations, top_n=25):
    """
    從多次Louvain結果建立共識矩陣
    
    Parameters:
    -----------
    clustering_results : list
        Louvain結果列表，格式為[(modularity, labels), ...]
        - modularity: 模組度分數 (float)
        - labels: 分群標籤數組 (np.ndarray, shape=(n_locations,))
    n_locations : int
        地點總數
    top_n : int
        取模組度最高的前N次結果
        建議值：20-30
    
    Returns:
    --------
    consensus_matrix : np.ndarray
        共識矩陣，形狀(n_locations, n_locations)，dtype=float32
        值範圍[0,1]，對角線為1.0
        consensus[i,j] = 地點i和j在top_n次中被分在同一群的頻率
    
    Notes:
    ------
    共識值的物理意義：
    - 高共識(如0.9): 穩定的社群連結
    - 低共識(如0.1): 不穩定或不同社群
    - 中等共識(如0.5): 邊界地點
    """
    print(f"\n【步驟5.1】建立共識矩陣")
    print(f"  總運行次數: {len(clustering_results)}")
    print(f"  取前{top_n}次最優結果")
    
    # 按模組度排序，取前top_n次
    sorted_results = sorted(clustering_results, 
                           key=lambda x: x[0], 
                           reverse=True)
    top_partitions = sorted_results[:top_n]
    
    modularities = [m for m, _ in top_partitions]
    print(f"  Top {top_n} 模組度範圍: {min(modularities):.4f} ~ {max(modularities):.4f}")
    
    # 初始化共識矩陣
    consensus_matrix = np.zeros((n_locations, n_locations), dtype=np.float32)
    
    print(f"\n【步驟5.2】統計同群頻率")
    
    # 對每次分群結果
    for idx, (modularity, labels) in enumerate(top_partitions):
        if idx % 5 == 0:
            print(f"  處理: {idx}/{top_n}", end='\r')
        
        # 檢查每對地點是否在同一群
        for i in range(n_locations):
            for j in range(n_locations):
                if labels[i] == labels[j]:
                    consensus_matrix[i, j] += 1
    
    print(f"  處理: {top_n}/{top_n}")
    
    # 標準化（除以總次數）
    consensus_matrix /= top_n
    
    print(f"✓ 共識矩陣建立完成")
    
    return consensus_matrix


def analyze_consensus_matrix(consensus_matrix):
    """
    分析共識矩陣的統計特性
    
    Parameters:
    -----------
    consensus_matrix : np.ndarray
        共識矩陣
    
    Returns:
    --------
    stats : dict
        統計信息
    """
    # 只看上三角（不包括對角線）
    upper_triangle = consensus_matrix[np.triu_indices_from(consensus_matrix, k=1)]
    
    stats = {
        'mean': float(np.mean(upper_triangle)),
        'std': float(np.std(upper_triangle)),
        'min': float(np.min(upper_triangle)),
        'max': float(np.max(upper_triangle)),
        'median': float(np.median(upper_triangle)),
    }
    
    # 分布統計
    # print(f"\n共識值分布（不含對角線）:")
    # print(f"  0.0-0.2: {(upper_triangle < 0.2).sum()} ({(upper_triangle < 0.2).sum()/len(upper_triangle)*100:.1f}%)")
    # print(f"  0.2-0.4: {((upper_triangle >= 0.2) & (upper_triangle < 0.4)).sum()} ({((upper_triangle >= 0.2) & (upper_triangle < 0.4)).sum()/len(upper_triangle)*100:.1f}%)")
    # print(f"  0.4-0.6: {((upper_triangle >= 0.4) & (upper_triangle < 0.6)).sum()} ({((upper_triangle >= 0.4) & (upper_triangle < 0.6)).sum()/len(upper_triangle)*100:.1f}%)")
    # print(f"  0.6-0.8: {((upper_triangle >= 0.6) & (upper_triangle < 0.8)).sum()} ({((upper_triangle >= 0.6) & (upper_triangle < 0.8)).sum()/len(upper_triangle)*100:.1f}%)")
    # print(f"  0.8-1.0: {(upper_triangle >= 0.8).sum()} ({(upper_triangle >= 0.8).sum()/len(upper_triangle)*100:.1f}%)")
    
    return stats


def save_consensus_matrix(consensus_matrix, period_name, output_dir):
    """
    保存共識矩陣
    
    Parameters:
    -----------
    consensus_matrix : np.ndarray
        共識矩陣
    period_name : str
        時期名稱
    output_dir : Path
        輸出目錄
    """
    filepath = output_dir / 'consensus_matrices' / f'consensus_{period_name}.npy'
    save_numpy(consensus_matrix, filepath)


def run_consensus_building(clustering_results=None, n_locations=None, period_name=None,
                           config=None, output_dir=None):
    """
    執行共識矩陣建立的主函數
    
    Parameters:
    -----------
    clustering_results : list, optional
        Louvain結果列表
        如果為None，自動從文件載入
    n_locations : int, optional
        地點總數
        如果clustering_results為None，必須提供
    period_name : str, optional
        時期名稱，例如'1965-1974'
    config : Config, optional
        配置對象，默認使用Config
    output_dir : Path, optional
        輸出目錄，默認使用config.PROCESSED_DATA_DIR
    
    Returns:
    --------
    consensus_matrix : np.ndarray
        共識矩陣，形狀(n_locations, n_locations)
    
    Notes:
    ------
    - 支援自動載入Louvain結果
    - 建議取前20-30次最優結果
    """
    if config is None:
        config = Config
    
    if output_dir is None:
        output_dir = config.PROCESSED_DATA_DIR
    
    print_step_header(5, "建立共識矩陣", period_name)
    
    # 自動載入Louvain結果（如果未提供）
    if clustering_results is None:
        if period_name is None:
            raise ValueError("必須提供 clustering_results 或 period_name")
        
        print("\n自動載入Louvain結果...")
        filepath = output_dir / 'louvain_results' / f'louvain_results_{period_name}.pkl'
        clustering_results = load_pickle(filepath)
        
        # 從結果推斷地點數
        if n_locations is None:
            _, first_labels = clustering_results[0]
            n_locations = len(first_labels)
            print(f"  推斷地點數: {n_locations}")
    
    if n_locations is None:
        raise ValueError("必須提供 n_locations")
    
    # 建立共識矩陣
    consensus_matrix = build_consensus_matrix(
        clustering_results,
        n_locations,
        top_n=config.TOP_N_PARTITIONS
    )
    
    # 保存結果
    save_consensus_matrix(consensus_matrix, period_name, output_dir)
    
    # 打印統計
    print_matrix_statistics(consensus_matrix, "\n共識矩陣")
    
    # 分析共識矩陣
    stats = analyze_consensus_matrix(consensus_matrix)
    
    print(f"\n✓ 步驟5完成")
    
    return consensus_matrix