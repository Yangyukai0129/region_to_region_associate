"""
步驟6：階層式分群（含Silhouette Score選K）
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import Counter
import sys
sys.path.append('.')

from config.config import Config
from src.utils import (
    load_numpy, save_numpy, save_json,
    print_step_header, get_timestamp
)


def consensus_to_distance(consensus_matrix):
    """
    將共識矩陣轉換為距離矩陣
    
    Parameters:
    -----------
    consensus_matrix : np.ndarray
        共識矩陣，形狀(n_locations, n_locations)
        值範圍[0,1]，越大表示越常在同一群
    
    Returns:
    --------
    distance_matrix : np.ndarray
        距離矩陣，形狀(n_locations, n_locations)
        值範圍[0,1]，越大表示越遠
        
    Notes:
    ------
    distance = 1 - consensus
    - consensus = 1.0 → distance = 0.0（完全相同）
    - consensus = 0.0 → distance = 1.0（完全不同）
    """
    distance_matrix = 1.0 - consensus_matrix
    
    # 確保對角線是0（自己到自己的距離）
    np.fill_diagonal(distance_matrix, 0.0)
    
    return distance_matrix


def check_min_cluster_size(labels, min_size):
    """
    檢查是否所有群組都滿足最小大小約束
    
    Parameters:
    -----------
    labels : np.ndarray
        分群標籤，形狀(n_locations,)
    min_size : int
        最小群大小
    
    Returns:
    --------
    is_valid : bool
        是否所有群都 >= min_size
    cluster_sizes : dict
        每個群的大小
    """
    cluster_sizes = Counter(labels)
    min_cluster_size = min(cluster_sizes.values())
    is_valid = min_cluster_size >= min_size
    
    return is_valid, dict(cluster_sizes)


def test_k_values(distance_matrix, k_range, min_cluster_size, linkage_method='ward'):
    """
    測試不同K值，計算Silhouette Score
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        距離矩陣
    k_range : range or list
        要測試的K值範圍
    min_cluster_size : int
        最小群大小約束
    linkage_method : str
        階層式分群的連結方法
    
    Returns:
    --------
    results : dict
        {
            'k_values': [測試的K值],
            'silhouette_scores': [對應的分數],
            'valid_k': [滿足約束的K值],
            'cluster_sizes': [每個K的群大小分布]
        }
    """
    print(f"\n【步驟6.2】測試不同K值")
    print(f"  K值範圍: {min(k_range)} ~ {max(k_range)}")
    print(f"  最小群大小: {min_cluster_size}")
    print(f"  連結方法: {linkage_method}")
    print(f"  開始時間: {get_timestamp()}")
    
    k_values = []
    silhouette_scores = []
    valid_k = []
    cluster_sizes_list = []
    
    for k in k_range:
        # 進度
        if k % 20 == 0 or k == min(k_range):
            print(f"  測試 K={k}...", end='\r')
        
        # 階層式分群
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage=linkage_method
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # 檢查最小群大小
        is_valid, sizes = check_min_cluster_size(labels, min_cluster_size)
        
        k_values.append(k)
        cluster_sizes_list.append(sizes)
        
        if is_valid:
            # 計算Silhouette Score（使用距離矩陣）
            # 注意：只有2個以上的群才能計算
            if k >= 2:
                score = silhouette_score(distance_matrix, labels, metric='precomputed')
                silhouette_scores.append(score)
                valid_k.append(k)
            else:
                silhouette_scores.append(None)
        else:
            silhouette_scores.append(None)  # 不滿足約束
    
    print(f"  測試完成: {len(k_range)}個K值")
    print(f"  滿足約束: {len(valid_k)}個K值")
    print(f"  完成時間: {get_timestamp()}")
    
    results = {
        'k_values': k_values,
        'silhouette_scores': silhouette_scores,
        'valid_k': valid_k,
        'cluster_sizes': cluster_sizes_list
    }
    
    return results


def select_best_k(results):
    """
    選擇最佳K值
    
    Parameters:
    -----------
    results : dict
        test_k_values的輸出
    
    Returns:
    --------
    best_k : int
        最佳分群數
    best_score : float
        最佳Silhouette分數
    """
    valid_k = results['valid_k']
    valid_scores = [results['silhouette_scores'][results['k_values'].index(k)] 
                   for k in valid_k]
    
    if len(valid_scores) == 0:
        raise ValueError("沒有任何K值滿足約束條件")
    
    best_idx = np.argmax(valid_scores)
    best_k = valid_k[best_idx]
    best_score = valid_scores[best_idx]
    
    print(f"\n【步驟6.3】選擇最佳K值")
    print(f"  最佳K值: {best_k}")
    print(f"  Silhouette分數: {best_score:.4f}")
    
    return best_k, best_score


def final_clustering(distance_matrix, n_clusters, linkage_method='ward'):
    """
    用最佳K值進行最終分群
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        距離矩陣
    n_clusters : int
        分群數
    linkage_method : str
        連結方法
    
    Returns:
    --------
    labels : np.ndarray
        分群標籤，形狀(n_locations,)
    cluster_info : dict
        分群統計資訊
    """
    print(f"\n【步驟6.4】最終分群")
    print(f"  使用K值: {n_clusters}")
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage=linkage_method
    )
    labels = clustering.fit_predict(distance_matrix)
    
    # 統計每個群的大小
    cluster_sizes = Counter(labels)
    
    cluster_info = {
        'n_clusters': n_clusters,
        'cluster_sizes': dict(cluster_sizes),
        'min_size': min(cluster_sizes.values()),
        'max_size': max(cluster_sizes.values()),
        'mean_size': np.mean(list(cluster_sizes.values())),
        'median_size': np.median(list(cluster_sizes.values()))
    }
    
    print(f"  群組統計:")
    print(f"    總群數: {n_clusters}")
    print(f"    最小群: {cluster_info['min_size']} 個地點")
    print(f"    最大群: {cluster_info['max_size']} 個地點")
    print(f"    平均群: {cluster_info['mean_size']:.1f} 個地點")
    print(f"    中位數: {cluster_info['median_size']:.1f} 個地點")
    
    return labels, cluster_info


def save_clustering_results(labels, silhouette_results, cluster_info, period_name, output_dir):
    """
    保存分群結果
    
    Parameters:
    -----------
    labels : np.ndarray
        分群標籤
    silhouette_results : dict
        Silhouette測試結果
    cluster_info : dict
        分群統計資訊
    period_name : str
        時期名稱
    output_dir : Path
        輸出目錄
    """
    # 保存標籤
    labels_path = output_dir / 'clusters' / f'clusters_{period_name}.npy'
    save_numpy(labels, labels_path)
    
    # 保存Silhouette結果
    silhouette_path = output_dir / 'clusters' / f'silhouette_{period_name}.json'
    # 將None轉為字符串以便JSON序列化
    silhouette_to_save = {
        'k_values': silhouette_results['k_values'],
        'silhouette_scores': [float(s) if s is not None else None 
                             for s in silhouette_results['silhouette_scores']],
        'valid_k': silhouette_results['valid_k'],
        'best_k': cluster_info['n_clusters'],
        'best_score': float(silhouette_results['silhouette_scores'][
            silhouette_results['k_values'].index(cluster_info['n_clusters'])
        ])
    }
    save_json(silhouette_to_save, silhouette_path)
    
    # 保存群組資訊
    cluster_info_path = output_dir / 'clusters' / f'cluster_info_{period_name}.json'
    save_json(cluster_info, cluster_info_path)


def run_hierarchical_clustering(consensus_matrix=None, period_name=None,
                                config=None, output_dir=None):
    """
    執行階層式分群的主函數
    
    Parameters:
    -----------
    consensus_matrix : np.ndarray, optional
        共識矩陣
        如果為None，自動從文件載入
    period_name : str, optional
        時期名稱
    config : Config, optional
        配置對象
    output_dir : Path, optional
        輸出目錄
    
    Returns:
    --------
    labels : np.ndarray
        分群標籤
    best_k : int
        最佳分群數
    silhouette_results : dict
        Silhouette測試結果
    """
    if config is None:
        config = Config
    
    if output_dir is None:
        output_dir = config.PROCESSED_DATA_DIR
    
    print_step_header(6, "階層式分群", period_name)
    
    # 自動載入共識矩陣（如果未提供）
    if consensus_matrix is None:
        if period_name is None:
            raise ValueError("必須提供 consensus_matrix 或 period_name")
        
        print("\n自動載入共識矩陣...")
        filepath = output_dir / 'consensus_matrices' / f'consensus_{period_name}.npy'
        consensus_matrix = load_numpy(filepath)
    
    # 步驟1：轉換為距離矩陣
    print("\n【步驟6.1】轉換為距離矩陣")
    distance_matrix = consensus_to_distance(consensus_matrix)
    print(f"  距離矩陣形狀: {distance_matrix.shape}")
    print(f"  距離範圍: {distance_matrix.min():.4f} ~ {distance_matrix.max():.4f}")
    print(f"  平均距離: {distance_matrix.mean():.4f}")
    
    # 步驟2：測試不同K值
    k_range = range(config.MIN_CLUSTERS, config.MAX_CLUSTERS + 1)
    silhouette_results = test_k_values(
        distance_matrix,
        k_range,
        min_cluster_size=config.MIN_CLUSTER_SIZE,
        linkage_method=config.LINKAGE_METHOD
    )
    
    # 步驟3：選擇最佳K
    best_k, best_score = select_best_k(silhouette_results)
    
    # 步驟4：最終分群
    labels, cluster_info = final_clustering(
        distance_matrix,
        n_clusters=best_k,
        linkage_method=config.LINKAGE_METHOD
    )
    
    # 保存結果
    save_clustering_results(
        labels,
        silhouette_results,
        cluster_info,
        period_name,
        output_dir
    )
    
    print(f"\n✓ 步驟6完成")
    
    return labels, best_k, silhouette_results