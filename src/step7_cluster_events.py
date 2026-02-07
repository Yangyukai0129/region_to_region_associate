"""
步驟7：建立群組×事件矩陣
"""

import numpy as np
import sys
sys.path.append('.')

from config.config import Config
from src.utils import (
    load_numpy, save_numpy, save_json,
    print_step_header
)


def create_cluster_event_matrix(event_matrix, cluster_labels, 
                                 participation_threshold=0.4):
    """
    從事件×地點矩陣和分群標籤建立群組×事件矩陣
    
    Parameters:
    -----------
    event_matrix : np.ndarray
        事件×地點矩陣，形狀(n_events, n_locations)
        值為 0 或 1
    cluster_labels : np.ndarray
        分群標籤，形狀(n_locations,)
        值為 0 ~ (K-1)，-1 表示孤立地點
    participation_threshold : float
        參與度閾值，預設 0.4 (40%)
        群組內 >= 40% 的地點在事件中有熱浪才標記為 1
    
    Returns:
    --------
    cluster_event_matrix : np.ndarray
        群組×事件矩陣，形狀(n_clusters, n_events)
        值為 0 或 1
    cluster_info : dict
        群組詳細資訊
    """
    n_events, n_locations = event_matrix.shape
    
    # 找出有效的群組 ID (排除 -1)
    valid_clusters = cluster_labels[cluster_labels >= 0]
    unique_clusters = np.unique(valid_clusters)
    n_clusters = len(unique_clusters)
    
    print(f"\n【步驟7.1】建立群組×事件矩陣")
    print(f"  事件數: {n_events}")
    print(f"  地點數: {n_locations}")
    print(f"  群組數: {n_clusters}")
    print(f"  孤立地點: {(cluster_labels == -1).sum()}")
    print(f"  參與度閾值: {participation_threshold} ({participation_threshold*100:.0f}%)")
    
    # 初始化群組×事件矩陣
    cluster_event_matrix = np.zeros((n_clusters, n_events), dtype=np.int8)
    
    # 記錄每個群組的資訊
    cluster_info_list = []
    
    # 對每個群組
    for cluster_idx, cluster_id in enumerate(unique_clusters):
        # 找出屬於這個群組的地點
        cluster_mask = cluster_labels == cluster_id
        cluster_size = cluster_mask.sum()
        cluster_location_indices = np.where(cluster_mask)[0]
        
        # 提取這個群組的事件資料
        cluster_events = event_matrix[:, cluster_mask]  # (n_events, cluster_size)
        
        # 計算每個事件中有多少地點參與
        participation_count = cluster_events.sum(axis=1)  # (n_events,)
        participation_ratio = participation_count / cluster_size
        
        # 應用閾值
        cluster_event_matrix[cluster_idx] = (
            participation_ratio >= participation_threshold
        ).astype(np.int8)
        
        # 記錄群組資訊
        cluster_info_list.append({
            'cluster_id': int(cluster_id),
            'cluster_index': int(cluster_idx),
            'size': int(cluster_size),
            'location_indices': cluster_location_indices.tolist(),
            'n_events_participated': int(cluster_event_matrix[cluster_idx].sum()),
            'participation_ratios': {
                'mean': float(participation_ratio.mean()),
                'max': float(participation_ratio.max()),
                'median': float(np.median(participation_ratio))
            }
        })
    
    # 統計
    events_per_cluster = cluster_event_matrix.sum(axis=1)
    clusters_per_event = cluster_event_matrix.sum(axis=0)
    
    print(f"\n群組統計:")
    print(f"  平均每群參與事件數: {events_per_cluster.mean():.1f}")
    print(f"  最小: {events_per_cluster.min()}")
    print(f"  最大: {events_per_cluster.max()}")
    
    print(f"\n事件統計:")
    print(f"  平均每事件涉及群組數: {clusters_per_event.mean():.1f}")
    print(f"  最小: {clusters_per_event.min()}")
    print(f"  最大: {clusters_per_event.max()}")
    print(f"  沒有群組參與的事件: {(clusters_per_event == 0).sum()}")
    
    # 整體資訊
    cluster_info = {
        'n_clusters': int(n_clusters),
        'n_events': int(n_events),
        'n_locations': int(n_locations),
        'n_isolated': int((cluster_labels == -1).sum()),
        'participation_threshold': float(participation_threshold),
        'clusters': cluster_info_list,
        'matrix_statistics': {
            'density': float(cluster_event_matrix.sum() / cluster_event_matrix.size),
            'events_per_cluster': {
                'mean': float(events_per_cluster.mean()),
                'min': int(events_per_cluster.min()),
                'max': int(events_per_cluster.max())
            },
            'clusters_per_event': {
                'mean': float(clusters_per_event.mean()),
                'min': int(clusters_per_event.min()),
                'max': int(clusters_per_event.max())
            }
        }
    }
    
    return cluster_event_matrix, cluster_info


def save_cluster_event_matrix(cluster_event_matrix, cluster_info, 
                              period_name, output_dir):
    """
    保存群組×事件矩陣和資訊
    
    Parameters:
    -----------
    cluster_event_matrix : np.ndarray
        群組×事件矩陣
    cluster_info : dict
        群組資訊
    period_name : str
        時期名稱
    output_dir : Path
        輸出目錄
    """
    # 保存矩陣
    matrix_path = output_dir / 'cluster_events' / f'cluster_event_matrix_{period_name}.npy'
    save_numpy(cluster_event_matrix, matrix_path)
    
    # 保存資訊
    info_path = output_dir / 'cluster_events' / f'cluster_event_info_{period_name}.json'
    save_json(cluster_info, info_path)
    
    print(f"\n✓ 已保存:")
    print(f"  矩陣: {matrix_path}")
    print(f"  資訊: {info_path}")


def run_cluster_event_matrix_creation(event_matrix=None, cluster_labels=None,
                                      period_name=None, config=None, 
                                      output_dir=None, 
                                      participation_threshold=0.4):
    """
    執行群組×事件矩陣建立的主函數
    
    Parameters:
    -----------
    event_matrix : np.ndarray, optional
        事件×地點矩陣
    cluster_labels : np.ndarray, optional
        分群標籤
    period_name : str, optional
        時期名稱
    config : Config, optional
        配置對象
    output_dir : Path, optional
        輸出目錄
    participation_threshold : float, optional
        參與度閾值 (預設 0.4)
    
    Returns:
    --------
    cluster_event_matrix : np.ndarray
        群組×事件矩陣
    cluster_info : dict
        群組資訊
    """
    if config is None:
        config = Config
    
    if output_dir is None:
        output_dir = config.PROCESSED_DATA_DIR
    
    print_step_header(7, "建立群組×事件矩陣", period_name)
    
    # 自動載入資料
    if event_matrix is None:
        if period_name is None:
            raise ValueError("必須提供 event_matrix 或 period_name")
        
        print("\n載入事件×地點矩陣...")
        filepath = output_dir / 'event_matrices' / f'event_matrix_{period_name}.npy'
        event_matrix = load_numpy(filepath)
        print(f"  形狀: {event_matrix.shape}")
    
    if cluster_labels is None:
        if period_name is None:
            raise ValueError("必須提供 cluster_labels 或 period_name")
        
        print("\n載入分群標籤...")
        filepath = output_dir / 'clusters' / f'clusters_{period_name}.npy'
        cluster_labels = load_numpy(filepath)
        print(f"  地點數: {len(cluster_labels)}")
        print(f"  群組數: {len(np.unique(cluster_labels[cluster_labels >= 0]))}")

        print(f"\n{'='*70}")
        print(f"參與度閾值: {participation_threshold} ({participation_threshold*100:.0f}%)")
        print(f"{'='*70}")
    
    # 建立群組×事件矩陣
    cluster_event_matrix, cluster_info = create_cluster_event_matrix(
        event_matrix,
        cluster_labels,
        participation_threshold=participation_threshold
    )
    
    # 保存結果
    save_cluster_event_matrix(
        cluster_event_matrix,
        cluster_info,
        period_name,
        output_dir
    )
    
    print(f"\n✓ 步驟7完成")
    
    return cluster_event_matrix, cluster_info