"""
步驟2：從ES結果建立事件×空間矩陣
"""

import numpy as np
from sklearn.cluster import DBSCAN
import sys
sys.path.append('.')

from config.config import Config
from src.utils import (
    load_pickle, save_numpy, save_json,
    print_step_header, print_matrix_statistics
)


def create_event_matrix_from_es(results, n_locations, config=None):
    """
    從ES的pairs資訊建立事件×空間矩陣
    
    Parameters:
    -----------
    results : list
        ES計算結果，格式為[(i, j, Q, es_ij, pairs), ...]
    n_locations : int
        地點總數
    config : Config, optional
        配置對象
    
    Returns:
    --------
    event_matrix : np.ndarray
        事件×空間矩陣，形狀(n_events, n_locations)
    event_info : list
        事件詳細資訊列表
    """
    if config is None:
        config = Config
    
    print("\n【步驟2.1】收集同步時間配對...")
    
    all_sync_pairs = []
    
    for i, j, Q, es_ij, pairs in results:
        if Q >= config.MIN_Q:
            for t_i, t_j in pairs:
                all_sync_pairs.append({
                    'time_avg': (t_i + t_j) / 2,
                    'locations': [i, j],
                    't_i': int(t_i),
                    't_j': int(t_j)
                })
    
    n_pairs = len(all_sync_pairs)
    print(f"  收集到 {n_pairs} 個同步配對 (Q >= {config.MIN_Q})")
    
    if n_pairs == 0:
        print("  ⚠️  沒有符合條件的同步配對")
        return None, None
    
    # 時間聚類
    print("\n【步驟2.2】時間聚類識別事件...")
    print(f"  DBSCAN參數: eps={config.DBSCAN_EPS}, min_samples={config.DBSCAN_MIN_SAMPLES}")
    
    times = np.array([p['time_avg'] for p in all_sync_pairs]).reshape(-1, 1)
    
    clustering = DBSCAN(
        eps=config.DBSCAN_EPS,
        min_samples=config.DBSCAN_MIN_SAMPLES
    ).fit(times)
    
    labels = clustering.labels_
    unique_events = set(labels[labels >= 0])
    n_events = len(unique_events)
    n_noise = (labels == -1).sum()
    
    print(f"  識別出 {n_events} 個事件")
    print(f"  雜訊配對: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    
    # 建立事件×空間矩陣
    print("\n【步驟2.3】建立事件×空間矩陣...")
    
    event_matrix = np.zeros((n_events, n_locations), dtype=np.int8)
    event_info = []
    
    for event_id, cluster_label in enumerate(sorted(unique_events)):
        mask = labels == cluster_label
        event_pairs = [all_sync_pairs[i] for i in np.where(mask)[0]]
        
        participating_locations = set()
        time_points = []
        
        for pair in event_pairs:
            participating_locations.update(pair['locations'])
            time_points.append(pair['time_avg'])
        
        for loc in participating_locations:
            event_matrix[event_id, loc] = 1
        
        event_info.append({
            'event_id': int(event_id),
            'center_time': float(np.mean(time_points)),
            'time_range': (float(min(time_points)), float(max(time_points))),
            'n_locations': len(participating_locations),
            'locations': list(sorted(participating_locations))
        })
    
    return event_matrix, event_info


def save_event_matrix(event_matrix, event_info, period_name, output_dir):
    """
    保存事件矩陣和事件資訊
    
    Parameters:
    -----------
    event_matrix : np.ndarray
        事件×空間矩陣
    event_info : list
        事件詳細資訊
    period_name : str
        時期名稱
    output_dir : Path
        輸出目錄
    """
    filepath_npy = output_dir / 'event_matrices' / f'event_matrix_{period_name}.npy'
    save_numpy(event_matrix, filepath_npy)
    
    filepath_json = output_dir / 'event_matrices' / f'event_info_{period_name}.json'
    save_json(event_info, filepath_json)


def run_event_matrix_creation(results=None, n_locations=None, period_name=None,
                              config=None, output_dir=None):
    """
    執行事件矩陣建立的主函數
    
    Parameters:
    -----------
    results : list, optional
        ES計算結果，如果為None則自動載入
    n_locations : int, optional
        地點總數
    period_name : str, optional
        時期名稱
    config : Config, optional
        配置對象
    output_dir : Path, optional
        輸出目錄
    
    Returns:
    --------
    event_matrix : np.ndarray
        事件×空間矩陣
    event_info : list
        事件詳細資訊
    """
    if config is None:
        config = Config
    
    if output_dir is None:
        output_dir = config.PROCESSED_DATA_DIR
    
    print_step_header(2, "建立事件×空間矩陣", period_name)
    
    # 自動載入ES結果（如果未提供）
    if results is None:
        if period_name is None:
            raise ValueError("必須提供 results 或 period_name")
        
        print("\n自動載入ES結果...")
        filepath = output_dir / 'es_results' / f'es_full_{period_name}.pkl'
        results = load_pickle(filepath)
        
        # 從結果推斷地點數
        if n_locations is None:
            max_idx = max(max(i, j) for i, j, _, _, _ in results)
            n_locations = max_idx + 1
            print(f"  推斷地點數: {n_locations}")
    
    if n_locations is None:
        raise ValueError("必須提供 n_locations")
    
    # 建立事件矩陣
    event_matrix, event_info = create_event_matrix_from_es(
        results, n_locations, config
    )
    
    if event_matrix is None:
        print("\n✗ 無法建立事件矩陣")
        return None, None
    
    # 保存結果
    save_event_matrix(event_matrix, event_info, period_name, output_dir)
    
    # 打印統計
    print_matrix_statistics(event_matrix, "\n事件×空間矩陣")
    
    n_locs = [info['n_locations'] for info in event_info]
    print(f"\n事件統計:")
    print(f"  事件總數: {len(event_info)}")
    print(f"  平均每事件涉及: {np.mean(n_locs):.1f} 個地點")
    print(f"  地點數範圍: {min(n_locs)} ~ {max(n_locs)}")
    
    print(f"\n✓ 事件矩陣建立完成")
    
    return event_matrix, event_info