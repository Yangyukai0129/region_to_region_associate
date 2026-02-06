"""
步驟2：從ES結果建立事件×空間矩陣 (記憶體優化版)
"""

import numpy as np
from sklearn.cluster import DBSCAN
import pickle
import sys
sys.path.append('.')

from config.config import Config
from src.utils import (
    save_numpy, save_json,
    print_step_header, print_matrix_statistics
)


def collect_sync_pairs_streaming(filepath, min_q, batch_size=50000):
    """
    串流式讀取 ES 結果,收集同步配對
    
    Parameters:
    -----------
    filepath : Path
        ES 結果文件路徑
    min_q : float
        Q 值閾值
    batch_size : int
        每次處理的配對數
    
    Yields:
    -------
    batch : list
        同步配對批次
    """
    print(f"\n【步驟2.1】串流式收集同步時間配對...")
    print(f"  MIN_Q 閾值: {min_q}")
    print(f"  批次大小: {batch_size}")
    
    batch = []
    total_pairs = 0
    accepted_pairs = 0
    
    print(f"  正在讀取文件...")
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
        total_count = len(results)
        print(f"  總地點對數: {total_count}")
        
        for idx, (i, j, Q, es_ij, pairs) in enumerate(results):
            total_pairs += 1
            
            # 只處理符合閾值的配對
            if Q >= min_q and len(pairs) > 0:
                for t_i, t_j in pairs:
                    batch.append({
                        'time_avg': (t_i + t_j) / 2,
                        'locations': [i, j],
                        't_i': int(t_i),
                        't_j': int(t_j)
                    })
                    accepted_pairs += 1
                    
                    # 批次已滿,yield 出去
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            
            # 進度顯示
            if (idx + 1) % 50000 == 0:
                print(f"  進度: {idx + 1}/{total_count} ({(idx + 1)/total_count*100:.1f}%), "
                      f"已收集配對: {accepted_pairs}")
    
    # 最後一批
    if batch:
        yield batch
    
    print(f"  完成: 收集到 {accepted_pairs} 個同步配對 (Q >= {min_q})")


def create_event_matrix_from_es(results, n_locations, config=None):
    """
    從ES的pairs資訊建立事件×空間矩陣 (原始版本,保留兼容性)
    
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


def create_event_matrix_from_es_streaming(filepath, n_locations, config=None):
    """
    從ES結果串流式建立事件×空間矩陣 (記憶體優化版)
    
    Parameters:
    -----------
    filepath : Path
        ES 結果文件路徑
    n_locations : int
        地點總數
    config : Config, optional
        配置對象
    
    Returns:
    --------
    event_matrix : np.ndarray
        事件×空間矩陣
    event_info : list
        事件詳細資訊
    """
    if config is None:
        config = Config
    
    # 第一遍: 收集所有同步配對的時間 (只存時間,省記憶體)
    print("\n【第一遍掃描】收集時間點用於聚類...")
    all_times = []
    
    # 第一遍: 收集時間點並去重
    print("\n【第一遍掃描】收集時間點用於聚類...")
    all_times_set = set()  # ← 改用 set 去重

    for batch in collect_sync_pairs_streaming(filepath, config.MIN_Q):
        times = [p['time_avg'] for p in batch]
        all_times_set.update(times)  # ← 自動去重
        
        # 控制記憶體
        if len(all_times_set) > 1000000:  # ← 降低上限到 100萬
            print(f"  ⚠️ 時間點過多 ({len(all_times_set)}),建議提高 MIN_Q")
            break

    all_times = sorted(all_times_set)  # ← 轉回 list 並排序
    del all_times_set  # ← 釋放記憶體

    if len(all_times) == 0:
        print("  ⚠️ 沒有符合條件的同步配對")
        return None, None

    print(f"  收集到 {len(all_times)} 個唯一時間點")
    
    if len(all_times) == 0:
        print("  ⚠️ 沒有符合條件的同步配對")
        return None, None
    
    print(f"  收集到 {len(all_times)} 個時間點")
    
    # DBSCAN 聚類
    print("\n【步驟2.2】時間聚類識別事件...")
    print(f"  DBSCAN參數: eps={config.DBSCAN_EPS}, min_samples={config.DBSCAN_MIN_SAMPLES}")
    
    times_array = np.array(all_times).reshape(-1, 1)
    clustering = DBSCAN(
        eps=config.DBSCAN_EPS,
        min_samples=config.DBSCAN_MIN_SAMPLES
    ).fit(times_array)
    
    labels = clustering.labels_
    unique_events = sorted(set(labels[labels >= 0]))
    n_events = len(unique_events)
    n_noise = (labels == -1).sum()
    
    print(f"  識別出 {n_events} 個事件")
    print(f"  雜訊配對: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    
    # 建立時間到事件的映射
    time_to_event = {}
    event_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_events)}
    
    for idx, label in enumerate(labels):
        if label >= 0:  # 忽略噪音
            time_to_event[all_times[idx]] = event_id_map[label]
    
    # 清理記憶體
    del times_array
    del all_times
    del labels
    
    # 第二遍: 建立事件矩陣
    print("\n【第二遍掃描】建立事件×空間矩陣...")
    event_matrix = np.zeros((n_events, n_locations), dtype=np.int8)
    
    # 記錄每個事件的資訊
    event_locations = [set() for _ in range(n_events)]
    event_times = [[] for _ in range(n_events)]
    
    batch_count = 0
    for batch in collect_sync_pairs_streaming(filepath, config.MIN_Q):
        batch_count += 1
        
        for pair in batch:
            time_avg = pair['time_avg']
            
            # 查找這個時間點屬於哪個事件
            if time_avg in time_to_event:
                event_id = time_to_event[time_avg]
                
                # 標記地點
                for loc in pair['locations']:
                    if loc < n_locations:  # 安全檢查
                        event_matrix[event_id, loc] = 1
                        event_locations[event_id].add(loc)
                
                # 記錄時間
                event_times[event_id].append(time_avg)
        
        if batch_count % 10 == 0:
            print(f"  已處理 {batch_count} 個批次")
    
    # 建立事件資訊
    print("\n【步驟2.3】整理事件資訊...")
    event_info = []
    
    for event_id in range(n_events):
        if len(event_times[event_id]) > 0:
            event_info.append({
                'event_id': int(event_id),
                'center_time': float(np.mean(event_times[event_id])),
                'time_range': (float(min(event_times[event_id])), 
                              float(max(event_times[event_id]))),
                'n_locations': len(event_locations[event_id]),
                'locations': list(sorted(event_locations[event_id]))
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
                              config=None, output_dir=None, use_streaming=True):
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
    use_streaming : bool, optional
        是否使用串流模式 (預設 True,節省記憶體)
    
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
    
    mode_str = "串流模式" if use_streaming else "標準模式"
    print_step_header(2, f"建立事件×空間矩陣 ({mode_str})", period_name)
    
    # 確定文件路徑
    filepath = output_dir / 'es_results' / f'es_full_{period_name}.pkl'
    
    if not filepath.exists():
        raise FileNotFoundError(f"找不到 ES 結果: {filepath}")
    
    # 顯示文件資訊
    file_size_mb = filepath.stat().st_size / 1024 / 1024
    print(f"\nES 結果文件: {filepath}")
    print(f"文件大小: {file_size_mb:.1f} MB")
    
    if file_size_mb > 1000:  # 超過 1GB
        print(f"⚠️ 文件較大,強烈建議使用串流模式")
        use_streaming = True
    
    # 推斷地點數 (如果未提供)
    if n_locations is None and results is None:
        print("\n推斷地點數...")
        with open(filepath, 'rb') as f:
            temp_results = pickle.load(f)
            # 只看前 1000 個來推斷
            sample = temp_results[:min(1000, len(temp_results))]
            max_idx = max(max(i, j) for i, j, _, _, _ in sample)
            n_locations = max_idx + 1
            del temp_results
        print(f"  推斷地點數: {n_locations}")
    
    # 選擇處理模式
    if use_streaming and results is None:
        # 串流模式: 不載入整個 results
        print(f"\n使用串流模式處理 (節省記憶體)")
        event_matrix, event_info = create_event_matrix_from_es_streaming(
            filepath, n_locations, config
        )
    else:
        # 標準模式: 載入整個 results
        if results is None:
            print("\n載入 ES 結果...")
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            print(f"  已載入 {len(results)} 個地點對")
            
            # 從結果推斷地點數
            if n_locations is None:
                max_idx = max(max(i, j) for i, j, _, _, _ in results)
                n_locations = max_idx + 1
                print(f"  推斷地點數: {n_locations}")
        
        print(f"\n使用標準模式處理")
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