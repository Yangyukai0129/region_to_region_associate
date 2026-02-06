"""
步驟1：Event Synchronization (ES) 識別同步事件
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import sys
sys.path.append('.')

from config.config import Config
from src.utils import save_pickle, save_csv, print_step_header, get_timestamp


def compute_es(events_i, events_j, tau_max):
    """
    計算Event Synchronization同步強度
    
    Parameters:
    -----------
    events_i : np.ndarray
        地點i的時間序列，形狀(n_time,)，值為0或1
    events_j : np.ndarray
        地點j的時間序列，形狀(n_time,)，值為0或1
    tau_max : int
        最大時間窗口（天）
    
    Returns:
    --------
    Q : float
        標準化同步強度，範圍[0,1]
    es_ij : int
        同步事件計數
    pairs : np.ndarray
        同步時間配對，形狀(n_pairs, 2)，包含[(t_i, t_j), ...]
    """
    # 找出事件發生的時間點
    t_i = np.where(events_i == 1)[0]
    t_j = np.where(events_j == 1)[0]
    n_i, n_j = len(t_i), len(t_j)
    
    # 如果任一地點沒有事件，返回0
    if n_i == 0 or n_j == 0:
        return 0.0, 0, np.empty((0, 2), dtype=np.int32)
    
    # 計算事件間隔（用於動態閾值）
    intervals_i = np.diff(t_i) if n_i > 1 else np.array([])
    intervals_j = np.diff(t_j) if n_j > 1 else np.array([])
    
    # 計算每個事件的動態閾值 tau
    if n_i > 1:
        tau_i = np.minimum(
            np.concatenate(([np.inf], intervals_i)),
            np.concatenate((intervals_i, [np.inf]))
        )
    else:
        tau_i = np.array([np.inf])
    
    if n_j > 1:
        tau_j = np.minimum(
            np.concatenate(([np.inf], intervals_j)),
            np.concatenate((intervals_j, [np.inf]))
        )
    else:
        tau_j = np.array([np.inf])
    
    # 計算同步
    es_ij = 0
    pairs = []
    c_ij = 0.0
    
    for a in range(n_i):
        for b in range(n_j):
            t_ij = abs(t_i[a] - t_j[b])
            tau_ab = 0.5 * min(tau_i[a], tau_j[b])
            
            # 判斷是否同步
            if t_ij < tau_ab and t_ij <= tau_max:
                es_ij += 1
                pairs.append((int(t_i[a]), int(t_j[b])))
            
            # 計算標準化因子
            if 0 < t_ij <= tau_max:
                c_ij += 0.5 * min(t_ij, tau_max)
    
    # 標準化Q值
    if n_i * n_j > 0:
        Q = c_ij / np.sqrt(n_i * n_j)
    else:
        Q = 0.0
    
    Q = min(Q, 1.0)
    
    return Q, es_ij, np.array(pairs, dtype=np.int32)


def process_pair(i, j, events_array, tau_max):
    """
    處理單個地點對的ES計算
    
    Parameters:
    -----------
    i : int
        地點i的索引
    j : int
        地點j的索引
    events_array : np.ndarray
        事件資料，形狀(n_time, n_locations)
    tau_max : int
        最大時間窗口
    
    Returns:
    --------
    tuple
        (i, j, Q, es_ij, pairs)
    """
    Q, es_ij, pairs = compute_es(
        events_array[:, i],
        events_array[:, j],
        tau_max
    )
    return i, j, Q, es_ij, pairs


def compute_all_es_pairs(events_data, tau_max, n_jobs=-1):
    """
    計算所有地點對的ES
    
    Parameters:
    -----------
    events_data : pd.DataFrame or np.ndarray
        事件資料，形狀(n_time, n_locations)
    tau_max : int
        最大時間窗口
    n_jobs : int
        並行核心數
    
    Returns:
    --------
    results : list
        [(i, j, Q, es_ij, pairs), ...]
    """
    # 統一轉為 numpy array
    if isinstance(events_data, pd.DataFrame):
        events_array = events_data.values
    else:
        events_array = np.asarray(events_data)
    
    n_time, n_grid = events_array.shape
    n_pairs = n_grid * (n_grid - 1) // 2
    
    print(f"\n計算ES同步強度：")
    print(f"  地點數: {n_grid}")
    print(f"  地點對數: {n_pairs}")
    print(f"  並行核心: {n_jobs if n_jobs > 0 else '全部'}")
    print(f"  開始時間: {get_timestamp()}")
    
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(process_pair)(i, j, events_array, tau_max)
        for i in range(n_grid)
        for j in range(i + 1, n_grid)
    )
    
    print(f"  完成時間: {get_timestamp()}")
    
    return results


def save_es_results(results, lat, lon, period_name, output_dir):
    """
    保存ES計算結果
    
    Parameters:
    -----------
    results : list
        ES計算結果
    lat : np.ndarray
        緯度數組
    lon : np.ndarray
        經度數組
    period_name : str
        時期名稱
    output_dir : Path
        輸出目錄
    
    Returns:
    --------
    df_es : pd.DataFrame
        ES結果摘要
    """
    # 保存完整結果（包含pairs）
    filepath_pkl = output_dir / 'es_results' / f'es_full_{period_name}.pkl'
    save_pickle(results, filepath_pkl)
    
    # 建立摘要DataFrame
    df_es = pd.DataFrame([
        {
            'node_i': f"{lat[i]}_{lon[i]}",
            'node_j': f"{lat[j]}_{lon[j]}",
            'lat_i': lat[i],
            'lon_i': lon[i],
            'lat_j': lat[j],
            'lon_j': lon[j],
            'Q': Q,
            'es_ij': es_ij,
            'n_pairs': len(pairs)
        }
        for i, j, Q, es_ij, pairs in results
    ])
    
    # 保存摘要CSV
    filepath_csv = output_dir / 'es_results' / f'es_summary_{period_name}.csv'
    save_csv(df_es, filepath_csv)
    
    return df_es


def run_es_calculation(events_period, lat, lon, period_name, 
                       config=None, output_dir=None):
    """
    執行ES計算的主函數
    
    Parameters:
    -----------
    events_period : pd.DataFrame or xr.DataArray or np.ndarray
        時期事件資料，形狀(n_time, n_locations)
    lat : np.ndarray
        緯度數組
    lon : np.ndarray
        經度數組
    period_name : str
        時期名稱
    config : Config, optional
        配置對象
    output_dir : Path, optional
        輸出目錄
    
    Returns:
    --------
    results : list
        ES計算結果
    """
    if config is None:
        config = Config
    
    if output_dir is None:
        output_dir = config.PROCESSED_DATA_DIR
    
    print_step_header(1, "ES識別同步事件", period_name)
    
    # ⭐ 修正: 轉換為 numpy array 或 DataFrame
    if not isinstance(events_period, (pd.DataFrame, np.ndarray)):
        print("\n轉換資料格式...")
        if hasattr(events_period, 'values'):
            # xarray DataArray → numpy array
            events_period = events_period.values
            print(f"  已轉為 numpy array: {events_period.shape}")
        else:
            raise ValueError(f"不支援的資料類型: {type(events_period)}")
    
    # 驗證形狀
    if isinstance(events_period, pd.DataFrame):
        n_time, n_locations = events_period.shape
    else:
        n_time, n_locations = events_period.shape
    
    print(f"\n輸入資料:")
    print(f"  類型: {type(events_period).__name__}")
    print(f"  形狀: ({n_time}, {n_locations})")
    print(f"  時間點數: {n_time}")
    print(f"  地點數: {n_locations}")
    
    # 計算ES
    results = compute_all_es_pairs(
        events_period,
        tau_max=config.TAU_MAX,
        n_jobs=config.ES_N_JOBS
    )
    
    # 保存結果
    df_es = save_es_results(results, lat, lon, period_name, output_dir)
    
    # 打印統計
    print(f"\n✓ ES計算完成:")
    print(f"  地點對數: {len(results)}")
    print(f"  Q值範圍: {df_es['Q'].min():.3f} ~ {df_es['Q'].max():.3f}")
    print(f"  Q值平均: {df_es['Q'].mean():.3f}")
    print(f"  Q >= {config.MIN_Q}: {(df_es['Q'] >= config.MIN_Q).sum()} "
          f"({(df_es['Q'] >= config.MIN_Q).sum()/len(df_es)*100:.1f}%)")
    
    return results