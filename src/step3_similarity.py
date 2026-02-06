"""
步驟3：計算空間Jaccard相似度
"""

import numpy as np
import sys
sys.path.append('.')

from config.config import Config
from src.utils import (
    load_numpy, save_numpy,
    print_step_header, print_matrix_statistics
)


def compute_jaccard_similarity(event_matrix):
    """
    從事件×空間矩陣計算Jaccard相似度
    
    Parameters:
    -----------
    event_matrix : np.ndarray
        事件×空間矩陣，形狀(n_events, n_locations)
        值為0或1
    
    Returns:
    --------
    similarity_matrix : np.ndarray
        相似度矩陣，形狀(n_locations, n_locations)，dtype=float32
        值範圍[0,1]，對角線為1.0
        
    Notes:
    ------
    Jaccard相似度 = |A ∩ B| / |A ∪ B|
    - A ∩ B: 兩地點共同參與的事件數
    - A ∪ B: 至少一個地點參與的事件數
    
    計算複雜度：O(n_locations^2 * n_events)
    對於800個地點，需要計算319,600對
    """
    n_events, n_locations = event_matrix.shape
    similarity_matrix = np.zeros((n_locations, n_locations), dtype=np.float32)
    
    print(f"\n【步驟3.1】計算Jaccard相似度")
    print(f"  地點數: {n_locations}")
    print(f"  需計算: {n_locations * (n_locations - 1) // 2} 對")
    
    # 計算相似度
    for i in range(n_locations):
        # 打印進度
        if i % 100 == 0:
            progress = i / n_locations * 100
            print(f"  進度: {i}/{n_locations} ({progress:.1f}%)", end='\r')
        
        # 地點i參與的事件
        events_i = event_matrix[:, i]
        
        for j in range(i, n_locations):
            # 地點j參與的事件
            events_j = event_matrix[:, j]
            
            # 計算交集和聯集
            intersection = np.logical_and(events_i, events_j).sum()
            union = np.logical_or(events_i, events_j).sum()
            
            # 計算Jaccard
            if union > 0:
                jaccard = intersection / union
            else:
                jaccard = 0.0
            
            # 對稱矩陣
            similarity_matrix[i, j] = jaccard
            similarity_matrix[j, i] = jaccard
    
    print(f"  進度: {n_locations}/{n_locations} (100.0%)")
    print(f"✓ Jaccard相似度計算完成")
    
    return similarity_matrix


def save_similarity_matrix(similarity_matrix, period_name, output_dir):
    """
    保存相似度矩陣
    
    Parameters:
    -----------
    similarity_matrix : np.ndarray
        相似度矩陣
    period_name : str
        時期名稱
    output_dir : Path
        輸出目錄
    """
    filepath = output_dir / 'similarity_matrices' / f'similarity_{period_name}.npy'
    save_numpy(similarity_matrix, filepath)


def run_similarity_calculation(event_matrix=None, period_name=None,
                               config=None, output_dir=None):
    """
    執行相似度計算的主函數
    
    Parameters:
    -----------
    event_matrix : np.ndarray, optional
        事件×空間矩陣
        如果為None，自動從文件載入
    period_name : str, optional
        時期名稱，例如'1965-1974'
    config : Config, optional
        配置對象，默認使用Config
    output_dir : Path, optional
        輸出目錄，默認使用config.PROCESSED_DATA_DIR
    
    Returns:
    --------
    similarity_matrix : np.ndarray
        相似度矩陣，形狀(n_locations, n_locations)
    
    Notes:
    ------
    - 支援自動載入事件矩陣
    - 計算可能較耗時（800×800約需幾分鐘）
    """
    if config is None:
        config = Config
    
    if output_dir is None:
        output_dir = config.PROCESSED_DATA_DIR
    
    print_step_header(3, "計算Jaccard相似度", period_name)
    
    # 自動載入事件矩陣（如果未提供）
    if event_matrix is None:
        if period_name is None:
            raise ValueError("必須提供 event_matrix 或 period_name")
        
        print("\n自動載入事件矩陣...")
        filepath = output_dir / 'event_matrices' / f'event_matrix_{period_name}.npy'
        event_matrix = load_numpy(filepath)

    active_node_indices = np.where(event_matrix.sum(axis=0) > 0)[0]
    event_matrix_active = event_matrix[:, active_node_indices]
    
    print(f"原始格點數: {event_matrix.shape[1]}")
    print(f"活躍格點數 (篩選後): {event_matrix_active.shape[1]}")
    
    # 計算相似度
    similarity_matrix = compute_jaccard_similarity(event_matrix)
    
    # 保存結果
    save_similarity_matrix(similarity_matrix, period_name, output_dir)
    
    print(f"\n✓ 步驟3完成")
    
    return similarity_matrix