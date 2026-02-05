"""
通用工具函數
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime


def save_pickle(obj, filepath):
    """
    保存pickle文件
    
    Parameters:
    -----------
    obj : any
        要保存的對象
    filepath : str or Path
        保存路徑
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    
    print(f"✓ 已保存: {filepath}")


def load_pickle(filepath):
    """
    加載pickle文件
    
    Parameters:
    -----------
    filepath : str or Path
        文件路徑
    
    Returns:
    --------
    obj : any
        加載的對象
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    
    print(f"✓ 已加載: {filepath}")
    return obj


def save_numpy(array, filepath):
    """
    保存numpy數組
    
    Parameters:
    -----------
    array : np.ndarray
        要保存的數組
    filepath : str or Path
        保存路徑
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(filepath, array)
    print(f"✓ 已保存: {filepath}")


def load_numpy(filepath):
    """
    加載numpy數組
    
    Parameters:
    -----------
    filepath : str or Path
        文件路徑
    
    Returns:
    --------
    array : np.ndarray
        加載的數組
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    array = np.load(filepath)
    print(f"✓ 已加載: {filepath}")
    return array


def save_json(data, filepath):
    """
    保存JSON文件
    
    Parameters:
    -----------
    data : dict or list
        要保存的數據
    filepath : str or Path
        保存路徑
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已保存: {filepath}")


def save_csv(df, filepath):
    """
    保存CSV文件
    
    Parameters:
    -----------
    df : pd.DataFrame
        要保存的DataFrame
    filepath : str or Path
        保存路徑
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"✓ 已保存: {filepath}")


def print_step_header(step_num, step_name, period_name=None):
    """
    打印步驟標題
    
    Parameters:
    -----------
    step_num : int
        步驟編號
    step_name : str
        步驟名稱
    period_name : str, optional
        時期名稱
    """
    print("\n" + "="*70)
    if period_name:
        print(f"步驟{step_num}：{step_name} - {period_name}")
    else:
        print(f"步驟{step_num}：{step_name}")
    print("="*70)


def print_matrix_statistics(matrix, name):
    """
    打印矩陣統計信息
    
    Parameters:
    -----------
    matrix : np.ndarray
        要統計的矩陣
    name : str
        矩陣名稱
    """
    print(f"\n{name} 統計:")
    print(f"  形狀: {matrix.shape}")
    print(f"  數據類型: {matrix.dtype}")
    print(f"  範圍: {matrix.min():.3f} ~ {matrix.max():.3f}")
    print(f"  平均: {matrix.mean():.3f}")
    
    if matrix.dtype in [np.int8, np.int16, np.int32, np.int64, bool]:
        print(f"  非零: {np.count_nonzero(matrix)} ({np.count_nonzero(matrix)/matrix.size*100:.1f}%)")


def get_timestamp():
    """
    獲取當前時間戳
    
    Returns:
    --------
    timestamp : str
        格式化的時間戳
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")