"""
配置文件：統一管理所有參數
"""

import os
from pathlib import Path

class Config:
    """分析流程的所有配置參數"""
    
    # ===== 路徑配置 =====
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # 資料路徑
    RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'D:\heatwave_event_synchronize_study\heatwave_starts_95threshold_2deg_k3.nc'
    PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
    
    # ===== 時間段配置 =====
    TIME_PERIODS = [
        ("1965-06-01", "2024-08-31", "1965-2024"),
    ]
    
    # ===== 步驟1: ES參數 =====
    TAU_MAX = 10                # 時間窗口（天）
    MIN_Q = 0.5                 # Q值閾值（過濾弱同步）
    ES_N_JOBS = -1              # 並行核心數（-1表示使用所有核心）
    
    # ===== 步驟2: 事件矩陣參數 =====
    DBSCAN_EPS = 10             # 時間聚類窗口（天）
    DBSCAN_MIN_SAMPLES = 3      # 最小樣本數（定義事件的最小配對數）

    # ===== 步驟3: 相似度計算參數 =====
    SIMILARITY_METRIC = 'jaccard'  # 相似度度量方式
    
    # ===== 步驟4: Louvain參數 =====
    NUM_LOUVAIN_RUNS = 100      # Louvain運行次數
    LOUVAIN_GAMMA = 1         # 解析度參數（控制社群大小）
    MIN_SIMILARITY = 0.01        # 建立網絡的最小相似度閾值
    
    # ===== 步驟5: 共識矩陣參數 =====
    TOP_N_PARTITIONS = 25       # 取模組度最高的前N次結果

    # ===== 步驟6: 階層式分群參數 =====
    # K值選擇範圍
    MIN_CLUSTERS = 1            # 最小分群數
    MAX_CLUSTERS = 200          # 最大分群數
    
    # 約束條件
    MIN_CLUSTER_SIZE = 6        # 每群最少地點數
    
    # 階層式分群方法
    LINKAGE_METHOD = 'average'     # 連結方法：'ward', 'complete', 'average'
    
    # 手動覆蓋（如果需要強制指定K值）
    FORCE_NUM_CLUSTERS = None   # 設為None則自動選擇，設為整數則強制使用該K值
    
    # ===== 輸出控制 =====
    VERBOSE = True              # 是否顯示詳細信息
    
    @classmethod
    def create_directories(cls):
        """創建所有必要的目錄"""
        directories = [
            cls.PROCESSED_DATA_DIR / 'es_results',
            cls.PROCESSED_DATA_DIR / 'event_matrices',
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        if cls.VERBOSE:
            print("✓ 目錄結構已創建")
    
    @classmethod
    def get_period_info(cls, period_name):
        """
        根據時期名稱獲取時期資訊
        
        Parameters:
        -----------
        period_name : str
            時期名稱，例如 '1965-1974'
        
        Returns:
        --------
        tuple or None
            (start_date, end_date, period_name) 或 None（如果未找到）
        """
        for period in cls.TIME_PERIODS:
            if period[2] == period_name:
                return period
        return None


# 創建目錄
Config.create_directories()