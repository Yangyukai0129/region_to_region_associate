"""
測試步驟2：建立事件×空間矩陣
"""
import sys
import numpy as np
sys.path.append('.')

from config.config import Config
from src.step2_event_matrix import run_event_matrix_creation


def main():
    # 選擇要處理的時期
    period_name = "1965-2024"
    
    print("\n" + "="*70)
    print(f"測試步驟2: 建立事件矩陣 - {period_name}")
    print("="*70)
    
    print(f"\n當前參數：")
    print(f"  MIN_Q = {Config.MIN_Q}")
    print(f"  DBSCAN_EPS = {Config.DBSCAN_EPS}")
    print(f"  DBSCAN_MIN_SAMPLES = {Config.DBSCAN_MIN_SAMPLES}")
    
    # 步驟2：建立事件矩陣
    # results=None 會自動載入步驟1的結果
    event_matrix, event_info = run_event_matrix_creation(
        results=None,  # 自動載入 es_full_{period}.pkl
        n_locations=None,  # 或者 None 讓它自動推斷
        period_name=period_name,
        config=Config
    )
    
    if event_matrix is not None:
        print("\n" + "="*70)
        print("✓ 步驟2完成！")
        print("="*70)
        print(f"\n產生的檔案：")
        print(f"  1. {Config.PROCESSED_DATA_DIR}/event_matrices/event_matrix_{period_name}.npy")
        print(f"  2. {Config.PROCESSED_DATA_DIR}/event_matrices/event_info_{period_name}.json")
        
        # 額外的診斷資訊
        print(f"\n事件矩陣診斷：")
        events_per_location = event_matrix.sum(axis=0)
        print(f"  每個地點參與的事件數：")
        print(f"    平均: {events_per_location.mean():.1f}")
        print(f"    中位數: {np.median(events_per_location):.1f}")
        print(f"    最小: {events_per_location.min()}")
        print(f"    最大: {events_per_location.max()}")
        print(f"    0個事件的地點: {(events_per_location == 0).sum()} ({(events_per_location == 0).sum()/5400*100:.1f}%)")
        
        print(f"\n下一步：")
        print(f"  執行 python test_step3_5.py 來計算相似度和共識矩陣")
    else:
        print("\n✗ 步驟2失敗")


if __name__ == "__main__":
    main()