"""
測試步驟2：建立事件×空間矩陣 (優化版)
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
    # use_streaming=True: 串流模式,節省記憶體 (推薦)
    # use_streaming=False: 標準模式,一次載入所有資料
    event_matrix, event_info = run_event_matrix_creation(
        results=None,  # 自動載入 es_full_{period}.pkl
        n_locations=5400,  # 明確指定地點數 (或 None 自動推斷)
        period_name=period_name,
        config=Config,
        use_streaming=True  # ⭐ 使用串流模式節省記憶體
    )
    
    if event_matrix is not None:
        print("\n" + "="*70)
        print("✓ 步驟2完成！")
        print("="*70)
        print(f"\n產生的檔案：")
        print(f"  1. {Config.PROCESSED_DATA_DIR}/event_matrices/event_matrix_{period_name}.npy")
        print(f"  2. {Config.PROCESSED_DATA_DIR}/event_matrices/event_info_{period_name}.json")
        
        # 額外的診斷資訊
        print(f"\n" + "="*70)
        print("事件矩陣診斷")
        print("="*70)
        
        # 地點參與度分析
        events_per_location = event_matrix.sum(axis=0)
        active_locations = (events_per_location > 0).sum()
        empty_locations = (events_per_location == 0).sum()
        
        print(f"\n地點參與統計：")
        print(f"  總地點數: {len(events_per_location)}")
        print(f"  有事件的地點: {active_locations} ({active_locations/len(events_per_location)*100:.1f}%)")
        print(f"  無事件的地點: {empty_locations} ({empty_locations/len(events_per_location)*100:.1f}%)")
        
        print(f"\n每個地點參與的事件數：")
        print(f"  平均: {events_per_location.mean():.2f}")
        print(f"  中位數: {np.median(events_per_location):.1f}")
        print(f"  最小: {events_per_location.min()}")
        print(f"  最大: {events_per_location.max()}")
        print(f"  標準差: {events_per_location.std():.2f}")
        
        # 分布統計
        print(f"\n事件數分布:")
        bins = [0, 1, 5, 10, 20, 50, 100, 1000]
        for i in range(len(bins)-1):
            count = ((events_per_location >= bins[i]) & (events_per_location < bins[i+1])).sum()
            print(f"  {bins[i]:4d}-{bins[i+1]:4d} 個事件: {count:4d} 個地點 ({count/len(events_per_location)*100:5.1f}%)")
        
        # 事件規模分析
        print(f"\n事件規模統計：")
        event_sizes = [info['n_locations'] for info in event_info]
        print(f"  平均每事件涉及: {np.mean(event_sizes):.1f} 個地點")
        print(f"  中位數: {np.median(event_sizes):.1f}")
        print(f"  最小: {min(event_sizes)}")
        print(f"  最大: {max(event_sizes)}")
        
        # 建議
        print(f"\n" + "="*70)
        print("建議")
        print("="*70)
        
        if empty_locations / len(events_per_location) > 0.5:
            print(f"⚠️ 超過 50% 的地點沒有參與任何事件")
            print(f"   建議:")
            print(f"   1. 降低 MIN_Q 閾值 (當前: {Config.MIN_Q})")
            print(f"   2. 或在計算相似度前過濾這些空地點")
        else:
            print(f"✓ 地點參與度良好")
        
        print(f"\n下一步：")
        print(f"  執行步驟3: 計算 Jaccard 相似度")
        print(f"  python test_step3.py")
    else:
        print("\n✗ 步驟2失敗")
        print("\n可能原因:")
        print(f"  1. MIN_Q 閾值過高 (當前: {Config.MIN_Q})")
        print(f"  2. ES 結果中沒有足夠的同步配對")
        print(f"\n建議:")
        print(f"  降低 MIN_Q 閾值,例如設為 0.0 或 0.1")


if __name__ == "__main__":
    main()