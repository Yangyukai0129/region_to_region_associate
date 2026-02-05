"""
測試步驟1-2：從熱浪資料到事件×空間矩陣
"""

import xarray as xr
import sys
sys.path.append('.')

from config.config import Config
from src.step1_es import run_es_calculation
from src.step2_event_matrix import run_event_matrix_creation


def main():
    # 選擇要處理的時期
    period_name = "1965-2024"
    
    print("\n" + "="*70)
    print(f"測試步驟1-2: {period_name}")
    print("="*70)
    
    # 讀取資料
    print("\n讀取熱浪資料...")
    heatwave_events = xr.open_dataset(Config.RAW_DATA_PATH)['t']
    
    # 獲取時期資訊
    period_info = Config.get_period_info(period_name)
    if period_info is None:
        print(f"錯誤：未找到時期 {period_name}")
        return
    
    start_date, end_date, _ = period_info
    print(f"時期: {start_date} ~ {end_date}")
    
    # 切分資料
    events_period = heatwave_events.sel(valid_time=slice(start_date, end_date))
    events_flat = events_period.stack(grid=['latitude', 'longitude']).transpose('valid_time', 'grid')
    
    # 提取座標
    grid_index = events_flat['grid'].to_index()
    lat = grid_index.get_level_values('latitude').values
    lon = grid_index.get_level_values('longitude').values
    n_locations = len(lat)
    
    print(f"資料形狀: {events_flat.shape}")
    print(f"地點數: {n_locations}")
    
    # 步驟1：ES計算
    results = run_es_calculation(
        events_flat,
        lat,
        lon,
        period_name,
        config=Config
    )
    
    # 步驟2：建立事件矩陣
    event_matrix, event_info = run_event_matrix_creation(
        results,
        n_locations,
        period_name,
        config=Config
    )
    
    if event_matrix is not None:
        print("\n" + "="*70)
        print("✓ 步驟1-2完成！")
        print("="*70)
        print(f"\n產生的檔案：")
        print(f"  1. {Config.PROCESSED_DATA_DIR}/es_results/es_full_{period_name}.pkl")
        print(f"  2. {Config.PROCESSED_DATA_DIR}/es_results/es_summary_{period_name}.csv")
        print(f"  3. {Config.PROCESSED_DATA_DIR}/event_matrices/event_matrix_{period_name}.npy")
        print(f"  4. {Config.PROCESSED_DATA_DIR}/event_matrices/event_info_{period_name}.json")


if __name__ == "__main__":
    main()