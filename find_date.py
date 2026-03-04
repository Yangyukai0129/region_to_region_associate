import numpy as np
import pandas as pd
import json

# 1. 載入資料 (加上 r 確保路徑讀取正確)
matrix_path = r'data\processed\cluster_events\cluster_event_matrix_1965-2024.npy'
info_path = r'data\processed\event_matrices\event_info_1965-2024.json'

# 載入矩陣 (shape: n_clusters, n_events)
cluster_event_matrix = np.load(matrix_path)

# 載入事件詳細資訊，指定編碼
with open(info_path, 'r', encoding='utf-8') as f:
    event_info = json.load(f)

# 2. 定位規則：找出 Cluster 0, 1, 2 同時活躍的事件 ID
# sync_mask = (cluster_event_matrix[0] == 1) & \
#             (cluster_event_matrix[1] == 1) & \
#             (cluster_event_matrix[2] == 1)
sync_mask = (cluster_event_matrix[0] == 1) & \
            (cluster_event_matrix[1] == 1)

target_event_ids = np.where(sync_mask)[0]
print(f"符合規則 {{0, 1, 2}} 同步的事件總數: {len(target_event_ids)}")

# 3. 執行日期轉換
results = []
for eid in target_event_ids:
    if eid < len(event_info): # 增加保護，避免 IndexError
        ctime = event_info[eid]['center_time']
        # 核心轉換：1970-01-01 起算的天數
        actual_date = pd.to_datetime(ctime, unit='D', origin='1970-01-01')
        
        results.append({
            'event_id': eid,
            'date': actual_date,
            'year': actual_date.year
        })

df_sync = pd.DataFrame(results)

# 4. 統計高頻年份與輸出清單
if not df_sync.empty:
    top_years = df_sync['year'].value_counts().head(5)
    all_years = df_sync['year'].value_counts()
    print(df_sync['year'].value_counts())

    print("\n=== 規則發生頻率所有年份 ===")
    print(df_sync)

    print("\n=== 規則發生頻率最高的前 10 名年份 (用於 NOAA 繪圖) ===")
    print(top_years)
    
    # 格式化輸出，方便直接複製到 NOAA 網頁
    year_list = sorted(top_years.index.astype(str).tolist())
    all_year_list = sorted(all_years.index.astype(str).tolist())
    print(f"\n建議輸入 NOAA 的年份清單: {', '.join(year_list)}")
    print(', '.join(all_year_list))
else:
    print("找不到符合條件的事件。")