"""
視覺化群組的地理分布
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import sys
sys.path.append('.')

from config.config import Config
from src.utils import load_numpy
import json

# 解決中文顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
# 解決負號 '-' 顯示為方塊的問題
plt.rcParams['axes.unicode_minus'] = False

def visualize_selected_clusters(cluster_labels, lats, lons, 
                                selected_clusters=None,
                                output_path=None, figsize=(20, 12)):
    """
    在地圖上繪製選定的群組分布
    
    Parameters:
    -----------
    cluster_labels : np.ndarray
        分群標籤 (5400,)
    lats : np.ndarray
        緯度座標 (30,)
    lons : np.ndarray
        經度座標 (180,)
    selected_clusters : list, optional
        要繪製的群組ID列表，例如 [0, 2, 3]
        如果為 None，繪製所有群組
    output_path : str, optional
        儲存路徑
    figsize : tuple
        圖片大小
    """
    # 重塑標籤為 2D 網格
    n_lats = len(lats)
    n_lons = len(lons)
    cluster_grid = cluster_labels.reshape(n_lats, n_lons)
    
    # 找出有效群組
    all_clusters = np.unique(cluster_labels[cluster_labels >= 0])
    
    # 決定要繪製哪些群組
    if selected_clusters is None:
        selected_clusters = list(all_clusters)
    else:
        # 檢查選擇的群組是否存在
        selected_clusters = [c for c in selected_clusters if c in all_clusters]
        if len(selected_clusters) == 0:
            print("⚠️ 選擇的群組都不存在!")
            return None, None
    
    n_selected = len(selected_clusters)
    print(f"\n繪製選定的 {n_selected} 個群組: {selected_clusters}")
    
    # 創建遮罩: 只顯示選定的群組
    masked_grid = np.full_like(cluster_grid, -1, dtype=float)
    for idx, cluster_id in enumerate(selected_clusters):
        masked_grid[cluster_grid == cluster_id] = idx
    
    # 將未選擇的群組設為 NaN (不顯示)
    masked_grid = np.where(masked_grid >= 0, masked_grid, np.nan)
    
    # 創建顏色映射
    import matplotlib.colors as mcolors
    colors = plt.cm.Set3(np.linspace(0, 1, n_selected))
    cmap = mcolors.ListedColormap(colors)
    
    # 創建地圖
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 添加地圖特徵
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    
    # 繪製選定的群組
    mesh = ax.pcolormesh(
        lons, lats, masked_grid,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=-0.5,
        vmax=n_selected - 0.5,
        alpha=0.7
    )
    
    # 添加網格線
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    
    # 添加色條
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8)
    cbar.set_label('Cluster ID', fontsize=12)
    cbar.set_ticks(range(n_selected))
    cbar.set_ticklabels([f'群組{c}' for c in selected_clusters])
    
    # 統計每個群組的地點數
    cluster_sizes = []
    for cluster_id in selected_clusters:
        n_locs = (cluster_grid == cluster_id).sum()
        cluster_sizes.append(n_locs)
    
    # 標題
    title = f'cluster0, cluster1 → cluster2'
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 保存
    # if output_path:
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #     print(f"✓ 已保存: {output_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def main():
    """主函數"""
    period_name = "1965-2024"
    
    # 載入分群結果
    print("載入分群標籤...")
    cluster_labels = load_numpy(
        Config.PROCESSED_DATA_DIR / 'clusters' / f'clusters_{period_name}.npy'
    )
    
    # 載入座標
    print("載入座標...")
    ds = xr.open_dataset(Config.RAW_DATA_PATH)
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    
    print(f"\n資料資訊:")
    print(f"  地點數: {len(cluster_labels)}")
    print(f"  緯度網格: {len(lats)}")
    print(f"  經度網格: {len(lons)}")
    print(f"  群組數: {len(np.unique(cluster_labels[cluster_labels >= 0]))}")
    
    # 2. 繪製總覽地圖
    print(f"\n" + "="*70)
    print("繪製群組總覽地圖")
    print("="*70)
    
    output_path = Config.PROCESSED_DATA_DIR / 'clusters' / f'clusters_map_{period_name}.png'
    visualize_selected_clusters(
        cluster_labels, lats, lons,
        selected_clusters=[0, 1, 2],  # ← 自己決定要畫哪些群組
        figsize=(20, 10)
    )
    
    
    print(f"\n✓ 視覺化完成!")


if __name__ == "__main__":
    main()