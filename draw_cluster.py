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

def visualize_clusters_on_map(cluster_labels, lats, lons, 
                               output_path=None, figsize=(20, 12)):
    """
    在地圖上繪製群組分布
    
    Parameters:
    -----------
    cluster_labels : np.ndarray
        分群標籤 (5400,)
    lats : np.ndarray
        緯度座標 (30,)
    lons : np.ndarray
        經度座標 (180,)
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
    unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
    n_clusters = len(unique_clusters)
    
    print(f"\n繪製 {n_clusters} 個群組...")
    
    # 創建顏色映射
    import matplotlib.colors as mcolors
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    cmap = mcolors.ListedColormap(colors)
    
    # 創建地圖
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 添加地圖特徵
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    
    # 繪製群組
    mesh = ax.pcolormesh(
        lons, lats, cluster_grid,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=-0.5,
        vmax=n_clusters - 0.5,
        alpha=0.7
    )
    
    # 添加網格線
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    
    # 添加色條
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8)
    cbar.set_label('Cluster ID', fontsize=12)
    cbar.set_ticks(range(n_clusters))
    cbar.set_ticklabels([f'群組{i}' for i in unique_clusters])
    
    # 標題
    plt.title('熱浪事件同步群組分布', fontsize=16, fontweight='bold', pad=20)
    
    # 保存
    # if output_path:
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #     print(f"✓ 已保存: {output_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def visualize_individual_clusters(cluster_labels, lats, lons, 
                                  output_dir=None, figsize=(8, 6)):
    """
    為每個群組單獨繪製地圖
    
    Parameters:
    -----------
    cluster_labels : np.ndarray
        分群標籤
    lats, lons : np.ndarray
        座標
    output_dir : Path, optional
        輸出目錄
    """
    from pathlib import Path
    
    # 重塑
    n_lats = len(lats)
    n_lons = len(lons)
    cluster_grid = cluster_labels.reshape(n_lats, n_lons)
    
    # 找出有效群組
    unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
    
    print(f"\n為每個群組繪製單獨地圖...")
    
    for cluster_id in unique_clusters:
        # 創建遮罩
        mask = np.where(cluster_grid == cluster_id, 1, np.nan)
        
        # 統計
        n_locations = (cluster_grid == cluster_id).sum()
        
        # 創建圖
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # 地圖特徵
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        
        # 繪製群組
        mesh = ax.pcolormesh(
            lons, lats, mask,
            transform=ccrs.PlateCarree(),
            cmap='tab10',
            alpha=0.8
        )
        
        # 網格線
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        
        # 標題
        plt.title(f'群組 {cluster_id} ({n_locations} 個地點)', 
                 fontsize=14, fontweight='bold')
        
        # 保存
        if output_dir:
            output_path = Path(output_dir) / f'cluster_{cluster_id}_map.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ 群組{cluster_id}: {output_path}")
        
        plt.tight_layout()
        plt.show()
        plt.close()


def analyze_cluster_geography(cluster_labels, lats, lons):
    """
    分析每個群組的地理特徵
    
    Returns:
    --------
    cluster_stats : dict
        每個群組的地理統計
    """
    # 重塑
    n_lats = len(lats)
    n_lons = len(lons)
    cluster_grid = cluster_labels.reshape(n_lats, n_lons)
    
    # 找出有效群組
    unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
    
    print(f"\n" + "="*70)
    print("群組地理特徵分析")
    print("="*70)
    
    cluster_stats = {}
    
    for cluster_id in unique_clusters:
        # 找出屬於這個群組的地點
        mask = cluster_grid == cluster_id
        cluster_lats, cluster_lons = np.where(mask)
        
        # 轉為實際座標
        actual_lats = lats[cluster_lats]
        actual_lons = lons[cluster_lons]
        
        # 統計
        n_locations = len(actual_lats)
        
        stats = {
            'cluster_id': int(cluster_id),
            'n_locations': n_locations,
            'lat_range': (float(actual_lats.min()), float(actual_lats.max())),
            'lon_range': (float(actual_lons.min()), float(actual_lons.max())),
            'lat_center': float(actual_lats.mean()),
            'lon_center': float(actual_lons.mean()),
            'lat_std': float(actual_lats.std()),
            'lon_std': float(actual_lons.std())
        }
        
        cluster_stats[f'cluster_{cluster_id}'] = stats
        
        print(f"\n群組 {cluster_id}:")
        print(f"  地點數: {n_locations}")
        print(f"  緯度範圍: {stats['lat_range'][0]:.1f}° ~ {stats['lat_range'][1]:.1f}°")
        print(f"  經度範圍: {stats['lon_range'][0]:.1f}° ~ {stats['lon_range'][1]:.1f}°")
        print(f"  中心點: ({stats['lat_center']:.1f}°, {stats['lon_center']:.1f}°)")
        print(f"  空間分散度: 緯度±{stats['lat_std']:.1f}°, 經度±{stats['lon_std']:.1f}°")
    
    return cluster_stats


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
    
    # 1. 分析地理特徵
    # cluster_stats = analyze_cluster_geography(cluster_labels, lats, lons)
    
    # 保存統計
    # stats_path = Config.PROCESSED_DATA_DIR / 'clusters' / f'cluster_geography_{period_name}.json'
    # with open(stats_path, 'w') as f:
    #     json.dump(cluster_stats, f, indent=2)
    # print(f"\n✓ 已保存地理統計: {stats_path}")
    
    # 2. 繪製總覽地圖
    print(f"\n" + "="*70)
    print("繪製群組總覽地圖")
    print("="*70)
    
    output_path = Config.PROCESSED_DATA_DIR / 'clusters' / f'clusters_map_{period_name}.png'
    visualize_clusters_on_map(
        cluster_labels, lats, lons,
        output_path=output_path,
        figsize=(20, 10)
    )
    
    # 3. 繪製個別群組地圖
    print(f"\n" + "="*70)
    print("繪製個別群組地圖")
    print("="*70)
    
    # output_dir = Config.PROCESSED_DATA_DIR / 'clusters' / 'individual_maps'
    # output_dir.mkdir(exist_ok=True)
    
    visualize_individual_clusters(
        cluster_labels, lats, lons,
        output_dir=None,
        figsize=(10, 6)
    )
    
    print(f"\n✓ 視覺化完成!")


if __name__ == "__main__":
    main()