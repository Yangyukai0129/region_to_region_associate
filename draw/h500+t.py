import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 1. 讀取 ERA5 或 NCEP 資料
ds = xr.open_dataset(r'D:\diffusion\download_nc\merged_geopotential.nc')
ds = ds.squeeze('pressure_level')
ds_t = xr.open_dataset(r'D:\diffusion\download_nc\0.05\merged_0.05.nc')
ds_t = ds_t.squeeze('pressure_level')  # 確保是1000hPa那層

# 2. 篩選 6-8 月並取每日中午 12 點的資料 (避免日變化干擾)
ds_summer = ds.sel(valid_time=ds.valid_time.dt.month.isin([6, 7, 8]))
ds_12pm = ds_summer.sel(valid_time=ds_summer.valid_time.dt.hour == 12)
ds_t_summer = ds_t.sel(valid_time=ds_t.valid_time.dt.month.isin([6, 7, 8]))
ds_t_12pm = ds_t_summer.sel(valid_time=ds_t_summer.valid_time.dt.hour == 12)

# 3. 計算 1965-2024 的長期平均態 (Climatology)
climatology = ds_12pm.z.mean(dim='valid_time')
clim_t = ds_t_12pm.t.mean(dim='valid_time')

# 4. 挑選你剛才找出的前五名高頻年份
target_years = [1991, 1998, 2003, 2006, 2007, 2010, 2012, 2018, 2019, 2020, 2021, 2022, 2024]
composite_data = ds_12pm.sel(valid_time=ds_12pm.valid_time.dt.year.isin(target_years))
composite_t = ds_t_12pm.sel(valid_time=ds_t_12pm.valid_time.dt.year.isin(target_years))

# 5. 計算合成距平 (Composite Anomaly)
# 單位換算: Geopotential (m^2/s^2) 除以 9.80665 得到位勢米 (gpm)
anomaly = (composite_data.z.mean(dim='valid_time') - climatology) / 9.80665
anomaly_t = composite_t.t.mean(dim='valid_time') - clim_t  # 單位已是 K，不需換算

print(f"T1000 距平最小值: {float(anomaly_t.min()):.2f} K")
print(f"T1000 距平最大值: {float(anomaly_t.max()):.2f} K")

# 6. 設定繪圖視窗
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree()) # 圓柱投影，與 NOAA 圖一致

# 設定範圍 (20N-80N, 全球經度)
ax.set_extent([-180, 180, 20, 80], crs=ccrs.PlateCarree())

# 繪製距平填色圖
levels = np.linspace(-30, 30, 21) # 設定色階，確保能看到 Wave-7 微小特徵
cf = anomaly.plot.contourf(
    ax=ax, transform=ccrs.PlateCarree(),
    levels=levels, cmap='RdBu_r', extend='both',
    add_colorbar=True, cbar_kwargs={'label': 'H500 Anomaly (gpm)'}
)

# T1000 距平：疊加等值線
levels_t = np.linspace(-1.5, 1.5, 10)  # ±4K，依你的資料調整
ct = anomaly_t.plot.contour(
    ax=ax, transform=ccrs.PlateCarree(),
    levels=levels_t, colors='black', linewidths=0.8
)
ax.clabel(ct, fmt='%+.1f', fontsize=8)  # 標上數值

# 加入地圖特徵
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

plt.title(f'T1000 & H500 ano', fontsize=14)

# txt_content = f"Jun to Aug: {', '.join(map(str, target_years))}."
# fig.text(0.45, 0.4, txt_content, ha='center', va='center', 
#          fontsize=11, style='italic', fontweight='bold',
#          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0))

# 8. 調整佈局 (調整間距確保不留白)
plt.subplots_adjust(bottom=0.25, top=0.9)

plt.show()