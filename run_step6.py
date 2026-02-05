"""
測試步驟6：階層式分群
"""

import sys
sys.path.append('.')

from config.config import Config
from src.step6_clustering import run_hierarchical_clustering
import numpy as np


def main():
    period_name = "1965-2024"
    
    print("\n" + "="*70)
    print(f"測試步驟6: {period_name}")
    print("="*70)
    
    # 先檢查共識矩陣
    print("\n檢查共識矩陣...")
    consensus_matrix = np.load(f'data/processed/consensus_matrices/consensus_{period_name}.npy')
    print(f"  形狀: {consensus_matrix.shape}")
    print(f"  平均: {consensus_matrix.mean():.4f}")
    
    upper = consensus_matrix[np.triu_indices_from(consensus_matrix, k=1)]
    print(f"  非對角線平均: {upper.mean():.4f}")
    print(f"  > 0的配對: {(upper > 0).sum():,} ({(upper > 0).sum()/len(upper)*100:.2f}%)")
    
    if consensus_matrix.mean() < 0.01:
        print("\n⚠️  警告：共識矩陣太稀疏（平均<0.01）")
        print("  建議：先解決步驟4-5的問題（降低MIN_SIMILARITY）")
        response = input("\n是否仍要繼續？(y/n): ")
        if response.lower() != 'y':
            return
    
    # 執行步驟6
    labels, best_k, silhouette_results = run_hierarchical_clustering(
        consensus_matrix=consensus_matrix,
        period_name=period_name,
        config=Config
    )
    
    print("\n" + "="*70)
    print("✓ 步驟6完成！")
    print("="*70)
    print(f"\n產生的檔案：")
    print(f"  1. {Config.PROCESSED_DATA_DIR}/clusters/clusters_{period_name}.npy")
    print(f"  2. {Config.PROCESSED_DATA_DIR}/clusters/silhouette_{period_name}.json")
    print(f"  3. {Config.PROCESSED_DATA_DIR}/clusters/cluster_info_{period_name}.json")
    
    print(f"\n最終結果：")
    print(f"  最佳K值: {best_k}")
    print(f"  分群標籤形狀: {labels.shape}")
    print(f"  群組標籤範圍: {labels.min()} ~ {labels.max()}")


if __name__ == "__main__":
    main()