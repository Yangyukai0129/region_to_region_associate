"""
測試步驟3-5：從事件矩陣到共識矩陣
"""

import sys
sys.path.append('.')

from config.config import Config
from src.step3_similarity import run_similarity_calculation
from src.step4_louvain import run_louvain_clustering
from src.step5_consensus import run_consensus_building


def main():
    # 選擇要處理的時期
    period_name = "1965-2024"
    
    print("\n" + "="*70)
    print(f"測試步驟3-5: {period_name}")
    print("="*70)
    
    # 步驟3：計算相似度
    similarity_matrix = run_similarity_calculation(
        event_matrix=None,  # 自動載入
        period_name=period_name,
        config=Config
    )
    
    # 步驟4：Louvain社群檢測
    clustering_results = run_louvain_clustering(
        similarity_matrix=similarity_matrix,
        period_name=period_name,
        config=Config
    )
    
    # 步驟5：建立共識矩陣
    consensus_matrix = run_consensus_building(
        clustering_results=clustering_results,
        n_locations=similarity_matrix.shape[0],
        period_name=period_name,
        config=Config
    )
    
    print("\n" + "="*70)
    print("✓ 步驟3-5完成！")
    print("="*70)
    print(f"\n產生的檔案：")
    print(f"  1. {Config.PROCESSED_DATA_DIR}/similarity_matrices/similarity_{period_name}.npy")
    print(f"  2. {Config.PROCESSED_DATA_DIR}/louvain_results/louvain_results_{period_name}.pkl")
    print(f"  3. {Config.PROCESSED_DATA_DIR}/consensus_matrices/consensus_{period_name}.npy")


if __name__ == "__main__":
    main()