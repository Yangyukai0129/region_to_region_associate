"""
測試步驟7：建立群組×事件矩陣
"""
import sys
import numpy as np
sys.path.append('.')

from config.config import Config
from src.step7_cluster_events import run_cluster_event_matrix_creation


def main():
    period_name = "1965-2024"
    
    print("\n" + "="*70)
    print(f"測試步驟7: 建立群組×事件矩陣 - {period_name}")
    print("="*70)
    
    # 測試不同閾值
    


        
    cluster_event_matrix, cluster_info = run_cluster_event_matrix_creation(
        period_name=period_name,
        config=Config,
        # participation_threshold=threshold
    )
        
    print(f"  矩陣形狀: {cluster_event_matrix.shape}")
    print(f"  矩陣密度: {cluster_info['matrix_statistics']['density']*100:.2f}%")
    print(f"  平均每群參與事件: {cluster_info['matrix_statistics']['events_per_cluster']['mean']:.1f}")


if __name__ == "__main__":
    main()