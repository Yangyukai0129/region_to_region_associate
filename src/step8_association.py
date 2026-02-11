"""
步驟8：FP-Growth 群組關聯規則挖掘
"""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import sys
sys.path.append('.')

from config.config import Config
from src.utils import (
    load_numpy, save_numpy, save_json,
    print_step_header, get_timestamp
)


def prepare_transactions(cluster_event_matrix):
    """
    將群組×事件矩陣轉為 FP-Growth 交易格式
    
    Parameters:
    -----------
    cluster_event_matrix : np.ndarray
        形狀 (n_clusters, n_events)
        值為 0 或 1
    
    Returns:
    --------
    transactions : list
        每個事件的群組列表
        例: [[0, 2], [1], [0, 1, 3], ...]
    df_transactions : pd.DataFrame
        One-hot 編碼 (n_events, n_clusters)
        FP-Growth 輸入格式
    """
    n_clusters, n_events = cluster_event_matrix.shape
    
    print(f"\n【準備交易資料】")
    print(f"  群組數: {n_clusters}")
    print(f"  事件數: {n_events}")
    
    # 方法1: 交易列表
    transactions = []
    empty_count = 0
    
    for event_id in range(n_events):
        # 找出參與這個事件的群組
        participating_clusters = np.where(cluster_event_matrix[:, event_id] == 1)[0]
        
        if len(participating_clusters) > 0:
            transactions.append(list(participating_clusters))
        else:
            empty_count += 1
    
    print(f"\n交易統計:")
    print(f"  有效交易: {len(transactions)}")
    print(f"  空交易(無群組): {empty_count}")
    
    if len(transactions) > 0:
        print(f"  平均每交易項目數: {np.mean([len(t) for t in transactions]):.2f}")
        
        # 項目數分布
        item_counts = [len(t) for t in transactions]
        for n in range(n_clusters + 1):
            count = item_counts.count(n)
            if count > 0:
                print(f"    {n} 個群組: {count} 事件 ({count/len(transactions)*100:.1f}%)")
    
    # 方法2: DataFrame (FP-Growth 需要)
    df_transactions = pd.DataFrame(
        cluster_event_matrix.T,  # 轉置: 事件×群組
        columns=[f'Cluster_{i}' for i in range(n_clusters)]
    ).astype(bool)
    
    # 移除空事件
    has_clusters = df_transactions.any(axis=1)
    df_transactions = df_transactions[has_clusters]
    
    print(f"\nDataFrame 形狀: {df_transactions.shape}")
    
    return transactions, df_transactions


def mine_frequent_itemsets(df_transactions, min_support):
    """
    使用 FP-Growth 挖掘頻繁項目集
    
    Parameters:
    -----------
    df_transactions : pd.DataFrame
        交易資料 (事件×群組)
    min_support : float
        最小支持度
    
    Returns:
    --------
    frequent_itemsets : pd.DataFrame
        頻繁項目集
    """
    print(f"\n【FP-Growth 挖掘】")
    print(f"  最小支持度: {min_support} ({min_support*100:.0f}%)")
    print(f"  開始時間: {get_timestamp()}")
    
    frequent_itemsets = fpgrowth(
        df_transactions,
        min_support=min_support,
        use_colnames=True
    )
    
    print(f"  完成時間: {get_timestamp()}")
    print(f"\n✓ 找到 {len(frequent_itemsets)} 個頻繁項目集")
    
    if len(frequent_itemsets) > 0:
        # 顯示 Top 10
        print(f"\n頻繁項目集 (Top 10 by support):")
        top_itemsets = frequent_itemsets.nlargest(min(10, len(frequent_itemsets)), 'support')
        
        for idx, row in top_itemsets.iterrows():
            itemset = ', '.join(sorted(row['itemsets']))
            support = row['support']
            n_items = len(row['itemsets'])
            print(f"  {itemset:30s} (size={n_items}): support={support:.3f}")
    
    return frequent_itemsets


def generate_association_rules(frequent_itemsets, min_confidence):
    """
    生成關聯規則
    
    Parameters:
    -----------
    frequent_itemsets : pd.DataFrame
        頻繁項目集
    min_confidence : float
        最小信賴度
    
    Returns:
    --------
    rules : pd.DataFrame
        關聯規則
    """
    print(f"\n【關聯規則生成】")
    print(f"  最小信賴度: {min_confidence} ({min_confidence*100:.0f}%)")
    
    if len(frequent_itemsets) == 0:
        print(f"  ⚠️ 沒有頻繁項目集,無法生成規則")
        return pd.DataFrame()
    
    try:
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )
        
        if len(rules) > 0:
            print(f"\n✓ 找到 {len(rules)} 條規則")
            
            # 排序並顯示
            rules_sorted = rules.sort_values('lift', ascending=False)
            
            print(f"\n強關聯規則 (Top 10 by lift):")
            for idx, row in rules_sorted.head(10).iterrows():
                ant = ', '.join(sorted(row['antecedents']))
                con = ', '.join(sorted(row['consequents']))
                
                print(f"\n  規則 {idx + 1}:")
                print(f"    {ant} → {con}")
                print(f"    Support: {row['support']:.3f}")
                print(f"    Confidence: {row['confidence']:.3f}")
                print(f"    Lift: {row['lift']:.3f}")
                print(f"    解釋: 當 {ant} 發生熱浪時,")
                print(f"          有 {row['confidence']*100:.1f}% 機率 {con} 也發生")
        else:
            print(f"\n  ⚠️ 沒有找到滿足條件的規則")
            print(f"     建議降低 min_confidence")
            rules = pd.DataFrame()
    
    except ValueError as e:
        print(f"\n  ⚠️ 無法生成規則: {e}")
        print(f"     可能原因: 沒有足夠的多項目集")
        rules = pd.DataFrame()
    
    return rules


def save_association_results(frequent_itemsets, rules, period_name, output_dir):
    """
    保存 FP-Growth 結果
    
    Parameters:
    -----------
    frequent_itemsets : pd.DataFrame
        頻繁項目集
    rules : pd.DataFrame
        關聯規則
    period_name : str
        時期名稱
    output_dir : Path
        輸出目錄
    """
    from pathlib import Path
    
    # 確保目錄存在
    fpgrowth_dir = Path(output_dir) / 'fpgrowth'
    fpgrowth_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存頻繁項目集
    if len(frequent_itemsets) > 0:
        itemsets_save = frequent_itemsets.copy()
        itemsets_save['itemsets'] = itemsets_save['itemsets'].apply(lambda x: sorted(list(x)))
        
        filepath_csv = fpgrowth_dir / f'frequent_itemsets_{period_name}.csv'
        itemsets_save.to_csv(filepath_csv, index=False)
        
        # 同時保存為 JSON
        itemsets_json = itemsets_save.to_dict('records')
        filepath_json = fpgrowth_dir / f'frequent_itemsets_{period_name}.json'
        save_json(itemsets_json, filepath_json)
        
        print(f"\n✓ 已保存頻繁項目集:")
        print(f"    CSV: {filepath_csv}")
        print(f"    JSON: {filepath_json}")
    else:
        print(f"\n  ⚠️ 沒有頻繁項目集可保存")
    
    # 保存關聯規則
    if len(rules) > 0:
        rules_save = rules.copy()
        rules_save['antecedents'] = rules_save['antecedents'].apply(lambda x: sorted(list(x)))
        rules_save['consequents'] = rules_save['consequents'].apply(lambda x: sorted(list(x)))
        
        filepath_csv = fpgrowth_dir / f'association_rules_{period_name}.csv'
        rules_save.to_csv(filepath_csv, index=False)
        
        # 同時保存為 JSON
        rules_json = rules_save.to_dict('records')
        filepath_json = fpgrowth_dir / f'association_rules_{period_name}.json'
        save_json(rules_json, filepath_json)
        
        print(f"\n✓ 已保存關聯規則:")
        print(f"    CSV: {filepath_csv}")
        print(f"    JSON: {filepath_json}")
    else:
        print(f"\n  ⚠️ 沒有關聯規則可保存")


def run_association_mining(cluster_event_matrix=None, period_name=None,
                           config=None, output_dir=None,
                           min_support=0.05, min_confidence=0.4):
    """
    執行關聯規則挖掘的主函數
    
    Parameters:
    -----------
    cluster_event_matrix : np.ndarray, optional
        群組×事件矩陣
        如果為None，自動從文件載入
    period_name : str, optional
        時期名稱
    config : Config, optional
        配置對象
    output_dir : Path, optional
        輸出目錄
    min_support : float, optional
        最小支持度 (預設 0.05 = 5%)
    min_confidence : float, optional
        最小信賴度 (預設 0.4 = 40%)
    
    Returns:
    --------
    frequent_itemsets : pd.DataFrame
        頻繁項目集
    rules : pd.DataFrame
        關聯規則
    """
    if config is None:
        config = Config
    
    if output_dir is None:
        output_dir = config.PROCESSED_DATA_DIR
    
    print_step_header(8, "FP-Growth 群組關聯規則挖掘", period_name)
    
    # 自動載入群組×事件矩陣（如果未提供）
    if cluster_event_matrix is None:
        if period_name is None:
            raise ValueError("必須提供 cluster_event_matrix 或 period_name")
        
        print("\n自動載入群組×事件矩陣...")
        filepath = output_dir / 'cluster_events' / f'cluster_event_matrix_{period_name}.npy'
        cluster_event_matrix = load_numpy(filepath)
    
    print(f"\n群組×事件矩陣:")
    print(f"  形狀: {cluster_event_matrix.shape}")
    print(f"  密度: {cluster_event_matrix.sum() / cluster_event_matrix.size * 100:.1f}%")
    
    # 步驟1：準備交易資料
    transactions, df_transactions = prepare_transactions(cluster_event_matrix)
    
    if len(df_transactions) == 0:
        print("\n❌ 錯誤: 沒有有效交易!")
        return pd.DataFrame(), pd.DataFrame()
    
    # 步驟2：挖掘頻繁項目集
    try:
        frequent_itemsets = mine_frequent_itemsets(df_transactions, min_support)
    except Exception as e:
        print(f"\n❌ FP-Growth 失敗: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    # 步驟3：生成關聯規則
    rules = generate_association_rules(frequent_itemsets, min_confidence)
    
    # 保存結果
    save_association_results(frequent_itemsets, rules, period_name, output_dir)
    
    print(f"\n✓ 步驟8完成")
    
    return frequent_itemsets, rules