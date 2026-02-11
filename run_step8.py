"""
執行步驟8：FP-Growth 群組關聯規則挖掘
"""

import sys
sys.path.append('.')

from config.config import Config
from src.step8_association import run_association_mining


def main():
    # 選擇要處理的時期
    period_name = "1965-2024"
    
    print("\n" + "="*70)
    print(f"執行步驟8: FP-Growth 關聯規則挖掘 - {period_name}")
    print("="*70)
    
    # 設定參數
    min_support = 0.2      # 5% 支持度
    min_confidence = 0.5    # 40% 信賴度
    
    print(f"\n參數設定:")
    print(f"  最小支持度: {min_support} ({min_support*100:.0f}%)")
    print(f"  最小信賴度: {min_confidence} ({min_confidence*100:.0f}%)")
    
    # 執行 FP-Growth
    frequent_itemsets, rules = run_association_mining(
        period_name=period_name,
        config=Config,
        min_support=min_support,
        min_confidence=min_confidence
    )
    
    # 結果摘要
    if len(frequent_itemsets) > 0 or len(rules) > 0:
        print("\n" + "="*70)
        print("✓ 步驟8完成!")
        print("="*70)
        
        print(f"\n結果摘要:")
        print(f"  頻繁項目集: {len(frequent_itemsets)} 個")
        print(f"  關聯規則: {len(rules)} 條")
        
        print(f"\n產生的檔案:")
        if len(frequent_itemsets) > 0:
            print(f"  1. {Config.PROCESSED_DATA_DIR}/fpgrowth/frequent_itemsets_{period_name}.csv")
            print(f"  2. {Config.PROCESSED_DATA_DIR}/fpgrowth/frequent_itemsets_{period_name}.json")
        if len(rules) > 0:
            print(f"  3. {Config.PROCESSED_DATA_DIR}/fpgrowth/association_rules_{period_name}.csv")
            print(f"  4. {Config.PROCESSED_DATA_DIR}/fpgrowth/association_rules_{period_name}.json")
        
        # 建議
        print(f"\n分析建議:")
        if len(rules) == 0 and len(frequent_itemsets) > 0:
            print(f"  ⚠️ 有頻繁項目集但無關聯規則")
            print(f"     建議: 降低 min_confidence (當前 {min_confidence})")
        elif len(frequent_itemsets) == 0:
            print(f"  ⚠️ 沒有頻繁項目集")
            print(f"     建議: 降低 min_support (當前 {min_support})")
            print(f"     或在步驟7降低 participation_threshold")
        else:
            print(f"  ✓ 成功挖掘出群組關聯模式")
            print(f"     可進一步分析規則的地理意義")
    else:
        print("\n✗ 步驟8未產生結果")
        print("\n可能原因:")
        print(f"  1. 支持度閾值過高 (當前 {min_support})")
        print(f"  2. 事件太少或群組參與度低")
        print(f"  3. 群組之間缺乏關聯")
        
        print("\n建議:")
        print(f"  1. 降低 min_support 到 0.01-0.03")
        print(f"  2. 在步驟7降低 participation_threshold")
        print(f"  3. 檢查群組×事件矩陣是否有效")


if __name__ == "__main__":
    main()