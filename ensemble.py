import os
import glob
import pandas as pd
import numpy as np
from src.utils import load_config

def main():
    # 1. 找到所有 output 目录下的预测结果 (假设你在 train.py 里保存了 val_preds.csv)
    # 你需要在 train.py 结束时把验证集的预测结果保存下来
    result_files = glob.glob("./output/*/val_predictions.csv")
    
    if not result_files:
        print("No prediction files found in ./output/")
        return

    print(f"Found {len(result_files)} models to ensemble.")
    
    # 2. 读取并简单的平均融合 (Simple Averaging)
    dfs = [pd.read_csv(f) for f in result_files]
    
    # 假设 csv 里有一列 'prob' 是预测概率
    ensemble_prob = np.zeros(len(dfs[0]))
    for df in dfs:
        ensemble_prob += df['prob'].values
    
    ensemble_prob /= len(dfs)
    
    # 3. 生成最终提交
    submission = dfs[0].copy()
    submission['prob'] = ensemble_prob
    submission.to_csv("submission.csv", index=False)
    print("Ensemble submission saved to submission.csv")

if __name__ == "__main__":
    main()
