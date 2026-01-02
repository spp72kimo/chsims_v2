import pandas as pd
import numpy as np

# ============================================================
# 讀取資料
# ============================================================
df = pd.read_csv('fusion_Video_videoMAE_with_labels.csv')

# 只取 test set
test_df = df[df['mode'] == 'test'].copy()
print(f"Test set 樣本數: {len(test_df)}")
print()

# ============================================================
# 評估函數
# ============================================================
def calculate_metrics(pred, label):
    """
    計算評估指標: Acc2, Acc2_weak, Corr, MAE
    
    Parameters:
    -----------
    pred : Series, 預測值
    label : Series, 標籤值
    
    Returns:
    --------
    dict : 包含各項指標的字典
    """
    # 過濾有效資料
    valid_mask = ~(pd.isna(pred) | pd.isna(label))
    pred_valid = pred[valid_mask].values
    label_valid = label[valid_mask].values
    
    n_samples = len(pred_valid)
    
    # 1. Acc2 (排除 label=0)
    acc2_mask = label_valid != 0
    if acc2_mask.sum() > 0:
        pred_binary = (pred_valid[acc2_mask] >= 0).astype(int)
        label_binary = (label_valid[acc2_mask] >= 0).astype(int)
        acc2 = (pred_binary == label_binary).mean()
        acc2_n = acc2_mask.sum()
    else:
        acc2 = np.nan
        acc2_n = 0
    
    # 2. Acc2_weak (只對 label 在 [-0.4, 0.4] 區間內的做分類)
    weak_mask = (label_valid >= -0.4) & (label_valid <= 0.4)
    if weak_mask.sum() > 0:
        pred_weak = pred_valid[weak_mask]
        label_weak = label_valid[weak_mask]
        pred_binary_weak = (pred_weak >= 0).astype(int)
        label_binary_weak = (label_weak >= 0).astype(int)
        acc2_weak = (pred_binary_weak == label_binary_weak).mean()
        acc2_weak_n = weak_mask.sum()
    else:
        acc2_weak = np.nan
        acc2_weak_n = 0
    
    # 3. Corr (Pearson correlation) - 使用 numpy
    if len(pred_valid) > 1:
        corr = np.corrcoef(pred_valid, label_valid)[0, 1]
    else:
        corr = np.nan
    
    # 4. MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred_valid - label_valid))
    
    return {
        'N_samples': n_samples,
        'Acc2': acc2,
        'Acc2_N': acc2_n,
        'Acc2_weak': acc2_weak,
        'Acc2_weak_N': acc2_weak_n,
        'Corr': corr,
        'MAE': mae
    }

# ============================================================
# 計算各模態對應自己 label 的指標
# ============================================================
print("=" * 70)
print("各模態對應自己 label 的評估結果 (Test Set)")
print("  - Visual: pred_v vs label_V")
print("  - Audio:  pred_a vs label_A")
print("  - Text:   pred_t vs label_T")
print("=" * 70)
print()

# Visual: pred_v vs label_V
visual_metrics = calculate_metrics(test_df['pred_v'], test_df['label_V'])

# Audio: pred_a vs label_A
audio_metrics = calculate_metrics(test_df['pred_a'], test_df['label_A'])

# Text: pred_t vs label_T
text_metrics = calculate_metrics(test_df['pred_t'], test_df['label_T'])

# 整理結果
results = {
    'Visual': visual_metrics,
    'Audio': audio_metrics,
    'Text': text_metrics
}

# 輸出表格
print(f"{'Modality':<10} {'Acc2':<10} {'Acc2_weak':<12} {'Corr':<10} {'MAE':<10}")
print("-" * 52)
for name, metrics in results.items():
    print(f"{name:<10} {metrics['Acc2']:.4f}     {metrics['Acc2_weak']:.4f}       {metrics['Corr']:.4f}     {metrics['MAE']:.4f}")

print()
print("=" * 70)
print("詳細資訊:")
print("=" * 70)
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  - 總樣本數: {metrics['N_samples']}")
    print(f"  - Acc2 (排除 label=0): {metrics['Acc2']:.4f} (N={metrics['Acc2_N']})")
    print(f"  - Acc2_weak (label ∈ [-0.4, 0.4]): {metrics['Acc2_weak']:.4f} (N={metrics['Acc2_weak_N']})")
    print(f"  - Pearson Correlation: {metrics['Corr']:.4f}")
    print(f"  - MAE: {metrics['MAE']:.4f}")