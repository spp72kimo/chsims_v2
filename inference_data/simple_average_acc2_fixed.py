import pandas as pd
import numpy as np

# ============================================================
# 讀取資料
# ============================================================
df = pd.read_csv("fusion_Video_videoMAE_with_labels.csv")

# 只取 test set
test_df = df[df["mode"] == "test"].copy()
print(f"Test set 樣本數: {len(test_df)}")
print()

# ============================================================
# 計算 Simple Average (skipna=True: 只平均有效的模態)
# ============================================================
test_df["pred_avg"] = test_df[["pred_v", "pred_a", "pred_t"]].mean(axis=1, skipna=True)


# ============================================================
# 評估函數
# ============================================================
def calculate_metrics(pred, label):
    """
    計算評估指標: Acc2, F1_score, Acc2_weak, Corr, R_square, MAE

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

    # 1. Acc2 (包含所有樣本，label>=0 為 positive)
    # positive: pred >= 0, negative: pred < 0
    # positive: label >= 0, negative: label < 0
    pred_binary = (pred_valid >= 0).astype(int)
    label_binary = (label_valid >= 0).astype(int)
    acc2 = (pred_binary == label_binary).mean()

    # 2. F1_score
    # True Positive, False Positive, False Negative
    TP = np.sum((pred_binary == 1) & (label_binary == 1))
    FP = np.sum((pred_binary == 1) & (label_binary == 0))
    FN = np.sum((pred_binary == 0) & (label_binary == 1))

    # Precision, Recall, F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # 3. Acc2_weak (只對 label 在 [-0.4, 0.4] 區間內的做分類)
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

    # 4. Corr (Pearson correlation) - 使用 numpy
    if len(pred_valid) > 1:
        corr = np.corrcoef(pred_valid, label_valid)[0, 1]
    else:
        corr = np.nan

    # 5. R_square (R²)
    if len(pred_valid) > 1:
        ss_res = np.sum((label_valid - pred_valid) ** 2)  # 殘差平方和
        ss_tot = np.sum((label_valid - np.mean(label_valid)) ** 2)  # 總平方和
        r_square = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    else:
        r_square = np.nan

    # 6. MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred_valid - label_valid))

    return {
        "N_samples": n_samples,
        "Acc2": acc2,
        "F1_score": f1_score,
        "Acc2_weak": acc2_weak,
        "Acc2_weak_N": acc2_weak_n,
        "Corr": corr,
        "R_square": r_square,
        "MAE": mae,
    }


# ============================================================
# 計算 Simple Average 及各模態對 y_true 的指標
# ============================================================
print("=" * 85)
print("Simple Average Fusion 評估結果 (Test Set, pred vs y_true)")
print("  - Simple Average: mean(pred_v, pred_a, pred_t, skipna=True) vs y_true")
print("  - Visual: pred_v vs y_true")
print("  - Audio:  pred_a vs y_true")
print("  - Text:   pred_t vs y_true")
print("=" * 85)
print()
print("分類規則: positive (pred >= 0), negative (pred < 0)")
print()

# Simple Average: pred_avg vs y_true
avg_metrics = calculate_metrics(test_df["pred_avg"], test_df["y_true"])

# Visual: pred_v vs y_true
visual_metrics = calculate_metrics(test_df["pred_v"], test_df["y_true"])

# Audio: pred_a vs y_true
audio_metrics = calculate_metrics(test_df["pred_a"], test_df["y_true"])

# Text: pred_t vs y_true
text_metrics = calculate_metrics(test_df["pred_t"], test_df["y_true"])

# 整理結果
results = {
    "Simple Avg": avg_metrics,
    "Visual": visual_metrics,
    "Audio": audio_metrics,
    "Text": text_metrics,
}

# 輸出表格
print(
    f"{'Method':<12} {'Acc2(↑)':<10} {'F1(↑)':<10} {'Acc2_weak(↑)':<14} {'Corr(↑)':<10} {'R²(↑)':<10} {'MAE(↓)':<10}"
)
print("-" * 76)
for name, metrics in results.items():
    print(
        f"{name:<12} {metrics['Acc2']:.4f}     {metrics['F1_score']:.4f}     {metrics['Acc2_weak']:.4f}         {metrics['Corr']:.4f}     {metrics['R_square']:.4f}     {metrics['MAE']:.4f}"
    )

print()
print("=" * 85)
print("Simple Average 詳細資訊:")
print("=" * 85)
print(f"  - 總樣本數: {avg_metrics['N_samples']}")
print(f"  - Acc2: {avg_metrics['Acc2']:.4f}")
print(f"  - F1_score: {avg_metrics['F1_score']:.4f}")
print(
    f"  - Acc2_weak (y_true ∈ [-0.4, 0.4]): {avg_metrics['Acc2_weak']:.4f} (N={avg_metrics['Acc2_weak_N']})"
)
print(f"  - Pearson Correlation: {avg_metrics['Corr']:.4f}")
print(f"  - R²: {avg_metrics['R_square']:.4f}")
print(f"  - MAE: {avg_metrics['MAE']:.4f}")
