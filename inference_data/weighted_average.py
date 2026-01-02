import pandas as pd
import numpy as np

# ============================================================
# 讀取資料
# ============================================================
df = pd.read_csv("fusion_Video_videoMAE_with_labels.csv")

# 分割 validation 和 test set
valid_df = df[df["mode"] == "valid"].copy()
test_df = df[df["mode"] == "test"].copy()

print(f"Validation set 樣本數: {len(valid_df)}")
print(f"Test set 樣本數: {len(test_df)}")
print()


# ============================================================
# 評估函數
# ============================================================
def calculate_acc2(pred, label):
    """計算 Acc2 (positive: >=0, negative: <0)"""
    valid_mask = ~(pd.isna(pred) | pd.isna(label))
    pred_valid = (
        np.array(pred)[valid_mask]
        if isinstance(pred, (pd.Series, list))
        else pred[valid_mask]
    )
    label_valid = (
        np.array(label)[valid_mask]
        if isinstance(label, (pd.Series, list))
        else label[valid_mask]
    )

    pred_binary = (pred_valid >= 0).astype(int)
    label_binary = (label_valid >= 0).astype(int)
    return (pred_binary == label_binary).mean()


def calculate_all_metrics(pred, label):
    """計算所有評估指標"""
    valid_mask = ~(pd.isna(pred) | pd.isna(label))
    pred_valid = (
        np.array(pred)[valid_mask]
        if isinstance(pred, (pd.Series, list))
        else pred[valid_mask]
    )
    label_valid = (
        np.array(label)[valid_mask]
        if isinstance(label, (pd.Series, list))
        else label[valid_mask]
    )

    n_samples = len(pred_valid)

    # Acc2
    pred_binary = (pred_valid >= 0).astype(int)
    label_binary = (label_valid >= 0).astype(int)
    acc2 = (pred_binary == label_binary).mean()

    # F1_score
    TP = np.sum((pred_binary == 1) & (label_binary == 1))
    FP = np.sum((pred_binary == 1) & (label_binary == 0))
    FN = np.sum((pred_binary == 0) & (label_binary == 1))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # Acc2_weak
    weak_mask = (label_valid >= -0.4) & (label_valid <= 0.4)
    if weak_mask.sum() > 0:
        pred_binary_weak = (pred_valid[weak_mask] >= 0).astype(int)
        label_binary_weak = (label_valid[weak_mask] >= 0).astype(int)
        acc2_weak = (pred_binary_weak == label_binary_weak).mean()
    else:
        acc2_weak = np.nan

    # Corr
    corr = np.corrcoef(pred_valid, label_valid)[0, 1] if len(pred_valid) > 1 else np.nan

    # R_square
    if len(pred_valid) > 1:
        ss_res = np.sum((label_valid - pred_valid) ** 2)
        ss_tot = np.sum((label_valid - np.mean(label_valid)) ** 2)
        r_square = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    else:
        r_square = np.nan

    # MAE
    mae = np.mean(np.abs(pred_valid - label_valid))

    return {
        "Acc2": acc2,
        "F1_score": f1_score,
        "Acc2_weak": acc2_weak,
        "Corr": corr,
        "R_square": r_square,
        "MAE": mae,
    }


def weighted_average_pred(df, w_v, w_a, w_t):
    """計算加權平均預測值，處理缺失值"""
    pred_v = df["pred_v"].values
    pred_a = df["pred_a"].values
    pred_t = df["pred_t"].values

    # 處理缺失值：只用有效的模態
    result = np.zeros(len(df))
    for i in range(len(df)):
        weights = []
        preds = []
        if not pd.isna(pred_v[i]):
            weights.append(w_v)
            preds.append(pred_v[i])
        if not pd.isna(pred_a[i]):
            weights.append(w_a)
            preds.append(pred_a[i])
        if not pd.isna(pred_t[i]):
            weights.append(w_t)
            preds.append(pred_t[i])

        if len(weights) > 0:
            # 重新正規化權重
            total_w = sum(weights)
            result[i] = sum(w * p for w, p in zip(weights, preds)) / total_w
        else:
            result[i] = np.nan

    return result


# ============================================================
# 方法 1: Grid Search
# ============================================================
print("=" * 85)
print("方法 1: Grid Search (從 Validation Set 找最佳權重)")
print("=" * 85)

best_acc2_grid = 0
best_weights_grid = (1 / 3, 1 / 3, 1 / 3)

for w_v in np.arange(0, 1.01, 0.05):
    for w_a in np.arange(0, 1.01 - w_v, 0.05):
        w_t = round(1 - w_v - w_a, 2)
        if w_t < 0:
            continue

        pred_weighted = weighted_average_pred(valid_df, w_v, w_a, w_t)
        acc2 = calculate_acc2(pred_weighted, valid_df["y_true"].values)

        if acc2 > best_acc2_grid:
            best_acc2_grid = acc2
            best_weights_grid = (w_v, w_a, w_t)

print(
    f"最佳權重 (V, A, T): ({best_weights_grid[0]:.2f}, {best_weights_grid[1]:.2f}, {best_weights_grid[2]:.2f})"
)
print(f"Validation Acc2: {best_acc2_grid:.4f}")

# ============================================================
# 方法 2: 根據單模態 Acc2 表現設定權重
# ============================================================
print()
print("=" * 85)
print("方法 2: 根據單模態 Acc2 表現設定權重 (從 Validation Set)")
print("=" * 85)

# 計算 validation set 上各模態的 Acc2
acc2_v_valid = calculate_acc2(valid_df["pred_v"], valid_df["y_true"])
acc2_a_valid = calculate_acc2(valid_df["pred_a"], valid_df["y_true"])
acc2_t_valid = calculate_acc2(valid_df["pred_t"], valid_df["y_true"])

print(f"Validation 單模態 Acc2:")
print(f"  Visual: {acc2_v_valid:.4f}")
print(f"  Audio:  {acc2_a_valid:.4f}")
print(f"  Text:   {acc2_t_valid:.4f}")

total_acc2 = acc2_v_valid + acc2_a_valid + acc2_t_valid
w_v_acc2 = acc2_v_valid / total_acc2
w_a_acc2 = acc2_a_valid / total_acc2
w_t_acc2 = acc2_t_valid / total_acc2

print(
    f"\n根據 Acc2 計算的權重 (V, A, T): ({w_v_acc2:.4f}, {w_a_acc2:.4f}, {w_t_acc2:.4f})"
)

# 驗證這組權重在 validation set 上的表現
pred_weighted_acc2 = weighted_average_pred(valid_df, w_v_acc2, w_a_acc2, w_t_acc2)
valid_acc2_method2 = calculate_acc2(pred_weighted_acc2, valid_df["y_true"].values)
print(f"Validation Acc2: {valid_acc2_method2:.4f}")

# ============================================================
# 方法 3: Optimization (Scipy)
# ============================================================
print()
print("=" * 85)
print("方法 3: Optimization 優化 (從 Validation Set)")
print("=" * 85)

from scipy.optimize import minimize


def neg_acc2(weights):
    """目標函數：負的 Acc2（因為 minimize 是最小化）"""
    w_v, w_a = weights
    w_t = 1 - w_v - w_a
    if w_t < 0 or w_v < 0 or w_a < 0:
        return 1.0  # 懲罰無效權重

    pred_weighted = weighted_average_pred(valid_df, w_v, w_a, w_t)
    acc2 = calculate_acc2(pred_weighted, valid_df["y_true"].values)
    return -acc2


# 初始值
x0 = [1 / 3, 1 / 3]

# 約束條件：w_v + w_a <= 1, w_v >= 0, w_a >= 0
bounds = [(0, 1), (0, 1)]
constraints = {"type": "ineq", "fun": lambda x: 1 - x[0] - x[1]}

result = minimize(neg_acc2, x0, method="SLSQP", bounds=bounds, constraints=constraints)
w_v_opt, w_a_opt = result.x
w_t_opt = 1 - w_v_opt - w_a_opt

print(f"最佳權重 (V, A, T): ({w_v_opt:.4f}, {w_a_opt:.4f}, {w_t_opt:.4f})")
print(f"Validation Acc2: {-result.fun:.4f}")

# ============================================================
# 在 Test Set 上評估所有方法
# ============================================================
print()
print("=" * 85)
print("在 Test Set 上評估所有方法")
print("=" * 85)
print()

# Simple Average
test_df["pred_simple_avg"] = test_df[["pred_v", "pred_a", "pred_t"]].mean(
    axis=1, skipna=True
)

# Grid Search Weighted
test_df["pred_grid"] = weighted_average_pred(test_df, *best_weights_grid)

# Acc2-based Weighted
test_df["pred_acc2_based"] = weighted_average_pred(
    test_df, w_v_acc2, w_a_acc2, w_t_acc2
)

# Optimization Weighted
test_df["pred_optimized"] = weighted_average_pred(test_df, w_v_opt, w_a_opt, w_t_opt)

# 計算各方法的指標
methods = {
    "Simple Avg": ("pred_simple_avg", (1 / 3, 1 / 3, 1 / 3)),
    "Grid Search": ("pred_grid", best_weights_grid),
    "Acc2-based": ("pred_acc2_based", (w_v_acc2, w_a_acc2, w_t_acc2)),
    "Optimized": ("pred_optimized", (w_v_opt, w_a_opt, w_t_opt)),
}

results = {}
for name, (col, weights) in methods.items():
    metrics = calculate_all_metrics(test_df[col], test_df["y_true"])
    metrics["weights"] = weights
    results[name] = metrics

# 輸出表格
print(
    f"{'Method':<14} {'Weights (V,A,T)':<24} {'Acc2(↑)':<10} {'F1(↑)':<10} {'Acc2_weak(↑)':<14} {'Corr(↑)':<10} {'R²(↑)':<10} {'MAE(↓)':<10}"
)
print("-" * 112)
for name, metrics in results.items():
    w = metrics["weights"]
    print(
        f"{name:<14} ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})          {metrics['Acc2']:.4f}     {metrics['F1_score']:.4f}     {metrics['Acc2_weak']:.4f}         {metrics['Corr']:.4f}     {metrics['R_square']:.4f}     {metrics['MAE']:.4f}"
    )

# 找出最佳方法
best_method = max(results.items(), key=lambda x: x[1]["Acc2"])
print()
print(f"最佳方法: {best_method[0]} (Test Acc2: {best_method[1]['Acc2']:.4f})")
