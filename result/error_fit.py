import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for containers/servers without X
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline

csv_file = "/home/firo/Documents/workspace/OpenPCDet_ws/distance_error_summary_all_persons.csv"
out_dir = "/home/firo/Documents/workspace/OpenPCDet_ws/src/OpenPCDet_ros/result/fit_plots"
os.makedirs(out_dir, exist_ok=True)

# 使用 mean_pred 作为纵轴
Y_COL = 'mean_pred'
Y_SCALE = 100.0  # m -> cm
Y_UNIT = "cm"

df = pd.read_csv(csv_file)

# ===== 论文风格 + 字体变大 =====
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2
})

# 统一所有图的 distance 刻度（单位 m）
XTICKS = sorted(df['true_distance'].unique().tolist())
DIST_MAX = max(XTICKS)
DIST_MIN = 0.0  # 坐标轴从 0 开始

def fit_and_plot(sub_df, title_tag):
    x = sub_df['true_distance'].values.astype(float)
    y = sub_df[Y_COL].values.astype(float) * Y_SCALE

    coeffs = np.polyfit(x, y, 1)
    a, b = np.round(coeffs, 4)
    y_pred = a * x + b

    sign = '+' if b >= 0 else '-'
    b_abs = abs(b)

    residuals = np.round(y - y_pred, 6)
    sse = np.round(np.sum(residuals**2), 6)
    y_mean = np.mean(y)
    sst = np.round(np.sum((y - y_mean)**2), 6)
    r2 = np.round(1 - sse / sst, 6) if sst > 0 else np.nan
    n = len(x)
    dof = max(n - 2, 1)
    residual_variance = np.round(sse / dof, 6)

    summary = {
        'tag': title_tag,
        'n_points': n,
        'a(slope)': a,
        'b(intercept)': b,
        'SSE': sse,
        'ResidualVariance': residual_variance,
        'R2': r2
    }

    order = np.argsort(x)
    x_sorted = x[order]
    y_pred_sorted = y_pred[order]

    # ==== 拟合图：True Distance vs Mean Pred ====
    plt.figure(figsize=(6, 5))
    ax = plt.gca()

    # 散点
    ax.scatter(x, y, label='Mean Pred', zorder=3)

    # 拟合线
    ax.plot(x_sorted, y_pred_sorted,
            color='red',
            label=f'Fit: y={a:.4f}x{sign}{b_abs:.4f}',
            zorder=2)

    # 理想 y=x 线（从 0 开始）
    ax.plot([DIST_MIN, DIST_MAX], [DIST_MIN * Y_SCALE, DIST_MAX * Y_SCALE],
            color='green', linestyle='--', label='Ideal y=x', zorder=1)

    # 轴标签 & 标题
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel(f"Mean Predicted Distance ({Y_UNIT})")
    ax.set_title(f"{title_tag} Mean Pred Fit (R²={r2:.6f})")

    # 统一刻度 & 从 0 开始 & 1:1 比例
    ax.set_xlim(DIST_MIN, DIST_MAX)
    ax.set_ylim(DIST_MIN * Y_SCALE, DIST_MAX * Y_SCALE)
    ax.set_xticks(XTICKS)
    ax.set_yticks([t * Y_SCALE for t in XTICKS])

    # 删除背景网格线
    # ax.grid(False)

    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title_tag}_{Y_COL}_fit.png"), dpi=300)
    plt.close()

    # ==== 残差图：误差 vs distance ====
    plt.figure(figsize=(6, 3.5))
    ax = plt.gca()

    # 仅保留 y=0 参考线
    ax.axhline(0, color='black', linewidth=1)

    ax.scatter(x, residuals, color='tab:orange')

    # 如果也想去掉这些竖线，可以把这个 for 循环注释掉
    for xi, ri in zip(x, residuals):
        ax.plot([xi, xi], [0, ri], color='lightgray', linewidth=0.6)

    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel(f"Residual (mean_pred - fit) ({Y_UNIT})")
    ax.set_title(f"{title_tag} Residuals (Var={residual_variance:.5f})")

    # x 轴从 0 开始
    ax.set_xlim(DIST_MIN, DIST_MAX)
    ax.set_xticks(XTICKS)

    # 删除背景网格线
    # ax.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title_tag}_{Y_COL}_residuals.png"), dpi=300)
    plt.close()

    return summary

# 绘制平滑的误差分布图
def analyze_error_distribution(df, out_dir):
    error_stats = []
    distances = sorted(df['true_distance'].unique())

    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    for i, distance in enumerate(distances):
        sub_df = df[df['true_distance'] == distance]
        errors = (sub_df['mean_pred'] - distance) * Y_SCALE
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        error_stats.append({
            'true_distance': distance,
            'mean_error': mean_error,
            'std_error': std_error
        })

        # 误差散点
        ax.scatter([distance] * len(errors), errors,
                   alpha=0.6,
                   label=f"{distance} m" if i == 0 else "")

    mean_errors = [stat['mean_error'] for stat in error_stats]
    std_errors = [stat['std_error'] for stat in error_stats]

    distances_np = np.array(distances)
    mean_errors_np = np.array(mean_errors)
    std_errors_np = np.array(std_errors)

    smooth_distances = np.linspace(distances_np.min(), distances_np.max(), 200)
    smooth_mean_errors = make_interp_spline(distances_np, mean_errors_np)(smooth_distances)
    smooth_std_errors = make_interp_spline(distances_np, std_errors_np)(smooth_distances)

    # 平滑均值误差线
    ax.plot(smooth_distances, smooth_mean_errors,
            color='red', linestyle='--', label='Mean Error (Smoothed)')

    # 误差区间带
    ax.fill_between(smooth_distances,
                    smooth_mean_errors - smooth_std_errors,
                    smooth_mean_errors + smooth_std_errors,
                    alpha=0.2, label='Std Error Range (Smoothed)')

    ax.axhline(0, color='black', linewidth=1, linestyle='--', label='Zero Error')

    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel(f"Error ({Y_UNIT})")
    ax.set_title("Smoothed Error Distribution by Distance")

    # x 轴从 0 开始
    ax.set_xlim(DIST_MIN, DIST_MAX)
    ax.set_xticks(XTICKS)

    # ax.grid(False)

    ax.legend()
    plt.tight_layout()
    error_plot_path = os.path.join(out_dir, "error_distribution_smoothed.png")
    plt.savefig(error_plot_path, dpi=300)
    plt.close()

    # 保存误差统计数据
    error_stats_df = pd.DataFrame(error_stats)
    error_stats_path = os.path.join(out_dir, "error_stats.csv")
    error_stats_df.to_csv(error_stats_path, index=False)
    print(f"Smoothed error distribution plot saved to: {error_plot_path}")
    print(f"Error statistics saved to: {error_stats_path}")

# === 主流程 ===
sample_counts = df.groupby('true_distance')['sample_count'].sum()
print(sample_counts)

all_summaries = []
for person in df['person'].unique():
    sub = df[df['person'] == person].copy()
    all_summaries.append(fit_and_plot(sub, person))

all_summaries.append(fit_and_plot(df, "all_persons"))

summary_df = pd.DataFrame(all_summaries)
summary_path = os.path.join(out_dir, f"linear_fit_summary_{Y_COL}.csv")
summary_df.to_csv(summary_path, index=False)
print(summary_df)
print(f"Saved plots & summary to: {out_dir}")

# 调用误差分布分析函数
analyze_error_distribution(df, out_dir)

# 绘制误差标准差图（距离轴从 0 开始）
error_stats = pd.read_csv(os.path.join(out_dir, "error_stats.csv"))
plt.figure(figsize=(8, 5))
ax = plt.gca()

ax.plot(error_stats['true_distance'].values,
        error_stats['std_error'].values,
        marker='o', label='Std Error')

ax.set_xlabel("True Distance (m)")
ax.set_ylabel(f"Standard Deviation of Error ({Y_UNIT})")
ax.set_title("Error Standard Deviation by Distance")

ax.set_xlim(DIST_MIN, DIST_MAX)
ax.set_xticks(XTICKS)
ax.set_ylim(bottom=0)  # std ≥ 0，从 0 开始

# ax.grid(False)

ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "error_std_by_distance.png"), dpi=300)
plt.close()
