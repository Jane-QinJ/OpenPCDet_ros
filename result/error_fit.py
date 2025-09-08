import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline

csv_file = "/home/firo/Documents/workspace/OpenPCDet_ws/distance_error_summary_all_persons.csv"
out_dir = "/home/firo/Documents/workspace/OpenPCDet_ws/src/OpenPCDet_ros/result/fit_plots"
os.makedirs(out_dir, exist_ok=True)

# 使用 mean_pred 作为纵轴
Y_COL = 'mean_pred'   # 改这里

df = pd.read_csv(csv_file)

# 新增：统一横轴刻度列表（全部 true_distance 唯一值）
XTICKS = sorted(df['true_distance'].unique().tolist())
# 如果仍有选择性过滤，可在过滤后重新赋值：XTICKS = sorted(filtered_df['true_distance'].unique())

def fit_and_plot(sub_df, title_tag):
    x = sub_df['true_distance'].values.astype(float)
    y = sub_df[Y_COL].values.astype(float)

    # 在拟合和计算时，限制数值为 6 位小数
    coeffs = np.polyfit(x, y, 1)
    a, b = np.round(coeffs, 4)  # 限制斜率和截距为 4 位小数
    y_pred = a * x + b

    # 用符号+绝对值来显示 b
    sign = '+' if b >= 0 else '-'
    b_abs = abs(b)

    residuals = np.round(y - y_pred, 6)  # 限制残差为 6 位小数
    sse = np.round(np.sum(residuals**2), 6)  # SSE 限制为 6 位小数
    y_mean = np.mean(y)
    sst = np.round(np.sum((y - y_mean)**2), 6)  # SST 限制为 6 位小数
    r2 = np.round(1 - sse / sst, 6) if sst > 0 else np.nan  # R² 限制为 6 位小数
    n = len(x)
    dof = max(n - 2, 1)
    residual_variance = np.round(sse / dof, 6)  # 残差方差限制为 6 位小数

    # 保存到 summary 时，限制为 6 位小数
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

    # 绘图时，限制显示为 6 位小数
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, color='tab:blue', label='Mean Pred')
    plt.plot(x_sorted, y_pred_sorted, color='red',
             label=f'Fit: y={a:.4f}x{sign}{b_abs:.4f}')
    plt.plot(x_sorted, x_sorted, color='green', linestyle='--', label='Ideal y=x')
    for xi, yi, ypi in zip(x, y, y_pred):
        plt.plot([xi, xi], [ypi, yi], color='gray', linewidth=0.7)
    plt.xlabel("True Distance (m)")
    plt.ylabel("Mean Predicted Distance (m)")
    plt.title(f"{title_tag} Mean Pred Fit (R²={r2:.6f})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(XTICKS)   # 强制使用统一刻度
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title_tag}_{Y_COL}_fit.png"), dpi=150)
    plt.close()

    # 残差图（残差 = mean_pred - 拟合值，不是与真值差）
    plt.figure(figsize=(6,3.2))
    plt.axhline(0, color='black', linewidth=1)
    plt.scatter(x, residuals, color='tab:orange')
    for xi, ri in zip(x, residuals):
        plt.plot([xi, xi], [0, ri], color='lightgray', linewidth=0.6)
    plt.xlabel("True Distance (m)")
    plt.ylabel("Residual (mean_pred - fit)")
    plt.title(f"{title_tag} Residuals (Var={residual_variance:.5f})")
    plt.grid(True, alpha=0.3)
    plt.xticks(XTICKS)   # 同样使用统一刻度
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title_tag}_{Y_COL}_residuals.png"), dpi=150)
    plt.close()

    return summary

# 绘制平滑的误差分布图
def analyze_error_distribution(df, out_dir):
    error_stats = []
    distances = sorted(df['true_distance'].unique())
    
    plt.figure(figsize=(8, 5))
    for distance in distances:
        sub_df = df[df['true_distance'] == distance]
        errors = sub_df['mean_pred'] - distance  # 误差 = 预测值 - 真实值
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        error_stats.append({
            'true_distance': distance,
            'mean_error': mean_error,
            'std_error': std_error
        })
        
        # 绘制误差点
        plt.scatter([distance] * len(errors), errors, alpha=0.6, label=f"{distance}m" if distance == distances[0] else "")
    
    # 提取误差均值和标准差
    mean_errors = [stat['mean_error'] for stat in error_stats]
    std_errors = [stat['std_error'] for stat in error_stats]
    
    # 使用样条插值平滑误差均值曲线
    distances_np = np.array(distances)
    mean_errors_np = np.array(mean_errors)
    std_errors_np = np.array(std_errors)
    
    # 创建样条插值函数
    smooth_distances = np.linspace(distances_np.min(), distances_np.max(), 200)  # 生成更多点
    smooth_mean_errors = make_interp_spline(distances_np, mean_errors_np)(smooth_distances)
    smooth_std_errors = make_interp_spline(distances_np, std_errors_np)(smooth_distances)
    
    # 绘制平滑的误差均值曲线
    plt.plot(smooth_distances, smooth_mean_errors, color='red', label='Mean Error (Smoothed)', linestyle='--')
    
    # 绘制平滑的误差标准差曲线（可选）
    plt.fill_between(smooth_distances, smooth_mean_errors - smooth_std_errors, smooth_mean_errors + smooth_std_errors,
                     color='blue', alpha=0.2, label='Std Error Range (Smoothed)')
    
    plt.axhline(0, color='black', linewidth=1, linestyle='--', label='Zero Error')
    plt.xlabel("True Distance (m)")
    plt.ylabel("Error (mean_pred - true_distance)")
    plt.title("Smoothed Error Distribution by Distance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    error_plot_path = os.path.join(out_dir, "error_distribution_smoothed.png")
    plt.savefig(error_plot_path, dpi=150)
    plt.close()
    
    # 保存误差统计数据
    error_stats_df = pd.DataFrame(error_stats)
    error_stats_path = os.path.join(out_dir, "error_stats.csv")
    error_stats_df.to_csv(error_stats_path, index=False)
    print(f"Smoothed error distribution plot saved to: {error_plot_path}")
    print(f"Error statistics saved to: {error_stats_path}")

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

# 绘制误差标准差图
error_stats = pd.read_csv("fit_plots/error_stats.csv")
plt.figure(figsize=(8, 5))
plt.plot(error_stats['true_distance'].values, error_stats['std_error'].values, marker='o', label='Std Error')
plt.xlabel("True Distance (m)")
plt.ylabel("Standard Deviation of Error")
plt.title("Error Standard Deviation by Distance")
plt.grid(alpha=0.3)
plt.legend()
plt.show()