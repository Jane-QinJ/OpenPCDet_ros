import numpy as np
import matplotlib.pyplot as plt

# 示例数据：模型检测结果和真值（单位：米）
# 用你的实际数据替换下面两个列表
predicted_distances = [2.8, 3.1, 2.95, 3.2, 2.7]
ground_truth_distances = [3.0, 3.0, 3.0, 3.0, 3.0]

# 计算偏差
errors = np.array(predicted_distances) - np.array(ground_truth_distances)

# 输出每个样本的偏差
for i, (pred, gt, err) in enumerate(zip(predicted_distances, ground_truth_distances, errors)):
    print(f"Sample {i+1}: Predicted={pred:.2f} m, Ground Truth={gt:.2f} m, Error={err:.2f} m")

# 画出偏差分布
plt.figure(figsize=(8,4))
plt.plot(errors, marker='o', label='Error (Predicted - Ground Truth)')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Distance Error (m)')
plt.title('Distance Detection Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()