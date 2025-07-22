import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("当前工作目录：", os.getcwd())
os.chdir("/home/firo/Documents/workspace/OpenPCDet_ws/src/OpenPCDet_ros/result/")
input_file = "detection_distances_3m_only.txt"
if not os.path.exists(input_file):
    raise FileNotFoundError(f"未找到 {input_file} 文件，当前工作目录：{os.getcwd()}")

# 读取数据文件
data = pd.read_csv(
    input_file,
    names=['frame', 'id', 'distance', 'score', 'class'],
    dtype={'frame': int, 'id': int, 'distance': float, 'score': float, 'class': str}
)

# 设置真值
true_distance = 3.0
data['abs_error'] = np.abs(data['distance'] - true_distance)

# 评估指标
mae = data['abs_error'].mean()
rmse = np.sqrt((data['abs_error'] ** 2).mean())
max_error = data['abs_error'].max()

print(f"参考真值: {true_distance} m")
print(f"平均绝对误差 MAE: {mae:.3f} m")
print(f"均方根误差 RMSE: {rmse:.3f} m")
print(f"最大误差 Max Error: {max_error:.3f} m")

# # 可视化
# plt.figure(figsize=(10, 4))
# plt.plot(data['frame'], data['distance'], label='Pred. Distance')
# plt.axhline(true_distance, color='green', linestyle='--', label='True Distance = 3.0 m')
# plt.xlabel('frame')
# plt.ylabel('distance(m)')
# plt.title('Pred Distance vs Frame')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("predicted_distance_vs_frame_fixed_truth.png")

# plt.figure(figsize=(10, 4))
# plt.plot(data['frame'], data['abs_error'], label='error', color='red')
# plt.xlabel('frame')
# plt.ylabel('error (m)')
# plt.title('distance error vs frame（truth = 3.0 m）')
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
# plt.savefig("distance_error_3m_analyse.png")

# window_size = 10  # 滑动窗口大小，可根据需要调整

# 平滑处理
# data['distance_smooth'] = data['distance'].rolling(window=window_size, center=True).mean()
# data['abs_error_smooth'] = data['abs_error'].rolling(window=window_size, center=True).mean()

# frame_index = data['frame'] - data['frame'].min() + 1

# 可视化
# plt.figure(figsize=(10, 4))
# plt.plot(frame_index, data['distance_smooth'], label='Smoothed Pred. Distance', color='blue')
# plt.axhline(true_distance, color='green', linestyle='--', label='True Distance = 3.0 m')
# plt.xlabel('Frame')
# plt.ylabel('Distance (m)')
# plt.title('Smoothed Predicted Distance vs Frame')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("predicted_distance_vs_frame_smoothed.png")

# plt.figure(figsize=(10, 4))
# plt.plot(frame_index, data['abs_error_smooth'], label='Smoothed Error', color='red')
# plt.xlabel('Frame')
# plt.ylabel('Error (m)')
# plt.title('Smoothed Distance Error vs Frame (Truth = 3.0 m)')
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
# plt.ylim(0, 1)  # 手动设置纵坐标范围，更加合理
# plt.savefig("distance_error_3m_analyse_smoothed.png")

plt.figure(figsize=(6, 6))
plt.scatter([true_distance]*len(data), data['distance'], color='b', label='Prediction')
plt.plot([true_distance-0.1, true_distance+0.1], [true_distance-0.1, true_distance+0.1], 'r--', label='Ideal: y=x')
plt.xlabel('Ground Truth Distance (m)')
plt.ylabel('Predicted Distance (m)')
plt.title('Prediction vs Ground Truth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_vs_truth_scatter.png")