import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 定义每个人和每个距离的文件路径（举例，需补全/修改为你的实际文件名）
persons = ["person1", "person2", "person3"]
distances = [3, 5, 7, 10, 13, 15, 17, 20]
base_dir = "src/OpenPCDet_ros/result/all_result_0.7"

files = []
for person in persons:
    for d in distances:
        files.append(f"{base_dir}/detection_distances_{person}_{d}m_only.txt")

# 也可以加入oka等其他来源
# files.append("/home/firo/Documents/workspace/OpenPCDet_ws/src/OpenPCDet_ros/result/oka_result/final_result_iou0.7/detection_distances_3m_only.txt")
# ...

all_data = []

for file in files:
    if not os.path.exists(file):
        print(f"Warning: {file} not found, skip.")
        continue
    filename = os.path.basename(file)
    # 提取person和true_distance
    parts = filename.split("_")
    if "person" in filename:
        person = parts[2]
        true_distance = float(parts[3].replace("m", ""))
    else:
        person = "oka"
        true_distance = float(parts[2].replace("m", ""))
    df = pd.read_csv(file, header=None, names=['frame', 'id', 'distance', 'score', 'class'])
    df['true_distance'] = true_distance
    df['person'] = person
    df['abs_error'] = np.abs(df['distance'] - true_distance)
    all_data.append(df)

# 合并所有数据
full_df = pd.concat(all_data, ignore_index=True)

# 分组统计
summary = full_df.groupby(['person', 'true_distance']).agg(
    sample_count=('frame', 'count'),
    mean_pred=('distance', 'mean'),
    mean_error=('abs_error', 'mean'),
    rmse=('abs_error', lambda x: np.sqrt(np.mean(x**2))),
    max_error=('abs_error', 'max')
).reset_index()

print(summary)
summary.to_csv("distance_error_summary_all_persons.csv", index=False)

# 画图：每个人一条线
plt.figure(figsize=(10,6))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, person in enumerate(summary['person'].unique()):
    person_data = summary[summary['person'] == person]
    plt.plot(person_data['true_distance'], person_data['mean_error'], marker='o', label=f'{person} Mean Error', color=colors[i % len(colors)])
    plt.plot(person_data['true_distance'], person_data['rmse'], marker='s', label=f'{person} RMSE', linestyle='--', color=colors[i % len(colors)])

plt.xlabel("Ground Truth Distance (m)")
plt.ylabel("Error (m)")
plt.title("Model Error vs. Ground Truth Distance (All Persons)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.ylim(0, 1)
plt.xticks(distances)
plt.savefig("error_vs_true_distance_all_persons.png")
