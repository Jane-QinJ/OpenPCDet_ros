import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# 每个人的文件列表（举例，按实际补全）
person_files = {
    "person1": [
        "src/OpenPCDet_ros/result/oka_result/fp_result_iou0.7/detection_distances_person1_3m_labeled.txt",
        "src/OpenPCDet_ros/result/oka_result/fp_result_iou0.7/detection_distances_person1_5m_labeled.txt",
        "src/OpenPCDet_ros/result/oka_result/fp_result_iou0.7/detection_distances_person1_7m_labeled.txt",
        "src/OpenPCDet_ros/result/oka_result/fp_result_iou0.7/detection_distances_person1_10m_labeled.txt",
        "src/OpenPCDet_ros/result/oka_result/fp_result_iou0.7/detection_distances_person1_13m_labeled.txt",
        "src/OpenPCDet_ros/result/oka_result/fp_result_iou0.7/detection_distances_person1_15m_labeled.txt",
        "src/OpenPCDet_ros/result/oka_result/fp_result_iou0.7/detection_distances_person1_17m_labeled.txt",
        "src/OpenPCDet_ros/result/oka_result/fp_result_iou0.7/detection_distances_person1_20m_labeled.txt",
        # 这里可以继续添加
        # ... 继续添加其它距离的文件
    ],
    "person2": [
        "src/OpenPCDet_ros/result/sensei_result/fp_result_iou0.7/detection_distances_person2_3m_labeled.txt",
        "src/OpenPCDet_ros/result/sensei_result/fp_result_iou0.7/detection_distances_person2_5m_labeled.txt",
        "src/OpenPCDet_ros/result/sensei_result/fp_result_iou0.7/detection_distances_person2_7m_labeled.txt",
        "src/OpenPCDet_ros/result/sensei_result/fp_result_iou0.7/detection_distances_person2_10m_labeled.txt",
        "src/OpenPCDet_ros/result/sensei_result/fp_result_iou0.7/detection_distances_person2_13m_labeled.txt",
        "src/OpenPCDet_ros/result/sensei_result/fp_result_iou0.7/detection_distances_person2_15m_labeled.txt",
        "src/OpenPCDet_ros/result/sensei_result/fp_result_iou0.7/detection_distances_person2_17m_labeled.txt",
        "src/OpenPCDet_ros/result/sensei_result/fp_result_iou0.7/detection_distances_person2_20m_labeled.txt",
        #    ... 其它距离
    ],
    "person3": [
        "src/OpenPCDet_ros/result/tamura_result/fp_result_iou0.7/detection_distances_person3_3m_labeled.txt",
        "src/OpenPCDet_ros/result/tamura_result/fp_result_iou0.7/detection_distances_person3_5m_labeled.txt",
        "src/OpenPCDet_ros/result/tamura_result/fp_result_iou0.7/detection_distances_person3_7m_labeled.txt",
        "src/OpenPCDet_ros/result/tamura_result/fp_result_iou0.7/detection_distances_person3_10m_labeled.txt",
        "src/OpenPCDet_ros/result/tamura_result/fp_result_iou0.7/detection_distances_person3_13m_labeled.txt",
        "src/OpenPCDet_ros/result/tamura_result/fp_result_iou0.7/detection_distances_person3_15m_labeled.txt",
        "src/OpenPCDet_ros/result/tamura_result/fp_result_iou0.7/detection_distances_person3_17m_labeled.txt",
        "src/OpenPCDet_ros/result/tamura_result/fp_result_iou0.7/detection_distances_person3_20m_labeled.txt",
        # ... 其它距离
    ]
}

all_results = []

for person, files in person_files.items():
    for file in files:
        basename = os.path.basename(file)
        true_distance = float(basename.split("_")[3].replace("m", ""))

        det_frames_tp = set()
        det_frames_fp = set()
        with open(file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame = int(parts[0])
                label = parts[-1]
                if label == "TP":
                    det_frames_tp.add(frame)
                elif label == "FP":
                    det_frames_fp.add(frame)

        if not det_frames_tp and not det_frames_fp:
            continue

        # 真值帧区间
        all_frames = det_frames_tp | det_frames_fp
        gt_min = min(all_frames)
        gt_max = max(all_frames)
        gt_frames = set(range(gt_min, gt_max + 1))

        tp = len(det_frames_tp)
        fp = len(det_frames_fp)
        fn = len(gt_frames - det_frames_tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        all_results.append({
            "person": person,
            "distance": true_distance,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

# 生成DataFrame和表格
df = pd.DataFrame(all_results)
df = df.sort_values(["person", "distance"])
print(df)
df.to_csv("precision_recall_summary_all_persons.csv", index=False)

# 可视化
plt.figure(figsize=(8,5))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, person in enumerate(df['person'].unique()):
    person_data = df[df['person'] == person]
    plt.plot(person_data["distance"], person_data["Precision"], marker='o', label=f"{person} Precision", color=colors[i])
    plt.plot(person_data["distance"], person_data["Recall"], marker='s', label=f"{person} Recall", linestyle='--', color=colors[i])
    plt.plot(person_data["distance"], person_data["F1"], marker='^', label=f"{person} F1", linestyle=':', color=colors[i])

plt.xlabel("Ground Truth Distance (m)")
plt.ylabel("Score")
plt.title("Precision, Recall, F1 vs. Distance (All Persons)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks([3, 5, 7, 10, 15, 17, 20])
plt.ylim(0, 1)
plt.savefig("precision_recall_f1_vs_distance_all_persons.png")
plt.show()