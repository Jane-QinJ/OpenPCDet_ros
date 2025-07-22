# 20m oka
# input_file = "src/OpenPCDet_ros/result/detection_distances_oka_20m.txt"
# output_file = "src/OpenPCDet_ros/result/detection_distances_20m_only.txt"

# with open(input_file, "r") as fin, open(output_file, "w") as fout:
#     for line in fin:
#         parts = line.strip().split(",")
#         if len(parts) < 5:
#             continue
#         distance = float(parts[2])
#         if 18.0 <= distance <= 22.0:
#             fout.write(line)

# 15m oka
# input_file = "src/OpenPCDet_ros/result/detection_distances_oka_15m.txt"
# output_file = "src/OpenPCDet_ros/result/detection_distances_15m_only.txt"

# with open(input_file, "r") as fin, open(output_file, "w") as fout:
#     for line in fin:
#         parts = line.strip().split(",")
#         if len(parts) < 5:
#             continue
#         distance = float(parts[2])
#         if 14.0 <= distance <= 16.0:
#             fout.write(line)
            
            
# 17m oka
# input_file = "src/OpenPCDet_ros/result/detection_distances_oka_17m.txt"
# output_file = "src/OpenPCDet_ros/result/detection_distances_17m_only.txt"

# with open(input_file, "r") as fin, open(output_file, "w") as fout:
#     for line in fin:
#         parts = line.strip().split(",")
#         if len(parts) < 5:
#             continue
#         distance = float(parts[2])
#         if 16.0 <= distance <= 18.0:
#             fout.write(line)

# 3m oka
# input_file = "src/OpenPCDet_ros/result/sensei_result/raw_result_iou0.7/detection_distances_person2_3m.txt"
# output_file = "src/OpenPCDet_ros/result/sensei_result/final_result_iou0.7/detection_distances_person2_3m_only.txt"

# with open(input_file, "r") as fin, open(output_file, "w") as fout:
#     for line in fin:
#         parts = line.strip().split(",")
#         if len(parts) < 5:
#             continue
#         distance = float(parts[2])
#         if 2.5 <= distance <= 3.5:
#             fout.write(line)

# 5m oka
# input_file = "src/OpenPCDet_ros/result/detection_distances_oka_5m.txt"
# output_file = "src/OpenPCDet_ros/result/detection_distances_5m_only.txt"

# with open(input_file, "r") as fin, open(output_file, "w") as fout:
#     for line in fin:
#         parts = line.strip().split(",")
#         if len(parts) < 5:
#             continue
#         distance = float(parts[2])
#         if 4.5 <= distance <= 5.5:
#             fout.write(line)


# input_file = "src/OpenPCDet_ros/result/sensei_result/raw_result_iou0.7/detection_distances_person2_5m.txt"
# output_file = "src/OpenPCDet_ros/result/sensei_result/final_result_iou0.7/detection_distances_person2_5m_only.txt"

# import collections

# # 读取所有数据并按帧分组
# frame_dict = collections.defaultdict(list)
# with open(input_file, "r") as fin:
#     for line in fin:
#         parts = line.strip().split(",")
#         if len(parts) < 5:
#             continue
#         frame = parts[0]
#         distance = float(parts[2])
#         frame_dict[frame].append((abs(distance - 5.0), line))

# # 每帧只保留距离最接近5m的那一行
# with open(output_file, "w") as fout:
#     for frame, items in frame_dict.items():
#         # 取距离5m最近的
#         items.sort()
#         fout.write(items[0][1])

import collections

input_file = "src/OpenPCDet_ros/result/tamura_result/raw_result_iou0.7/detection_distances_person3_20m.txt"
output_file = "src/OpenPCDet_ros/result/tamura_result/final_result_iou0.7/detection_distances_person3_20m_only.txt"

frame_dict = collections.defaultdict(list)
with open(input_file, "r") as fin:
    for line in fin:
        parts = line.strip().split(",")
        if len(parts) < 5:
            continue
        frame = parts[0]
        distance = float(parts[2])
        # 只保留在14~16m范围内的数据
        if 19.0 <= distance <= 21.0:
            frame_dict[frame].append((abs(distance - 20.0), line))

with open(output_file, "w") as fout:
    for frame, items in frame_dict.items():
        if items:  # 该帧有满足条件的数据
            items.sort()
            fout.write(items[0][1])