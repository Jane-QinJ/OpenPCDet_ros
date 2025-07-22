import collections

input_file = "src/OpenPCDet_ros/result/tamura_result/raw_result_iou0.7/detection_distances_person3_20m.txt"
output_file = "src/OpenPCDet_ros/result/tamura_result/fp_result_iou0.7/detection_distances_person3_20m_labeled.txt"

true_distance = 20.0
tolerance = 0.5  # 容忍范围
exclude_range = None  # 杂质范围

frame_dict = collections.defaultdict(list)
with open(input_file, "r") as fin:
    for line in fin:
        parts = line.strip().split(",")
        if len(parts) < 5:
            continue
        frame = parts[0]
        distance = float(parts[2])
        frame_dict[frame].append((distance, line))

with open(output_file, "w") as fout:
    for frame, items in frame_dict.items():
        # 如果没有杂质区间，filtered 就等于 items
        if exclude_range is not None:
            filtered = [item for item in items if not (exclude_range[0] <= item[0] <= exclude_range[1])]
        else:
            filtered = items
        if not filtered:
            continue
        filtered_sorted = sorted(filtered, key=lambda x: abs(x[0] - true_distance))
        tp_flag = False
        for idx, (distance, line) in enumerate(filtered_sorted):
            if not tp_flag and abs(distance - true_distance) <= tolerance:
                fout.write(line.strip() + ",TP\n")
                tp_flag = True
            elif abs(distance - true_distance) > tolerance:
                fout.write(line.strip() + ",FP\n")