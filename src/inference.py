#! /usr/bin/env python3.8
"""
Created on Thu Aug  6 11:27:43 2020

@author: Javier del Egido Sierra and Carlos Gómez-Huélamo

===

Modified on 23 Dec 2022
@author: Kin ZHANG (https://kin-zhang.github.io/)

Part of codes also refers: https://github.com/kwea123/ROS_notes
"""

# General use imports
import os
import time
import glob
from pathlib import Path

# ROS imports
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker

# Math and geometry imports
import math
import numpy as np
import torch

# OpenPCDet imports
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti

# Kin's utils
from utils.draw_3d import Draw3DBox
from utils.global_def import *
from utils import *

import yaml
import os
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
with open(f"{BASE_DIR}/launch/config.yaml", 'r') as f:
    try:
        para_cfg = yaml.safe_load(f, Loader=yaml.FullLoader)
    except:
        para_cfg = yaml.safe_load(f)

cfg_root = para_cfg["cfg_root"]
model_path = para_cfg["model_path"]
threshold = para_cfg["threshold"]
pointcloud_topic = para_cfg["pointcloud_topic"]
RATE_VIZ = para_cfg["viz_rate"]
output_file = para_cfg.get("output_file")
inference_time_list = []


def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    """
    Create a sensor_msgs.PointCloud2 from an array of points.
    """
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

def calculate_distance(box):
    """
    Calculate the Euclidean distance from the origin to the object's center.
    Args:
        box: A bounding box array [x, y, z, ...].
    Returns:
        distance: The Euclidean distance.
    """
    x, y, z = box[:3]  # Assuming box contains [x, y, z, ...]
    distance = np.sqrt(x**2 + y**2 + z**2)
    return distance


def publish_distance_text(distance, box, frame_id,rate=10):
    """
    Publish the distance of the detected object as text in RViz.
    Args:
        distance: The distance to the object.
        box: The bounding box of the object [x, y, z, ...].
        frame_id: The frame ID to associate with the marker.
    """
    lifetime = 1.0/rate
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "distance_text"
    #  Ensure the ID is within the range of int32
    marker.id = abs(hash(str(box))) % 2147483647  # Modulo to fit within int32 bounds
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration(lifetime)

    # Set the position of the text slightly above the object
    marker.pose.position.x = box[0]
    marker.pose.position.y = box[1]
    marker.pose.position.z = box[2] + 1.0  # Offset above the object
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Set the text to display the distance
    marker.text = f"{distance:.2f} m"

    # Set the scale and color of the text
    marker.scale.z = 0.5  # Text size
    marker.color.a = 1.0  # Alpha
    marker.color.r = 1.0  # Red
    marker.color.g = 1.0  # Green
    marker.color.b = 1.0  # Blue

    # Publish the marker directly using the ROS publisher
    distance_text_publisher.publish(marker)

def rslidar_callback(msg):
    select_boxs, select_types = [],[]
    if proc_1.no_frame_id:
        proc_1.set_viz_frame_id(msg.header.frame_id)
        print(f"{bc.OKGREEN} setting marker frame id to lidar: {msg.header.frame_id} {bc.ENDC}")
        proc_1.no_frame_id = False

    frame = msg.header.seq # frame id -> not timestamp
    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    # print(f"points data from msg_cloud: {msg_cloud} /t")
    
    np_p = get_xyz_points(msg_cloud, True)

    # Print points data received from the LiDAR
    # print(f"Points data received for frame {frame}:")
    # print(np_p)


    scores, dt_box_lidar, types, pred_dict = proc_1.run(np_p, frame)
    # === 新增：写入txt文件 ===
    # print("Writing detection results to txt...")
    # with open(output_file, "a") as f:
    #     for i, score in enumerate(scores):
    #         print(f"score: {score}")  # 调试用
    #         if score > threshold:
    #             distance = calculate_distance(dt_box_lidar[i])
    #             obj_type = pred_dict['name'][i]
    #             print(f"Writing: {frame},{i},{distance:.3f},{score:.3f},{obj_type}")  # 调试用
    #             f.write(f"{frame},{i},{distance:.3f},{score:.3f},{obj_type}\n")
    for i, score in enumerate(scores):
        if score>threshold:
            select_boxs.append(dt_box_lidar[i])
            select_types.append(pred_dict['name'][i])
            # Calculate the distance of the detected object
            distance = calculate_distance(dt_box_lidar[i])
            
            # Publish the distance as a text marker in RViz
            publish_distance_text(distance, dt_box_lidar[i], msg.header.frame_id)

    if(len(select_boxs)>0):
        # traker id is set into -1
        proc_1.pub_rviz.publish_3dbox(np.array(select_boxs), -1, pred_dict['name'])
        print_str = f"Frame id: {frame}. Prediction results: \n"
        for i in range(len(pred_dict['name'])):
            print_str += f"Type: {pred_dict['name'][i]:.3s} Prob: {scores[i]:.2f}\n"
        print(print_str)
    else:
        print(f"\n{bc.FAIL} No confident prediction in this time stamp {bc.ENDC}\n")
    print(f" -------------------------------------------------------------- ")

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

class Processor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None
        self.pub_rviz = None
        self.no_frame_id = True
        self.rate = RATE_VIZ

    def set_pub_rviz(self, box3d_pub, marker_frame_id = 'velodyne'):
        self.pub_rviz = Draw3DBox(box3d_pub, marker_frame_id, self.rate)
    
    def set_viz_frame_id(self, marker_frame_id):
        self.pub_rviz.set_frame_id(marker_frame_id)

    def initialize(self):
        self.read_config()
        
    def read_config(self):
        config_path = self.config_path
        cfg_from_yaml_file(self.config_path, cfg)
        self.logger = common_utils.create_logger()
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path("/home/kin/workspace/OpenPCDet/tools/000002.bin"),
            ext='.bin')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        print("Model path: ", self.model_path)
        self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device).eval()

    def get_template_prediction(self, num_samples):
        """
        Generates a template dictionary for predictions with pre-allocated numpy arrays.

        Args:
            num_samples (int): The number of samples for which the prediction template is created.

        Returns:
            dict: A dictionary containing the following keys with pre-allocated numpy arrays:
                - 'name': Array of zeros with shape (num_samples,).
                - 'truncated': Array of zeros with shape (num_samples,).
                - 'occluded': Array of zeros with shape (num_samples,).
                - 'alpha': Array of zeros with shape (num_samples,).
                - 'bbox': Array of zeros with shape (num_samples, 4).
                - 'dimensions': Array of zeros with shape (num_samples, 3).
                - 'location': Array of zeros with shape (num_samples, 3).
                - 'rotation_y': Array of zeros with shape (num_samples,).
                - 'score': Array of zeros with shape (num_samples,).
                - 'boxes_lidar': Array of zeros with shape (num_samples, 7).
        """
        ret_dict = {
            'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
            'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
            'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
            'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
            'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
        }
        return ret_dict

    def run(self, points, frame):
        t_t = time.time()
        num_features = 4 # X,Y,Z,intensity       
        self.points = points.reshape([-1, num_features])
        assert self.points.shape[0] > 0, "No points in lidar data, please check your lidar message."
        print(f"Total points: {self.points.shape[0]}")
        timestamps = np.empty((len(self.points),1))
        timestamps[:] = frame

        input_dict = {
            'points': self.points,
            'frame_id': frame,
        }

        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        torch.cuda.synchronize()
        t = time.time()

        pred_dicts, _ = self.net.forward(data_dict)
        
        torch.cuda.synchronize()
        inference_time = time.time() - t
        inference_time_list.append(inference_time)
        mean_inference_time = sum(inference_time_list)/len(inference_time_list)

        boxes_lidar = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
        scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()
        types = pred_dicts[0]["pred_labels"].detach().cpu().numpy()

        pred_boxes = np.copy(boxes_lidar)
        pred_dict = self.get_template_prediction(scores.shape[0])

        pred_dict['name'] = np.array(cfg.CLASS_NAMES)[types - 1]
        pred_dict['score'] = scores
        pred_dict['boxes_lidar'] = pred_boxes

        return scores, boxes_lidar, types, pred_dict
    
 # The script initializes the ROS node,
# subscribes to the LiDAR point cloud topic, and sets up the publisher for visualization.
# The node starts spinning to process incoming messages and perform 3D object detection.
if __name__ == "__main__":
    no_frame_id = False
    proc_1 = Processor_ROS(cfg_root, model_path)
    print(f"\n{bc.OKCYAN}Config path: {bc.BOLD}{cfg_root}{bc.ENDC}")
    print(f"{bc.OKCYAN}Model path: {bc.BOLD}{model_path}{bc.ENDC}")
    # print(f"If it's not correct please change in the config file... \n")

    proc_1.initialize()
    rospy.init_node('object_3d_detector_node')
    sub_lidar_topic = [pointcloud_topic]

    cfg_from_yaml_file(cfg_root, cfg)
    
    sub_ = rospy.Subscriber(sub_lidar_topic[0], PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)
    pub_rviz = rospy.Publisher('detect_3dbox',MarkerArray, queue_size=10)
    distance_text_publisher = rospy.Publisher('distance_text_marker', Marker, queue_size=10)
    proc_1.set_pub_rviz(pub_rviz)
    print(f"{bc.HEADER} ====================== {bc.ENDC}")
    print(" ===> [+] PCDet ros_node has started. Try to Run the rosbag file")
    print(f"{bc.HEADER} ====================== {bc.ENDC}")

    rospy.spin()
