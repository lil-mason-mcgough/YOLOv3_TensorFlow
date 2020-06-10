import os
import argparse

from yolov3_wizyoung.convert_weight import convert_weight
from yolov3_wizyoung.utils.config_utils import YoloArgs


parser = argparse.ArgumentParser(description='YOLOV3 hyperparameter optimization')
parser.add_argument("--config_file", type=str, default="./config.yaml",
                    help="The path to the model's yaml config file.")
parser.add_argument("--weights_dir", type=str, default="../app_model",
                    help="The directory where Darknet weights are stored.")
parsed_args = parser.parse_args()

yolo_args = YoloArgs(parsed_args.config_file)

convert_weight(
    os.path.join(parsed_args.weights_dir, 'yolov3.weights'), 
    os.path.join(parsed_args.weights_dir, 'yolov3.ckpt'),
    [416, 416],
    yolo_args.anchors)