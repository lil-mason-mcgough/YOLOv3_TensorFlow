import os
import subprocess

from yolov3_wizyoung.train import train
from yolov3_wizyoung.labels.convert_labels import convert_kitti_data_to_yolo
from yolov3_wizyoung.utils.config_utils import YoloArgs

def make_dir_exist(dirname):
    try:
        os.makedirs(dirname)
    except (OSError, FileExistsError):
        pass

def gsutil_rsync(src_dir, dst_dir):
    process = subprocess.run(['gsutil', '-m', 'rsync', '-r', src_dir, dst_dir], 
                         stdout=subprocess.PIPE, 
                         universal_newlines=True)
    print(process.stdout)
    return process


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="YOLO-V3 train procedure.")
    parser.add_argument("--job-dir", type=str, default="",
                    help="The directory of cloud resources for this job.")
    parser.add_argument("--config_file", type=str, default="./config.yaml",
                    help="The path of the yaml configuration file.")
    parser.add_argument("--bucket_name", type=str, default="lil-ml",
                    help="The bucket name of cloud resources.")
    parser.add_argument("--data_prefix", type=str, default="product_detection/data/kitti_dewalt_escondido",
                    help="The path to bucket training data dir relative to the bucket root.")
    parser.add_argument("--data_dl_dir", type=str, default="./app_data",
                    help="The local directory to save bucket training data.")
    parser.add_argument("--model_prefix", type=str, default="product_detection/models/darknet_weights",
                    help="The path to bucket model dir relative to the bucket root.")
    parser.add_argument("--model_dl_dir", type=str, default="./app_model",
                    help="The the local directory to save model data.")
    parsed_args = parser.parse_args()

    output_model_dir = 'output_model'
    make_dir_exist(output_model_dir)

    # download data and pretrained model from GCS
    make_dir_exist(parsed_args.data_dl_dir)
    gsutil_rsync(
        os.path.join('gs://' + parsed_args.bucket_name, parsed_args.data_prefix), 
        parsed_args.data_dl_dir)
    make_dir_exist(parsed_args.model_dl_dir)
    gsutil_rsync(
        os.path.join('gs://' + parsed_args.bucket_name, parsed_args.model_prefix), 
        parsed_args.model_dl_dir)

    # generate new training paths from data
    data_subsets = {'training': 'train', 'validation': 'val', 'testing_real': 'test'}
    convert_kitti_data_to_yolo(
        parsed_args.data_dl_dir, 
        parsed_args.data_dl_dir, 
        data_subsets=data_subsets)
    
    # load configs
    yolo_args = YoloArgs(parsed_args.config_file)

    # adapt some yolo_args attributes to cloud
    make_dir_exist(yolo_args.save_dir)
    make_dir_exist(yolo_args.log_dir)

    # train model
    train(yolo_args)
    if parsed_args.job_dir == '':
        parsed_args.job_dir = 'gs://{}'.format(parsed_args.bucket_name)
    gsutil_rsync(output_model_dir, parsed_args.job_dir)
