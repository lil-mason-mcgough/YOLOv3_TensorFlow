import os
import subprocess

# from google.cloud import storage

from yolov3_wizyoung.train import train
from yolov3_wizyoung.convert_labels import convert_kitti_data_to_yolo
from yolov3_wizyoung.utils.config_utils import YoloArgs

def make_dir_exist(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        pass

def gsutil_rsync(src_dir, dst_dir):
    process = subprocess.run(['gsutil', '-m', 'rsync', '-r', src_dir, dst_dir], 
                         stdout=subprocess.PIPE, 
                         universal_newlines=True)
    print(process.stdout)
    return process

# def download_folder_from_gcs(bucket_name, prefix, output_dir):
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
#     for blob in blobs:
#         file_path = os.path.join(output_dir, os.path.relpath(blob.name, prefix))
#         dirname = os.path.dirname(file_path)
#         make_dir_exist(dirname)
#         blob.download_to_filename(file_path)

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
    os.makedirs(output_model_dir)

    # download data and pretrained model from GCS
    os.makedirs(parsed_args.data_dl_dir)
    gsutil_rsync(
        os.path.join('gs://' + parsed_args.bucket_name, parsed_args.data_prefix), 
        parsed_args.data_dl_dir)
    os.makedirs(parsed_args.model_dl_dir)
    gsutil_rsync(
        os.path.join('gs://' + parsed_args.bucket_name, parsed_args.model_prefix), 
        parsed_args.model_dl_dir)
    
    # load configs
    yolo_args = YoloArgs(parsed_args.config_file)

    # generate new training paths from data
    data_subsets = {'training': 'train', 'validation': 'val', 'testing_real': 'test'}
    convert_kitti_data_to_yolo(
        parsed_args.data_dl_dir, 
        parsed_args.data_dl_dir, 
        data_subsets=data_subsets)

    # adapt some yolo_args attributes to cloud
    yolo_args.train_file = os.path.join(parsed_args.data_dl_dir, os.path.basename(yolo_args.train_file))
    yolo_args.val_file = os.path.join(parsed_args.data_dl_dir, os.path.basename(yolo_args.val_file))
    yolo_args.restore_path = os.path.join(parsed_args.model_dl_dir, os.path.basename(yolo_args.restore_path))
    yolo_args.save_dir = os.path.join(output_model_dir, 'checkpoint')
    yolo_args.log_dir = os.path.join(output_model_dir, 'logs')
    yolo_args.progress_log_path = os.path.join(yolo_args.log_dir, os.path.basename(yolo_args.progress_log_path))
    yolo_args.class_name_path = os.path.join(parsed_args.data_dl_dir, os.path.basename(yolo_args.class_name_path))
    make_dir_exist(yolo_args.save_dir)
    make_dir_exist(yolo_args.log_dir)

    # train model
    train(yolo_args)
    if parsed_args.job_dir == '':
        parsed_args.job_dir = 'gs://{}'.format(parsed_args.bucket_name)
    gsutil_rsync(output_model_dir, parsed_args.job_dir)
