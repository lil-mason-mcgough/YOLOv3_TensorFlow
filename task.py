import os, shutil
import subprocess

from yolov3_wizyoung.train import train
from yolov3_wizyoung.utils.config_utils import YoloArgs
from yolov3_wizyoung.utils.misc_utils import make_dir_exist, reset_dir, gsutil_rsync, gsutil_cp


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="YOLOV3 train procedure.")
    parser.add_argument("--job_dir", type=str, default="",
        help="The directory of cloud resources for this job.")
    parser.add_argument("--config_file", type=str, default="./config.yaml",
        help="The path of the yaml configuration file.")
    parser.add_argument("--bucket_name", type=str, default="lil-ml",
        help="The bucket name containing data on GCS (i.e. gs://<bucket_name>).")
    parser.add_argument("--reset_dirs", action="store_true",
        help="If set, clears the save and log dirs before execution.")
    parsed_args = parser.parse_args()

    # load configs
    yolo_args = YoloArgs(parsed_args.config_file)

    # reset checkpoint and logs dirs
    if parsed_args.reset_dirs:
        reset_dir(yolo_args.save_dir)
        reset_dir(yolo_args.log_dir)

    # train model
    train(yolo_args)
    if parsed_args.job_dir == '':
        parsed_args.job_dir = 'gs://{}'.format(parsed_args.bucket_name)
    gsutil_rsync(yolo_args.save_dir, os.path.join(parsed_args.job_dir, 'checkpoint'))
    gsutil_rsync(yolo_args.log_dir, os.path.join(parsed_args.job_dir, 'logs'))
    gsutil_cp(parsed_args.config_file, parsed_args.job_dir)
