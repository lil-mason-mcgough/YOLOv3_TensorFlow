import os, shutil
import subprocess

from yolov3_wizyoung.train import train
from yolov3_wizyoung.labels.convert_labels import convert_kitti_data_to_yolo, convert_makesense_data_to_yolo
from yolov3_wizyoung.utils.config_utils import YoloArgs
from yolov3_wizyoung.utils.misc_utils import make_dir_exist, reset_dir, gsutil_rsync

def prime_dataset_paths(data_dir, classes):
    # generate new training paths from data
    data_subsets = {'training_syn_+_real': 'train', 'validation': 'val'}
    convert_kitti_data_to_yolo(data_dir, data_dir, data_subsets=data_subsets)

    # generate test data
    convert_makesense_data_to_yolo(
        os.path.join(data_dir, 'testing_real_half', 'image_2'),
        os.path.join(data_dir, 'testing_real_half', 'labels_dewalt_escondido_subset'),
        classes,
        os.path.join(data_dir, 'test.txt'))

    # append real training imgs to list
    if parsed_args.use_real_train_data:
        syn_data_path = os.path.join(data_dir, 'train_syn.txt')
        convert_makesense_data_to_yolo(
            os.path.join(data_dir, 'training_syn_+_real', 'makesense_images'), 
            os.path.join(data_dir, 'training_syn_+_real', 'makesense_labels'), 
            classes, 
            syn_data_path)
        with open(os.path.join(data_dir, 'train.txt'), 'a') as f_out:
            with open(syn_data_path, 'r') as f_in:
                for l_in in f_in:
                    f_out.write(l_in)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="YOLOV3 train procedure.")
    parser.add_argument("--job-dir", type=str, default="",
                    help="The directory of cloud resources for this job.")
    parser.add_argument("--config_file", type=str, default="./config.yaml",
                    help="The path of the yaml configuration file.")
    parser.add_argument("--bucket_name", type=str, default="lil-ml",
                    help="The bucket name of cloud resources.")
    parser.add_argument("--data_prefix", type=str, default="product_detection/data/kitti_dewalt_escondido",
                    help="The path to bucket training data dir relative to the bucket root.")
    parser.add_argument("--data_dl_dir", type=str, default="../app_data",
                    help="The local directory to save bucket training data.")
    parser.add_argument("--model_prefix", type=str, default="product_detection/models/darknet_weights",
                    help="The path to bucket model dir relative to the bucket root.")
    parser.add_argument("--model_dl_dir", type=str, default="../app_model",
                    help="The the local directory to save model data.")
    parser.add_argument("--use_real_train_data", type=bool, default=False,
                    help="If True, add real training data in training_syn_+_real folder.")
    parsed_args = parser.parse_args()

    # load configs
    yolo_args = YoloArgs(parsed_args.config_file)

    # download data and pretrained model from GCS
    make_dir_exist(parsed_args.data_dl_dir)
    gsutil_rsync(
        os.path.join('gs://' + parsed_args.bucket_name, parsed_args.data_prefix), 
        parsed_args.data_dl_dir)
    make_dir_exist(parsed_args.model_dl_dir)
    gsutil_rsync(
        os.path.join('gs://' + parsed_args.bucket_name, parsed_args.model_prefix), 
        parsed_args.model_dl_dir)

    prime_dataset_paths(parsed_args.data_dl_dir, yolo_args.classes)

    # reset checkpoint and logs dirs
    reset_dir(yolo_args.save_dir)
    reset_dir(yolo_args.log_dir)

    # train model
    train(yolo_args)
    if parsed_args.job_dir == '':
        parsed_args.job_dir = 'gs://{}'.format(parsed_args.bucket_name)
    gsutil_rsync(yolo_args.save_dir, os.path.join(parsed_args.job_dir, 'checkpoint'))
    gsutil_rsync(yolo_args.log_dir, os.path.join(parsed_args.job_dir, 'logs'))
