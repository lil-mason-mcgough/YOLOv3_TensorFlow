import os, shutil
import subprocess

from yolov3_wizyoung.train import train
from yolov3_wizyoung.labels.convert_labels import read_class_names, convert_kitti_data_to_yolo, convert_makesense_data_to_yolo
from yolov3_wizyoung.utils.config_utils import YoloArgs
from yolov3_wizyoung.utils.misc_utils import make_dir_exist, reset_dir, gsutil_rsync

def prime_dataset_paths(data_subsets, output_data_dir, classes):
    os.makedirs(output_data_dir, exist_ok=True)
    for data_subset, out_filename in data_subsets.items():
        output_data_path = os.path.join(output_data_dir, out_filename)
        kitti_imgs_dir = os.path.join(data_subset, 'image_2')
        kitti_labels_dir = os.path.join(data_subset, 'label_2')
        convert_kitti_data_to_yolo(kitti_imgs_dir, kitti_labels_dir, output_data_path, classes)

        makesense_imgs_dir = os.path.join(data_subset, 'makesense_images')
        makesense_labels_dir = os.path.join(data_subset, 'makesense_labels')
        if os.path.isdir(makesense_imgs_dir) and os.path.isdir(makesense_labels_dir):
            makesense_list_path = os.path.join(output_data_dir, 'makesense_tmp.txt')
            convert_makesense_data_to_yolo(
                makesense_imgs_dir,
                makesense_labels_dir,
                classes,
                makesense_list_path)

            with open(output_data_path, 'a') as f_out:
                with open(makesense_list_path, 'r') as f_in:
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
    parsed_args = parser.parse_args()

    # download data and pretrained model from GCS
    make_dir_exist(parsed_args.data_dl_dir)
    data_source_url = os.path.join('gs://' + parsed_args.bucket_name, parsed_args.data_prefix)
    gsutil_rsync(data_source_url, parsed_args.data_dl_dir)
    make_dir_exist(parsed_args.model_dl_dir)
    model_source_url = os.path.join('gs://' + parsed_args.bucket_name, parsed_args.model_prefix)
    gsutil_rsync(model_source_url, parsed_args.model_dl_dir)

    data_subsets = {
        os.path.join(parsed_args.data_dl_dir, 'training'): 'train.txt', 
        os.path.join(parsed_args.data_dl_dir, 'validation'): 'val.txt',
        os.path.join(parsed_args.data_dl_dir, 'testing'): 'test.txt'}
    classes = read_class_names(os.path.join(parsed_args.data_dl_dir, 'classes.names'))
    prime_dataset_paths(data_subsets, parsed_args.data_dl_dir, classes)

    # load configs
    yolo_args = YoloArgs(parsed_args.config_file)

    # reset checkpoint and logs dirs
    reset_dir(yolo_args.save_dir)
    reset_dir(yolo_args.log_dir)

    # train model
    train(yolo_args)
    if parsed_args.job_dir == '':
        parsed_args.job_dir = 'gs://{}'.format(parsed_args.bucket_name)
    gsutil_rsync(yolo_args.save_dir, os.path.join(parsed_args.job_dir, 'checkpoint'))
    gsutil_rsync(yolo_args.log_dir, os.path.join(parsed_args.job_dir, 'logs'))
