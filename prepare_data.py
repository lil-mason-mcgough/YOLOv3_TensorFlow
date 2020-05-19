import os

from yolov3_wizyoung.labels.convert_labels import read_class_names, combine_dataset_paths
from yolov3_wizyoung.utils.misc_utils import make_dir_exist, gsutil_rsync

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="YOLOV3 train procedure.")
    parser.add_argument("--data_bucket", type=str, default="",
        help="The path to bucket training data dir. Ignored if set to \"\".")
    parser.add_argument("--model_bucket", type=str, default="",
        help="The path to bucket model dir. Ignored if set to \"\".")
    parser.add_argument("--local_data_dir", type=str, default="../app_data",
        help="The local directory containing training data. If <data_bucket> set, copies remote data to this directory.")
    parser.add_argument("--local_model_dir", type=str, default="../app_model",
        help="The the local directory containing model data. If <model_bucket> set, copies remote data to this directory.")
    parsed_args = parser.parse_args()

    # download data from GCS
    if len(parsed_args.data_bucket) > 0:
        make_dir_exist(parsed_args.local_data_dir)
        gsutil_rsync(parsed_args.data_bucket, parsed_args.local_data_dir)
    if len(parsed_args.model_bucket) > 0:
        make_dir_exist(parsed_args.local_model_dir)
        gsutil_rsync(parsed_args.model_bucket, parsed_args.local_model_dir)

    # create files listing paths to data
    data_subsets = {
        os.path.join(parsed_args.local_data_dir, 'training'): 'train.txt', 
        os.path.join(parsed_args.local_data_dir, 'validation'): 'val.txt',
        os.path.join(parsed_args.local_data_dir, 'testing'): 'test.txt'}
    classes = read_class_names(os.path.join(parsed_args.local_data_dir, 'classes.names'))
    combine_dataset_paths(data_subsets, parsed_args.local_data_dir, classes)