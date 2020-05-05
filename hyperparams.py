import os
import copy

import yaml
from yolov3_wizyoung.train import train
from yolov3_wizyoung.utils.config_utils import YoloArgs
from yolov3_wizyoung.utils.misc_utils import make_dir_exist

def parse_hyperparams(hyperparam_config_file):
    with open(hyperparam_config_file) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def eval_hyperparams(hyperparams):
    def _eval_hyperparams(hyperparams, all_params, select_params=None):
        hyperparams = copy.deepcopy(hyperparams)
        try:
            key = list(hyperparams.keys())[0]
        except IndexError:
            return False

        if select_params is None:
            select_params = {}

        val = hyperparams.pop(key)
        for el in val:
            select_params[key] = el
            if not _eval_hyperparams(hyperparams, all_params, select_params):
                all_params.append(copy.deepcopy(select_params))
        return True
    
    all_params = []
    _eval_hyperparams(hyperparams, all_params)
    return all_params



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YOLOV3 hyperparameter optimization')
    parser.add_argument("--config_file", type=str, default="./config.yaml",
                        help="The path to the model's yaml config file.")
    parser.add_argument("--hyperparam_config_file", type=str, default="./hyperparams.yaml",
                        help="The path to the hyperparameters yaml config file.")
    parsed_args = parser.parse_args()

    yolo_args = YoloArgs(parsed_args.config_file)
    hyperparams = parse_hyperparams(parsed_args.hyperparam_config_file)
    param_combos = eval_hyperparams(hyperparams)

    base_save_dir = yolo_args.save_dir
    base_log_dir = yolo_args.log_dir
    for param_combo in param_combos:
        for p, v in param_combo.items():
            yolo_args.__setattr__(p, v)
            yolo_args.recompute()

        params_string = '--'.join(['{}-{}'.format(k, v) for k, v in param_combo.items()])

        new_save_dir = os.path.join(base_save_dir, 'hyperparams', params_string)
        new_log_dir = os.path.join(base_log_dir, 'hyperparams', params_string)

        yolo_args.save_dir = new_save_dir
        yolo_args.log_dir = new_log_dir
        yolo_args.progress_log_path = os.path.join(new_log_dir, 'progress.log')
        make_dir_exist(new_save_dir)
        make_dir_exist(new_log_dir)

        print('Training hyperparameters: {}'.format(param_combo))
        train(yolo_args)