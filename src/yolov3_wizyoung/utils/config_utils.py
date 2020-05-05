# coding: utf-8

import math

import yaml
import numpy as np

from .misc_utils import read_class_names


class YoloArgs(object):
    _float_args = [
        'batch_norm_decay',
        'weight_decay',
        'learning_rate_init',
        'lr_decay_factor',
        'lr_lower_bound',
        'pw_values',
        'nms_threshold',
        'score_threshold',
        'eval_threshold'
    ]

    def recompute(self):
        self.train_batch_num = int(math.ceil(float(self.train_img_cnt) / self.batch_size))
        self.lr_decay_freq = int(self.train_batch_num * self.lr_decay_epoch)
        self.pw_boundaries = [float(i) * self.train_batch_num + self.global_step for i in self.pw_boundaries]

    def __init__(self, conf_path):
        with open(conf_path) as f:
            args_dict = yaml.load(f, Loader=yaml.FullLoader)

        # convert scientific notation strings into floats
        for arg in self._float_args:
            val = args_dict[arg]
            if isinstance(val, (list, tuple)):
                val = list(map(float, val))
            else:
                val = float(val)
            args_dict[arg] = val

        # parse some params
        args_dict['anchors'] = np.array(args_dict['anchors'])
        args_dict['classes'] = read_class_names(args_dict['class_name_path'])
        args_dict['class_num'] = len(args_dict['classes'])
        args_dict['train_img_cnt'] = len(open(args_dict['train_file'], 'r').readlines())
        args_dict['val_img_cnt'] = len(open(args_dict['val_file'], 'r').readlines())

        for key, val in args_dict.items():
            self.__setattr__(key, val)
        self.recompute()
