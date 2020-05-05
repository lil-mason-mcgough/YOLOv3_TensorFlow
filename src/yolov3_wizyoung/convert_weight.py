# coding: utf-8
# for more details about the yolo darknet weights file, refer to
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import numpy as np

from .utils.misc_utils import parse_anchors, load_weights
from .model import yolov3

def convert_weight(weight_path, save_path, img_dims, anchors):
    model = yolov3(80, anchors)
    with tf.Session() as sess:
        inputs = tf.placeholder(tf.float32, [1, img_dims[0], img_dims[1], 3])

        with tf.variable_scope('yolov3'):
            feature_map = model.forward(inputs)
            print('Feature map: {}'.format(feature_map.shape))

        saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

        load_ops = load_weights(tf.global_variables(scope='yolov3'), weight_path)
        sess.run(load_ops)
        saver.save(sess, save_path=save_path)
        print('TensorFlow model checkpoint has been saved to {}'.format(save_path))

if __name__ == '__main__':
    weight_path = './data/darknet_weights/yolov3.weights'
    save_path = './data/darknet_weights/yolov3.ckpt'
    img_dims = [416, 416]
    anchors = parse_anchors('./data/yolo_anchors.txt')
    
    convert_weight(weight_path, save_path, img_dims, anchors)
