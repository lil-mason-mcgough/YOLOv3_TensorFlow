# coding: utf-8

from __future__ import division, print_function

import os, glob
import argparse

import tensorflow as tf
import numpy as np
import cv2

from yolov3_wizyoung.utils.misc_utils import parse_anchors, read_class_names, AverageMeter
from yolov3_wizyoung.utils.nms_utils import gpu_nms
from yolov3_wizyoung.utils.plot_utils import get_color_table, plot_one_box
from yolov3_wizyoung.utils.data_aug import letterbox_resize, resize_with_bbox
from yolov3_wizyoung.utils.eval_utils import voc_eval, parse_gt_rec
from yolov3_wizyoung.utils.config_utils import YoloArgs
from yolov3_wizyoung.utils.data_utils import parse_line
from yolov3_wizyoung.model import yolov3

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")

parser.add_argument("eval_file", type=str, help="The testing images list file.")

parser.add_argument("output_dir", type=str, help="The directory of output images.")

parser.add_argument("--config_path", type=str, default="./config.yaml",
                    help="The path of the config file.")

parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")

parser.add_argument("--score_threshold", type=float, default=0.5,
                    help="Threshold of the probability of the classes in nms operation.")
                
args = parser.parse_args()

# args params
yolo_args = YoloArgs(args.config_path)
args.anchors = yolo_args.anchors
args.classes = yolo_args.classes
args.num_class = len(args.classes)
args.img_size = yolo_args.img_size
args.letterbox_resize = yolo_args.letterbox_resize
args.nms_threshold = yolo_args.nms_threshold
args.nms_topk = yolo_args.nms_topk
color_table = get_color_table(args.num_class)

# parse eval file
rawlines = open(args.eval_file, 'r').readlines()
img_paths = []
for line in rawlines:
    img_id, img_path, boxes, labels, img_width, img_height = parse_line(line)
    img_paths.append(img_path)
gt_dict = parse_gt_rec(args.eval_file, args.img_size, 
    letterbox_resize=args.letterbox_resize)

with tf.Session() as sess:
    is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
    input_data = tf.placeholder(tf.float32, [1, args.img_size[1], args.img_size[0], 3], name='input_data')
    pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
    pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, is_training=False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, 
        args.nms_topk, args.score_threshold, args.nms_threshold)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    val_preds = []
    for img_id, img_path in enumerate(img_paths):
        # read image
        img_ori = cv2.imread(img_path)

        # preprocess image
        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.img_size[0], args.img_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(args.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = img[np.newaxis, :] / 255.

        # predict image bboxes, scores, and labels
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], 
            feed_dict={input_data: img, is_training: False})
        for i in range(len(labels_)):
            val_preds.append([img_id, *boxes_[i], scores_[i], labels_[i]])

        # rescale the coordinates to the original image
        if args.letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori/float(args.img_size[0]))
            boxes_[:, [1, 3]] *= (height_ori/float(args.img_size[1]))

        print("box coords:")
        print(boxes_)
        print('*' * 30)
        print("scores:")
        print(scores_)
        print('*' * 30)
        print("labels:")
        print(labels_)

        # save predictions to display image
        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], 
                label=args.classes[labels_[i]] + ', {:.2f}%'.format(
                    scores_[i] * 100), color=color_table[labels_[i]])
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(args.output_dir, '{}.jpg'.format(img_name))
        cv2.imwrite(output_path, img_ori)
        print('Saved {}'.format(output_path))

    rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
    for class_num in range(args.num_class):
        npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, class_num, 
            iou_thres=0.5, use_07_metric=False)
        rec_total.update(rec, npos)
        prec_total.update(prec, nd)
        ap_total.update(ap, 1)

        class_name = args.classes[class_num]
        print('{:2d} - {:32s}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}, ({:4d} true, {:4d} pred)'.format(
            class_num, class_name, rec, prec, ap, npos, nd))

    mAP = ap_total.average
    print('final mAP: {:.4f}'.format(mAP))
    print("recall: {:.3f}, precision: {:.3f}".format(rec_total.average, prec_total.average))
