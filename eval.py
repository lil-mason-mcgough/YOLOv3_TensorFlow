# coding: utf-8

from __future__ import division, print_function

import os
import argparse

import tensorflow as tf
import numpy as np
import cv2
from tqdm import trange

from yolov3_wizyoung.utils.data_utils import get_batch_data
from yolov3_wizyoung.utils.misc_utils import parse_anchors, read_class_names, AverageMeter
from yolov3_wizyoung.utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu, \
    get_preds_gpu, voc_eval, parse_gt_rec
from yolov3_wizyoung.utils.data_aug import letterbox_resize
from yolov3_wizyoung.utils.nms_utils import gpu_nms
from yolov3_wizyoung.utils.config_utils import YoloArgs
from yolov3_wizyoung.utils.plot_utils import get_color_table, plot_one_box

from yolov3_wizyoung.model import yolov3

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="YOLO-V3 eval procedure.")

parser.add_argument("--eval_file", type=str, default="./data/my_data/val.txt",
                    help="The path of the validation or test txt file.")

parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")

parser.add_argument("--config_path", type=str, default="./config.yaml",
                    help="The path of the config file.")

parser.add_argument("--output_dir", type=str, default="./output",
                    help="The directory of output images.")                    

parser.add_argument("--num_threads", type=int, default=10,
                    help="Number of threads for image processing used in tf.data pipeline.")

parser.add_argument("--prefetech_buffer", type=int, default=5,
                    help="Prefetech_buffer used in tf.data pipeline.")

parser.add_argument("--score_threshold", type=float, default=0.5,
                    help="Threshold of the probability of the classes in nms operation.")

args = parser.parse_args()

# args params
yolo_args = YoloArgs(args.config_path)
args.anchors = yolo_args.anchors
args.classes = yolo_args.classes
args.letterbox_resize = yolo_args.letterbox_resize
args.nms_threshold = yolo_args.nms_threshold
args.nms_topk = yolo_args.nms_topk
args.img_size = yolo_args.img_size
args.use_voc_07_metric = yolo_args.use_voc_07_metric
args.class_num = len(args.classes)
label_lines = [x.split(' ') for x in open(args.eval_file, 'r').readlines()]
args.img_paths = {int(x[0]): x[1] for x in label_lines}
args.img_cnt = len(label_lines)
color_table = get_color_table(args.class_num)

# setting placeholders
# is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
# handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, args.class_num, 
    args.nms_topk, args.score_threshold, args.nms_threshold)

##################
# tf.data pipeline
##################
val_dataset = tf.data.TextLineDataset(args.eval_file)
val_dataset = val_dataset.batch(1)
val_dataset = val_dataset.map(
    lambda x: tf.py_func(get_batch_data, [x, args.class_num, args.img_size, 
        args.anchors, 'val', False, False, args.letterbox_resize], 
        [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads)
val_dataset.prefetch(args.prefetech_buffer)
iterator = val_dataset.make_one_shot_iterator()

image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
image_ids.set_shape([None])
y_true = [y_true_13, y_true_26, y_true_52]
image.set_shape([None, args.img_size[1], args.img_size[0], 3])
for y in y_true:
    y.set_shape([None, None, None, None, None])

##################
# Model definition
##################
yolo_model = yolov3(args.class_num, args.anchors)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=False)
loss = yolo_model.compute_loss(pred_feature_maps, y_true)
y_pred = yolo_model.predict(pred_feature_maps)

saver_to_restore = tf.train.Saver()

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer()])
    saver_to_restore.restore(sess, args.restore_path)

    print('\n----------- start to eval -----------\n')

    val_loss_total, val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    val_preds = []

    for j in trange(args.img_cnt):
        __image_ids, __image, __y_pred, __loss = sess.run([image_ids, image, y_pred, loss])
        
        pred_content = get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, 
            pred_scores_flag, __image_ids, __y_pred)
        # pred_content: [[image_id, x_min, y_min, x_max, y_max, score, label],...]
        
        val_preds.extend(pred_content)
        val_loss_total.update(__loss[0])
        val_loss_xy.update(__loss[1])
        val_loss_wh.update(__loss[2])
        val_loss_conf.update(__loss[3])
        val_loss_class.update(__loss[4])

        # display output prediction
        img_path = args.img_paths[__image_ids[0]]
        img_ori = cv2.imread(img_path)
        # rescale the coordinates to the original image
        boxes_ = np.array([x[1:5] for x in pred_content], dtype=np.float32)
        labels_ = np.array([x[-1] for x in pred_content], dtype=np.int64)
        scores_ = np.array([x[-2] for x in pred_content], dtype=np.float32)
        if len(boxes_) > 0:
            if args.letterbox_resize:
                _, resize_ratio, dw, dh = letterbox_resize(img_ori, args.img_size[0], args.img_size[1])
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                height_ori, width_ori = img_ori.shape[:2]
                boxes_[:, [0, 2]] *= (width_ori / float(args.img_size[0]))
                boxes_[:, [1, 3]] *= (height_ori / float(args.img_size[1]))

        # save predictions to display image
        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], 
                label=args.classes[labels_[i]] + ', {:.2f}%'.format(
                    scores_[i] * 100), color=color_table[labels_[i]])
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(args.output_dir, '{}.jpg'.format(img_name))
        cv2.imwrite(output_path, img_ori)

    rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
    gt_dict = parse_gt_rec(args.eval_file, args.img_size, 
        args.letterbox_resize)
    # filter low confidence predictions
    print('mAP eval:')
    for ii in range(args.class_num):
        class_name = args.classes[ii]
        npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, 
            iou_thres=0.5, use_07_metric=args.use_voc_07_metric)
        rec_total.update(rec, npos)
        prec_total.update(prec, nd)
        ap_total.update(ap, 1)
        print('{:2d} - {:32s}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}, ({:4d} true, {:4d} pred)'.format(
            ii, class_name, rec, prec, ap, npos, nd))

    mAP = ap_total.average
    print('final mAP: {:.4f}'.format(mAP))
    print("recall: {:.3f}, precision: {:.3f}".format(rec_total.average, prec_total.average))
    print("total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}".format(
        val_loss_total.average, val_loss_xy.average, val_loss_wh.average, val_loss_conf.average, val_loss_class.average
    ))
