# YOLOv3-WizYoung

## Introduction

This repo is a refactor of the [YOLOv3 implementation](https://github.com/wizyoung/YOLOv3_TensorFlow) by GitHub user [wizyoung](https://github.com/wizyoung/). It has been restructured to be deployable with `setuptools` and some additional scripts and modules have been added.

## Installation

Setup has been slightly simplified from the original project. Simply install it to the python environment using `pip`:

```
pip install .
```

The package can then be imported:

```
import yolov3_wizyoung
```

## Usage

Unlike the original package which used `args.py` for training configuration parameters, this one relies on a YAML file. This decouples project configuration from package implementation, making it simpler to manage multiple YAML files for various datasets / configurations and preventing application-specific configuration from entering version control. 

### Generate data annotations

Data needs to be consumable in a specific format to be used by this project. That format is described in section 7.1 of the original documentation (below). Several helper scripts are in yolov3_wizyoung to help with various formats.

There are some helper scripts to assist with converting the data from the NVIDIA ISAAC scene into the format described in section 7.1 of the original documentation. The `prepare_data.py` script simultaneously converts data that comes from the NVIDIA ISAAC scene and data that comes from makesense.io bounding box annotations into the format for this project. The input data to `prepare_data.py` must be stored in a directory either in GCS or on local file storage with the following structure:

```
classes.names
training
+---image_2
+---labels_2
+---makesense_images
+---makesense_labels
validation
+---image_2
+---labels_2
+---makesense_images
+---makesense_labels
testing
+---image_2
+---labels_2
+---makesense_images
+---makesense_labels
```

The `image_2` and `labels_2` files are generated directly from the NVIDIA ISAAC scene. The `makesense_images` and `makesense_labels` files are hand-made annotations which come directly from makesense.io. Neither set of files is strictly necessary for `training`, `validation`, or `testing` folders, but they must come in pairs. The `prepare_data.py` script will optionally copy these files down to local storage, then combine the ISAAC and makesense.io images into `train.txt`, `val.txt`, and `test.txt` lists. Update the `config.yaml` file to point to these lists.

### Generate anchors

When using your own data for YOLO, you probably want to generate your own anchors. This can be done using the following code, converting literals to reflect your project:

```
from yolov3_wizyoung.get_kmeans import parse_anno, get_kmeans

train_path = './data/train.txt'
n_anchors = 9
img_size = [416, 416]

anno_result = parse_anno(train_path, target_size=img_size)
anchors, avg_iou = get_kmeans(anno_result, n_anchors)
print(anchors)
```

Copy the anchors as they appear in the console into `config.yaml` under the line `anchors: ...`. You can also use the `calculate_anchors.py` script.

### Download Darknet weights

The Darknet weights can be downloaded [here](https://pjreddie.com/media/files/yolov3.weights). After doing so, they need to be converted into a checkpoint file for TensorFlow to consume. Execute the following code, converting literals to reflect your project structure:

```
from yolov3_wizyoung import convert_weight

anchors = <replace with anchors in previous section>

convert_weight(
    './yolov3.weights', 
    './yolov3.ckpt',
    [416, 416] # input image dimensions,
    anchors)
```

### Edit config

Some parameters in `config.yaml` need to be modified to use this project. At minimum the first paragraph of parameters need to reflect your project. 

To run a training session using your own configurations, modify `config.yaml` and then use the following example:

```
from yolov3_wizyoung.train import train
from yolov3_wizyoung.utils.config_utils import YoloArgs

config_file = 'config.yaml'
args = YoloArgs(config_file)
train(args)
```

### Visualize results

There are two scripts for validating and visualizing trained models. The first, `eval.py`, will calculate the recall, precision, and mAP of a model checkpoint against a testing dataset. The second script, `visualize.py`, will display the predicted bounding boxes, classes, and confidence scores for a test set of images and save them to a given directory. 

## Additional notes

For more information, the original documentation can be found below:

## (ORIGINAL DOCUMENTATION) YOLOv3_TensorFlow  

### 1. Introduction

This is my implementation of [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) in pure TensorFlow. It contains the full pipeline of training and evaluation on your own dataset. The key features of this repo are:

- Efficient tf.data pipeline
- Weights converter (converting pretrained darknet weights on COCO dataset to TensorFlow checkpoint.)
- Extremely fast GPU non maximum supression.
- Full training and evaluation pipeline.
- Kmeans algorithm to select prior anchor boxes.

### 2. Requirements

Python version: 2 or 3

Packages:

- tensorflow >= 1.8.0 (theoretically any version that supports tf.data is ok)
- opencv-python
- tqdm

### 3. Weights convertion

The pretrained darknet weights file can be downloaded [here](https://pjreddie.com/media/files/yolov3.weights). Place this weights file under directory `./data/darknet_weights/` and then run:

```shell
python convert_weight.py
```

Then the converted TensorFlow checkpoint file will be saved to `./data/darknet_weights/` directory.

You can also download the converted TensorFlow checkpoint file by me via [[Google Drive link](https://drive.google.com/drive/folders/1mXbNgNxyXPi7JNsnBaxEv1-nWr7SVoQt?usp=sharing)] or [[Github Release](https://github.com/wizyoung/YOLOv3_TensorFlow/releases/)] and then place it to the same directory.

### 4. Running demos

There are some demo images and videos under the `./data/demo_data/`. You can run the demo by:

Single image test demo:

```shell
python test_single_image.py ./data/demo_data/messi.jpg
```

Video test demo:

```shell
python video_test.py ./data/demo_data/video.mp4
```

Some results:

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/data/demo_data/results/dog.jpg?raw=true)

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/data/demo_data/results/messi.jpg?raw=true)

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/data/demo_data/results/kite.jpg?raw=true)

Compare the kite detection results with TensorFlow's offical API result [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/img/kites_detections_output.jpg).

(The kite detection result is under input image resolution 1344x896)

### 5. Inference speed

How fast is the inference speed? With images scaled to 416*416:


| Backbone              |   GPU    | Time(ms) |
| :-------------------- | :------: | :------: |
| Darknet-53 (paper)    | Titan X  |    29    |
| Darknet-53 (my impl.) | Titan XP |   ~23    |

why is it so fast? Check the ImageNet classification result comparision from the paper:

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/docs/backbone.png?raw=true)

### 6. Model architecture

For better understanding of the model architecture, you can refer to the following picture. With great thanks to [Levio](https://blog.csdn.net/leviopku/article/details/82660381) for your excellent work!

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/docs/yolo_v3_architecture.png?raw=true)

### 7. Training

#### 7.1 Data preparation 

(1) annotation file

Generate `train.txt/val.txt/test.txt` files under `./data/my_data/` directory. One line for one image, in the format like `image_index image_absolute_path img_width img_height box_1 box_2 ... box_n`. Box_x format: `label_index x_min y_min x_max y_max`. (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).) `image_index` is the line index which starts from zero. `label_index` is in range [0, class_num - 1].

For example:

```
0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
...
```

Since so many users report to use tools like LabelImg to generate xml format annotations, I add one demo script on VOC dataset to do the convertion. Check the `misc/parse_voc_xml.py` file for more details.

(2)  class_names file:

Generate the `data.names` file under `./data/my_data/` directory. Each line represents a class name.

For example:

```
bird
person
bike
...
```

The COCO dataset class names file is placed at `./data/coco.names`.

(3) prior anchor file:

Using the kmeans algorithm to get the prior anchors:

```
python get_kmeans.py
```

Then you will get 9 anchors and the average IoU. Save the anchors to a txt file.

The COCO dataset anchors offered by YOLO's author is placed at `./data/yolo_anchors.txt`, you can use that one too.

The yolo anchors computed by the kmeans script is on the resized image scale.  The default resize method is the letterbox resize, i.e., keep the original aspect ratio in the resized image.

#### 7.2 Training

Using `train.py`. The hyper-parameters and the corresponding annotations can be found in `args.py`:

```shell
CUDA_VISIBLE_DEVICES=GPU_ID python train.py
```

Check the `args.py` for more details. You should set the parameters yourself in your own specific task.

### 8. Evaluation

Using `eval.py` to evaluate the validation or test dataset. The parameters are as following:

```shell
$ python eval.py -h
usage: eval.py [-h] [--eval_file EVAL_FILE] 
               [--restore_path RESTORE_PATH]
               [--anchor_path ANCHOR_PATH] 
               [--class_name_path CLASS_NAME_PATH]
               [--batch_size BATCH_SIZE]
               [--img_size [IMG_SIZE [IMG_SIZE ...]]]
               [--num_threads NUM_THREADS]
               [--prefetech_buffer PREFETECH_BUFFER]
               [--nms_threshold NMS_THRESHOLD]
               [--score_threshold SCORE_THRESHOLD] 
               [--nms_topk NMS_TOPK]
```

Check the `eval.py` for more details. You should set the parameters yourself. 

You will get the loss, recall, precision, average precision and mAP metrics results.

For higher mAP, you should set score_threshold to a small number.

### 9. Some tricks

Here are some training tricks in my experiment:

(1) Apply the two-stage training strategy or the one-stage training strategy:

Two-stage training:

First stage: Restore `darknet53_body` part weights from COCO checkpoints, train the `yolov3_head` with big learning rate like 1e-3 until the loss reaches to a low level.

Second stage: Restore the weights from the first stage, then train the whole model with small learning rate like 1e-4 or smaller. At this stage remember to restore the optimizer parameters if you use optimizers like adam.

One-stage training:

Just restore the whole weight file except the last three convolution layers (Conv_6, Conv_14, Conv_22). In this condition, be careful about the possible nan loss value.

(2) I've included many useful training strategies in `args.py`:

- Cosine decay of lr (SGDR)
- Multi-scale training
- Label smoothing
- Mix up data augmentation
- Focal loss

These are all good strategies but it does **not** mean they will definitely improve the performance. You should choose the appropriate strategies for your own task.

This [paper](https://arxiv.org/abs/1902.04103) from gluon-cv has proved that data augmentation is critical to YOLO v3, which is completely in consistent with my own experiments. Some data augmentation strategies that seems reasonable may lead to poor performance. For example, after introducing random color jittering, the mAP on my own dataset drops heavily. Thus I hope  you pay extra attention to the data augmentation.

(4) Loss nan? Setting a bigger warm_up_epoch number or smaller learning rate and try several more times. If you fine-tune the whole model, using adam may cause nan value sometimes. You can try choosing momentum optimizer.

### 10. Fine-tune on VOC dataset

I did a quick train on the VOC dataset. The params I used in my experiments are included under `misc/experiments_on_voc/` folder for your reference. The train dataset is the VOC 2007 + 2012 trainval set, and the test dataset is the VOC 2007 test set.

Finally with the 416\*416 input image, I got a 87.54% test mAP (not using the 07 metric). No hard-try fine-tuning. You should get the similar or better results.

My pretrained weights on VOC dataset can be downloaded [here](https://drive.google.com/drive/folders/1ICKcJPozQOVRQnE1_vMn90nr7dejg0yW?usp=sharing).

### 11. TODO

[ ] Multi-GPUs with sync batch norm. 

[ ] Maybe tf 2.0 ?

-------

### Credits:

I referred to many fantastic repos during the implementation:

[YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)

[qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)

[eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

[pjreddie/darknet](https://github.com/pjreddie/darknet)

[dmlc/gluon-cv](https://github.com/dmlc/gluon-cv/tree/master/scripts/detection/yolo)

