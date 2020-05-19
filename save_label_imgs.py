import os, glob

import cv2

from yolov3_wizyoung.utils.config_utils import YoloArgs
from yolov3_wizyoung.utils.plot_utils import get_color_table, plot_one_box


def imread(img_path):
    img = cv2.imread(img_path)
    return img

def parse_line(line):
    line_s = line.split(' ')
    try:
        i = int(line_s[0])
        img_path = line_s[1]
        dims = (line_s[2], line_s[3])

        labels_s = line_s[4:]
        n_labels = len(labels_s) // 5
        labels = []
        boxes = []
        for i in range(n_labels):
            labels.append(int(labels_s[i * 5]))
            boxes.append(labels_s[i * 5 + 1:(i + 1) * 5])

    except IndexError:
        raise('Line does not match expected format. "{}"'.format(line))

    return i, img_path, dims, labels, boxes

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
    parser.add_argument("input_file", type=str, help="The txt file for input images.")
    parser.add_argument("output_dir", type=str, help="The directory of output images.")
    parser.add_argument("--config_path", type=str, default="./config.yaml",
                        help="The path to the config file.")
    args = parser.parse_args()

    yolo_args = YoloArgs(args.config_path)
    args.anchors = yolo_args.anchors
    args.classes = yolo_args.classes
    color_table = get_color_table(len(args.classes))

    with open(args.input_file, 'r') as f:
        img_lines = f.read()

    for img_line in img_lines.split('\n'):
        if len(img_line) == 0:
            continue

        _, img_path, dims, labels, boxes = parse_line(img_line)
        img = imread(img_path)

        for i in range(len(boxes)):
            x0, y0, x1, y1 = boxes[i]
            plot_one_box(
                img, [x0, y0, x1, y1], 
                label=args.classes[labels[i]],
                color=color_table[labels[i]])
        cv2.imshow('Labels', img)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(args.output_dir, img_name + '.jpg'), img)

    