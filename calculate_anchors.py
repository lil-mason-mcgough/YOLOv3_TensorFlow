from yolov3_wizyoung.get_kmeans import parse_anno, get_kmeans

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Calculate YOLO anchors.")
    parser.add_argument("annotation_path", type=str,
        help="The path to the train.txt file listing training data paths.")
    parsed_args = parser.parse_args()

    # target resize format: [width, height]
    # if target_resize is speficied, the anchors are on the resized image scale
    # if target_resize is set to None, the anchors are on the original image scale
    target_size = [416, 416]
    anno_result = parse_anno(parsed_args.annotation_path, target_size=target_size)
    anchors, ave_iou = get_kmeans(anno_result, 9)

    print('anchors are:')
    print(anchors)
    print('the average iou is:')
    print(ave_iou)