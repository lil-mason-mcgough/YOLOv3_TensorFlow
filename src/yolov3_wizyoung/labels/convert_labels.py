import os
from glob import glob
from warnings import warn

from PIL import Image

IMG_EXT = '.png'
LABEL_EXT = '.txt'


def _write_bbox_line(img_path, label_path, img_idx, detections):
    img_c, img_r = Image.open(img_path).size
    assert os.path.isfile(img_path), \
        '{} has no corresponding file {}.'.format(label_path, img_path)
    img_line = '{} {} {} {} '.format(img_idx, img_path, img_c, img_r)
    return img_line + ' '.join(detections)

def read_class_names(class_names_path):
    with open(class_names_path, 'r') as f:
        class_names = [name for name in f.read().split('\n') if len(name) > 0]
    assert len(class_names) == len(set(class_names)), \
        'Duplicate class names in {}.'.format(class_names_path)
    return class_names

def convert_kitti_data_to_yolo(kitti_dataset_dir, yolo_output_dir, data_subsets=None):
    if data_subsets is None:
        data_subsets = {'training': 'train', 'validation': 'val', 'testing': 'test'}
    # make output paths
    os.makedirs(yolo_output_dir, exist_ok=True)

    # get class names
    class_names_path = os.path.join(kitti_dataset_dir, 'classes.names')
    class_names = read_class_names(class_names_path)
    class_idxs = {name: idx for idx, name in enumerate(class_names)}

    # write class names to new classes file
    with open(os.path.join(yolo_output_dir, 'data.names'), 'w') as f:
        for l in class_names:
            f.write(l + '\n')

    # parse bounding boxes from kitti format
    for data_subset, out_data_subset in data_subsets.items():
        label_dir = os.path.join(kitti_dataset_dir, data_subset, 'label_2')
        label_paths = glob(os.path.join(label_dir, '*' + LABEL_EXT))
        if not len(label_paths) > 0:
            warn('No labels found in {}'.format(label_dir))
        
        output_data_path = os.path.join(yolo_output_dir, 
            '{}{}'.format(out_data_subset, LABEL_EXT))
        with open(output_data_path, 'w') as f_out:
            print('Writing file: {}'.format(output_data_path))
            for i, label_path in enumerate(label_paths):
                # get detections part of line
                with open(label_path, 'r') as f_in:
                    detections = []
                    for l_in in f_in:
                        pl = l_in.split(' ')
                        class_idx = str(class_idxs[pl[0]])
                        bbox = list(map(lambda x: str(int(round(float(x)))), 
                            [pl[4], pl[5], pl[6], pl[7]]))
                        detection_str = ' '.join([class_idx] + bbox)
                        detections.append(detection_str)

                # get image part of line
                img_name = os.path.splitext(os.path.basename(label_path))[0] + IMG_EXT
                img_path = os.path.join(kitti_dataset_dir, data_subset, 'image_2', img_name)

                line = _write_bbox_line(img_path, label_path, i, detections)
                if len(line.split(' ')) < 9:
                    continue
                f_out.write(line + '\n')

def convert_makesense_data_to_yolo(imgs_dir, labels_dir, class_names, output_file):
    img_paths = glob(os.path.join(imgs_dir, '*' + IMG_EXT))
    print('Imgs found: {}'.format(len(img_paths)))
    with open(output_file, 'w') as f_out:
        print('Writing file: {}'.format(output_file))
        for i, img_path in enumerate(img_paths):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_c, img_r = Image.open(img_path).size
            label_path = os.path.join(labels_dir, img_name + LABEL_EXT)
            with open(label_path, 'r') as f_in:
                detections = []
                for l_in in f_in:
                    pl = l_in.strip().split(' ')
                    class_idx = pl[0]
                    bbox_xywh = [img_c * float(pl[1]), img_r * float(pl[2]), img_c * float(pl[3]), img_r * float(pl[4])]
                    width = bbox_xywh[2]
                    height = bbox_xywh[3]
                    bbox = [
                        bbox_xywh[0] - width / 2.,
                        bbox_xywh[1] - height / 2.,
                        bbox_xywh[0] + width / 2.,
                        bbox_xywh[1] + height / 2.
                    ]
                    bbox = list(map(lambda x: str(int(round(x))), bbox))
                    detection_str = ' '.join([class_idx] + bbox)
                    detections.append(detection_str)

                line = _write_bbox_line(img_path, label_path, i, detections)
                if len(line.split(' ')) < 9:
                    continue
                f_out.write(line + '\n')


if __name__ == '__main__':
    kitti_dataset_dir = '/home/bricklayer/Workspace/ai-brain/product_detection/data/kitti_dewalt_escondido'
    yolo_output_dir = 'data/dewalt_escondido'
    convert_kitti_data_to_yolo(kitti_dataset_dir, yolo_output_dir)

    makesense_imgs_dir = '/home/bricklayer/Workspace/ai-brain/product_detection/data/kitti_dewalt_escondido/testing_real/image_2'
    makesense_labels_dir = '/home/bricklayer/Workspace/ai-brain/product_detection/data/kitti_dewalt_escondido/testing_real/labels_dewalt_escondido_subset'
    yolo_output_file = 'data/dewalt_escondido/test.txt'
    class_names = read_class_names('/home/bricklayer/Workspace/ai-brain/product_detection/data/kitti_dewalt_escondido/classes.names')
    convert_makesense_data_to_yolo(makesense_imgs_dir, makesense_labels_dir, class_names, yolo_output_file)