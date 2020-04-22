import os


def load_classes_file(file_path):
    with open(file_path, 'r') as f:
        classes = []
        for l in f:
            class_name = l.strip()
            if len(class_name) == 0:
                continue
            classes.append(class_name)

    return classes

def convert_bboxes_to_automl(data_dir, output_path, gcs_prefixes):
    data_files = {'TRAIN': 'train.txt', 'VALIDATE': 'val.txt', 'TEST': 'test.txt'}
    
    classes_file_path = os.path.join(data_dir, 'data.names')
    classes_list = [c.replace('-', '_') for c in load_classes_file(classes_file_path)]

    with open(output_path, 'w') as f_out:
        n_lines = 0
        for data_set, data_file in data_files.items():
            data_path = os.path.join(data_dir, data_file)
            with open(data_path, 'r') as f_in:
                for l_in in f_in:
                    l_list = l_in.strip().split(' ')
                    img_path = l_list[1]
                    img_dims = l_list[2:4]

                    img_name = os.path.basename(img_path)
                    gcs_path = os.path.join(gcs_prefixes[data_set], img_name)

                    img_detections = l_list[4:]
                    n_detections = len(img_detections) // 5
                    assert len(img_detections) % 5 == 0, \
                        'Improper format in {} (line {})'.format(data_path, l_list[0])
                    for det_idx in range(n_detections):
                        img_label_idx = img_detections[det_idx * 5]
                        img_bbox = img_detections[5 * det_idx + 1:5 * (det_idx + 1)]
                        label = classes_list[int(img_label_idx)]
                        ulc_c = float(int(img_bbox[0])) / int(img_dims[0])
                        ulc_r = float(int(img_bbox[1])) / int(img_dims[1])
                        brc_c = float(int(img_bbox[2])) / int(img_dims[0])
                        brc_r = float(int(img_bbox[3])) / int(img_dims[1])

                        line = ','.join([
                            data_set,
                            gcs_path,
                            label,
                            '{:.4f}'.format(ulc_c),
                            '{:.4f}'.format(ulc_r),
                            '',
                            '',
                            '{:.4f}'.format(brc_c),
                            '{:.4f}'.format(brc_r),
                            '',
                            ''
                        ]) + '\n'
                        f_out.write(line)
                        n_lines += 1
        print('Wrote {} lines to {}'.format(n_lines, output_path))


if __name__ == '__main__':
    data_dir = 'data/dewalt_escondido'
    output_path = 'data.csv'
    gcs_prefixes = {
        'TRAIN': 'gs://lil-ml/product_detection/data/kitti_dewalt_escondido/training/image_2',
        'VALIDATE': 'gs://lil-ml/product_detection/data/kitti_dewalt_escondido/validation/image_2',
        'TEST': 'gs://lil-ml/product_detection/data/kitti_dewalt_escondido/testing_real/image_2'
    }
    convert_bboxes_to_automl(data_dir, output_path, gcs_prefixes)