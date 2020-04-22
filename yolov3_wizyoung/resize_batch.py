from PIL import Image


def load_resize_save(src_path, dst_path, max_res=512):
    img = Image.open(src_path)

    src_dims = img.size
    max_dim = max(range(len(src_dims)), key=lambda i: src_dims[i])
    resize_ratio = max_res / src_dims[max_dim]
    dst_dims = list(map(lambda x: round(resize_ratio * x), src_dims))
    dst_dims[max_dim] = max_res
    dst_dims = tuple(dst_dims)
    img = img.resize(dst_dims)

    img.save(dst_path)


if __name__ == '__main__':
    import os, glob

    src_dir = '/home/bricklayer/Workspace/ai-brain/dewalt-subset/braincorp_images/*.jpg'
    dst_dir = '/home/bricklayer/Workspace/ai-brain/product_detection/data/kitti_dewalt_escondido/testing_real/image_2'
    dst_ext = '.png'

    src_paths = glob.glob(src_dir)
    for src_path in src_paths:
        src_name = os.path.splitext(os.path.basename(src_path))[0]
        dst_path = os.path.join(dst_dir, src_name + dst_ext)
        load_resize_save(src_path, dst_path)
        print('Saved {}'.format(dst_path))
