from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'opencv-python>=4.2',
    'Pillow>=5.1.0',
    'pytz>=2019.3',
    'pyyaml>=5.3.1',
    'tensorflow-gpu>=1.15.0,<2.0.0',
    'tqdm>=4.43.0',
    'requests>=2.23.0',
]
setup(
    name='yolov3-wizyoung',
    version='0.1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description='YOLO for LIL (based on https://github.com/wizyoung/YOLOv3_TensorFlow)'
)