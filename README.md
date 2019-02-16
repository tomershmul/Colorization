# Deep Learning Final Project - Colorful Image Colorization

Implementation of https://arxiv.org/pdf/1603.08511.pdf

# Train
Training example
Train images Folder: '../data/imagenet_20classes_128x128/'
Model will be saved to folder: '../model/models/'

python ./train.py --update_lr 1 --new_arch 0 --num_epochs 100 --dataset "ImageNet128" --image_dir "../data/imagenet_20classes_128x128"

# Validation
Validation example
Validation images Folder: '../data/val/'
Inferenced images will be saved to folder: '../data/colorimg/'

python ./sample_imagenet.py --new_arch 0 --ckpt ../model/models/128x128_orig_arch/model-new0-100-640.ckpt

