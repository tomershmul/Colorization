# Deep Learning Final Project - Colorful Image Colorization

Implementation of https://arxiv.org/pdf/1603.08511.pdf

# Train
Training example

Train images Folder: '../data/imagenet_20classes_128x128/'

Model will be saved to folder: '../model/models/'

python ./train.py --update_lr 1 --new_arch 0 --num_epochs 100 --dataset "ImageNet128" --image_dir "../data/imagenet_20classes_128x128"

## Training from check point

python ./train.py --update_lr 1 --new_arch 0 --ckpt ../model/models/model-new0-51-512.ckpt --num_epochs 1 --dataset "ImageNet64" --image_dir "../data/imagenet_20classes_64x64"

### Train Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `model_path` | path for saving trained models | ../model/models/ |
| `image_dir` | directory for train images | ../data/imagenet128/ |
| `log_step` | step size for prining log info | 100 |
| `save_step` | step size for saving trained models | 5 |
| `image_dir` | directory for train images | ../data/imagenet128/ |
| `ckpt` | Path to checkpoint |  |
| `num_epochs` | Number of Epochs for training | 50 |
| `batch_size` | Batch Size for training | 20 |
| `num_workers` | Number of CPU Workers | 16 |
| `learning_rate` | Learning Rate Starting Value | 1e-4 |
| `update_lr` | Boolean, if TRUE updating lr every num_epochs/3 | TRUE |
| `new_arch` | Boolean, TRUE -> Dilated Model; False -> Compact Model | TRUE |
| `dataset` | Images size in dateset (ImageNet128 or ImageNet64) | ImageNet128 |


# Validation
Validation example
Validation images Folder: '../data/val/'
Inferenced images will be saved to folder: '../data/colorimg/'

python ./sample_imagenet.py --new_arch 0 --ckpt ../model/models/128x128_orig_arch/model-new0-100-640.ckpt

### Validation Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `ckpt` | Path to checkpoint |  |
| `new_arch` | Boolean, TRUE -> Dilated Model; False -> Compact Model | TRUE |
| `dataset` | Images size in dateset (ImageNet128 or ImageNet64) | ImageNet128 |


