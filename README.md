# [Image Style Disentangling for Instance-level Facial Attribute Transfer (CVIU 2021)](https://github.com/XuyangGuo/xuyangguo.github.io/raw/master/database/STD-GAN/STD-GAN.pdf)

[Xuyang Guo](https://xuyangguo.github.io/), [Meina Kan](http://vipl.ict.ac.cn/homepage/mnkan/Publication/), [Zhenliang He](https://lynnho.github.io/), Xingguang Song, [Shiguang Shan](https://scholar.google.com/citations?user=Vkzd7MIAAAAJ)

![TransferShow](https://raw.githubusercontent.com/XuyangGuo/xuyangguo.github.io/master/database/STD-GAN/resources/transfer.png)

![TransferBangs](https://raw.githubusercontent.com/XuyangGuo/xuyangguo.github.io/master/database/STD-GAN/resources/transfer_bangs.png)

![TransferEyeclasses](https://raw.githubusercontent.com/XuyangGuo/xuyangguo.github.io/master/database/STD-GAN/resources/transfer_eyeglasses.png)

![TransferMultiAttributes](https://raw.githubusercontent.com/XuyangGuo/xuyangguo.github.io/master/database/STD-GAN/resources/transfer_multi.png)

## Abstract

> Instance-level facial attribute transfer aims at transferring an attribute including its style from a source face to a target one. Existing studies have limitations on fidelity or correctness. To address this problem, we propose a weakly supervised style disentangling method embedded in Generative Adversarial Network (GAN) for accurate instance-level attribute transfer, using only binary attribute annotations. In our method, the whole attributes transfer process is designed as two steps for easier transfer, which first removes the original attribute or transfers it to a neutral state and then adds the attributes style disentangled from a source face. Moreover, a style disentangling module is proposed to extract the attribute style of an image used in the adding step. Our method aims for accurate attribute style transfer. However, it is also capable of semantic attribute editing as a special case, which is not achievable with existing instance-level attribute transfer methods. Comprehensive experiments on CelebA Dataset show that our method can transfer the style more precisely than existing methods, with an improvement of 39\% in user study, 16.5\% in accuracy, and about 3.3 in FID.

![archi](https://raw.githubusercontent.com/XuyangGuo/xuyangguo.github.io/master/database/STD-GAN/resources/architecture.png)

## Installation

Clone this repo.

```bash
git clone https://github.com/XuyangGuo/STD-GAN.git
cd STD-GAN
```

The code requires python 3.6.

We recommend using Anaconda to regulate packages.

Dependencies:
- PyTorch 1.4
- torchvision, tensorboardX, pillow, skimage
- tqdm, addict

## Editing by Pretrained Model

Please download [the pre-trained model](https://drive.google.com/drive/folders/1UmOnL38F8KutH30hlNr0X1UOybs0ewZ_?usp=sharing), move it to `./` with the correct path `CtrlHair/model_trained`.

```bash
python test.py -a @AttributesName -t @TargetImagePath -s @SourceImagePath -r @ResultImagePath -c 001
```

Here are explanations of parameters:

- `-a @AttributesName` The edited attribute name. Make sure the attribute name appears in the `use_atts` field of the `config` in `config.py`. For example, the 001 config can use `Bangs`, `Smiling`, `Eyeglasses`, `Black_Hair` (`Black_Hair` here refers to turning black hair into other colors). Separate multi attributes to be edited with commas and without spaces.
- `-t @TargetImagePath` The path of target image to edit. Please make sure that the crop method of this image is same as training set. The pretrained model uses CelebA official cropped images.
- `-s @SourceImagePath` The path of style source image. The requirements for the image are the same as the target image. If the source image is not provided, a random style code is sampled from the uniform distribution U(-1,1).
- `-r @ResultImagePath` The path of saving the result image.
- `-c xxx` By default, the value is 001, which corresponds to the config in `experiment_name` in `config.py`. If you want to use other config, please add it to `config.py` and call it by the prefix of its `experiment_name`.

We provide some target images and sample images of each attribute in `input_imgs` for convenient usage.
Example of usage:
```bash
python test.py -a Bangs,Smiling -t input_imgs/Bangs/0/01.jpg -s input_imgs/Bangs/1/02.jpg -r temp/temp.jpg -c 001
```

In addition, for the performance of the validation set during training, please refer to the `sample_training` and `sample_training_multi_diff_x0` folders under `model_trained/001_default`, and refer to `summaries` for the value change of loss funciton.

#### Editing with Batch

If you want to edit with a mass batch, or want to achieve editing functions such as interpolation, multi style sampling, and continuous gradient, etc, please modify the interfaces of `test.py -> test_transfer`.

## Training New Models

#### Data Preparation
Please download the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 

#### Training Networks

```bash
python train.py -c xxx
```

Here is explanation of parameter:
- `-c xxx` Config number. Please write all training-related configurations into `config.py`. For example, write a new `addict.Dict({…})`, fill `002_xxx` into the `experiment_name` field, then use `python train.py -c 002` for training.

Explanation of important hyper-parameters in config:
- `gpu`: GPU number.
- `img_dir`: Cropped images directory of CelebA dataset.
- `att_file`: Annotation file of CelebA attributes.
- `use_atts`: Attributes for training. It is recommended to train three to four attributes at a time.
- `rec_after`: When to start introducing image reconstruction training.
- `display_frequency`: How many iterations to perform the editing of the validation set once.
- `save_frequency`: How many iterations to save the model once.
- `multi_training`: When to start multi-attribute simultaneous editing training.

## Citation
If you use this code for your research, please cite our papers.
```
@article{guo2021image,
  title={Image style disentangling for instance-level facial attribute transfer},
  author={Guo, Xuyang and Kan, Meina and He, Zhenliang and Song, Xingguang and Shan, Shiguang},
  journal={Computer Vision and Image Understanding (CVIU)},
  volume={207},
  pages={103205},
  year={2021},
  publisher={Elsevier}
}
```

This work also inspires our subsequent work [X. Guo, et al., CtrlHair (ECCV2022)](https://github.com/XuyangGuo/CtrlHair) for controllable hair editing.

## References & Acknowledgments
- [LynnHo / AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow)
