# U-Nets

## 01_U-net

### PyTorch

#### 01_Uygar Kurt [YT](https://youtu.be/HS3Q_90hnDg?si=NmitL-5xzu2EiGwn), [GitHub](https://github.com/uygarkurt/UNet-PyTorch)

- torch.utils.data.random_split

---

#### 02_Digital_srini

- Loss functions for segmentations - [[02_Maths#Loss functions |losses]] 
	- IoU 
	- Dice coefficient
	- Focal loss (2017) - [paper](https://arxiv.org/pdf/1708.02002) 
- Multi class semantic segmentation model training
	- Weight distribution for different class
	- Predict large image using patchify library (**stack in tiff file**).
	- Explore some methods like smooth blending between two patchs
- ==Segmentation models - Library for backbone== 
	- Available in both pytorch and tensorflow
	- Many pre-trained backbones are available for transfer learning.
	- [Refer](https://smp.readthedocs.io/en/latest/) documentation for more information.
	- Example [code](https://github.com/bnsreenu/python_for_microscopists/blob/master/210_multiclass_Unet_using_VGG_resnet_inception.py) 
	- Refer my [Kaggle](https://www.kaggle.com/code/dhamur/u-net-with-backbones) notebook for practical implementations. 
- LinkNet in smp
	- Light weight when compared with unet
	- unet (2015) - [paper](https://arxiv.org/pdf/1505.04597) 
	- linknet (2017) - [paper](https://arxiv.org/pdf/1707.03718) 
- Ensemble technique to enhance the u-net model
	- Use different model (min 3 models)
	- Tutorials - [GitHub](https://github.com/bnsreenu/python_for_microscopists/blob/master/213-ensemble_sign_language.py) [YT](https://www.youtube.com/watch?v=NFIYdYjJams&list=PLZsOBAyNTZwbR08R959iCvYT3qzhxvGOE&index=18) 
	- Practice - [Kaggle](https://www.kaggle.com/code/dhamur/u-net-with-backbones-ensembling) 
	- Backbone comparison [paper](https://iopscience.iop.org/article/10.1088/1742-6596/1544/1/012196/pdf) 

---

#### 03_tools

- torchvision.utils.make_grid
- torchvision.utils.draw_segmentation_mask

---

#### 04_Aladdin Persson [YT](https://www.youtube.com/watch?v=IHq1t7NxS8k&t=278s), [GitHub](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/train.py)

- albumentations
- checkpoint
- autocast and GradeScale
- extensions pt, pth, ckpt, tar and safetensors
- torchvision.utils.save_image
- BatchNorm2d in unet
- pinmemory, num_workers
- t.numel, t.permutr | np.ravel
- tqdm.set_postfix
- Dice loss

---

### Tensorflow
