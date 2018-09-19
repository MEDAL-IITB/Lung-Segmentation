# Hybrid Dilated Convolution and Dense Upsampling Convolution for Lung Segmentation
*   [Introduction](#introduction)
*   [What are HDC and DUC?](#what-are-hdc-and-duc)
*   [Architecture](#architecture)
* [Training](#training)
* [Results](#results)
*   [References](#references)
## Introduction

The paper titled "Understanding Convolution for Semantic Segmentation" [$^{[1]}$](https://arxiv.org/abs/1702.08502.pdf) describes Hybrid Dilated Convolution (HDC) and Dense Upsampling Convolution (DUC) frameworks on a base CNN architecture for semantic segmentation of images. This architecture achieves state-of-the-art (at the time of submission) results on Cityscapes, KITTI Road estimation (overall) and PASCALVOC 2012 datasets. 

We use the HDC and DUC frameworks proposed in the paper above on the Montgomery County X-Ray Set which contains 138 posterior-anterior chest X-Ray images. 

## What are HDC and DUC?

![
](https://lh3.googleusercontent.com/Hc5LBZAz15_VEPg3nxUP5vxfsawHip3xSwILiVhBDvQSlScaQYtv_k3WIyBFNIOsaiPtNri5z2g0 "hdc_arch")
**Hybrid Dilated Convolution (HDC)** is a set of convolution layers in which each layer has a different rate of dilation ($r_i$). HDC helps detect finer details at higher resolutions. It effectively enlarges receptive fields of the network to aggregate global information using lesser number of parameters than a conventional convolution with a same-sized receptive field hence making it computationally more efficient. It also eliminates the issues of gridding which causes loss of local information.

Suppose, we have N convolutional layers with kernel size K*K that have dilation rates of [$r_1, ..., r_i , ..., r_n$], the goal of HDC is to let the final size of the RF of a series of convolutional operations fully covers a square region without any holes or missing edges.
The “maximum distance between two nonzero values” is defined as $M_i = max[$M_i+1$-$2r_i , M_i+1$−$2(M_i+1$−$r_i), r_i$ ],  with $M_n = r_n$. The design goal is to let $M_2 ≤ K$.

Note-Dilation rates such as [2,4,8] can't be used as they gridding still happens with rates which have a common factor.

![
](https://lh3.googleusercontent.com/rlkiBpYIHdu3bssClJUAssPeNW7UwxurR3EzM7BNO0RQCbjTbG34Ym-h3EpiBdWQ6NVwGauAT5fX "dilatedconv")
(a)Conventional Dilation causes gridding (b) HDC prevents gridding

**Dense Upsampling Convolutional** is a convolution operation performed on the feature map (h x w x c) obtained as the output of the backbone CNN to output a feature map of dimensions (H x W x L) where h=H/d, w=W/d, c=$d^2$L, H*W*W x C are the dimensions of the input image, L is the number of classes of segmentation, and d is downsampling factor. The output of the DUC layer is then  reshaped to H × W × L with a softmax layer, and an element-wise argmax operator is applied to get the final label map. The upsampling done here is dense in the sense that we are performing calculations on every pixel and no zeroes are involved. This is better than bilinear upsampling as DUC is learnable. DUC is particularly good for at detecting small and far off objects. Also, it is very easy to implement.

## Architecture

For our task of lung segmentation from X-ray images, we use a ResNet101 (with pretrained weights from ImageNet) architecture with a HDC unit and then a DUC unit. The HDC unit consisted of grouping every four blocks together in the res4b module and using 1, 2, 5, and 9 dilation rates respectively. We also modify the dilation rates of the res5b module to 5, 9 and 17. 

## Training

We used a random split of 57 images for training, 20 images for cross-validation and 61 testing images. The loss used was a combination of Binary cross entropy, dice and inverse dice losses. The metrics chosen for evaluation are mIoU and Mean Dice scores.
images for testing.
Since, the dataset is small, augmentation was implemented. The images were flipped horizontally and vertically. Gaussian noise was then added to these images. 
After observing initial training results, a weighted mean of Binary Cross Entropy Loss, Dice Loss and Inverse Dice Loss was chosen
 with weights 1, 1 and 1.5. 

The model with the best validation score was chosen which is the following-
Trained for 100 epochs with an Adam Optimiser and a batch size of 5 and a scheduled learning rate.
The cross-validation scores were Mean IoU: 0.7863257 and Mean Dice: 0.8781220

## Results
The results over the test data were Mean IoU: 0.7461923 and Mean Dice: 0.8500786.
![
](https://lh3.googleusercontent.com/02cq0q1Jj1AI_U3laEazAQ8wdISfC_mvDsFzU369v0oW1ByDGkEKUtyadnIT0Es7NTeIXGEb6NKb "30")
![
](https://lh3.googleusercontent.com/7sDIuCWYfxR7u-cj_LykVV-fWF7Ql8jE9G443Mv5OkpnBlIksT3_xlK0vPb03flWwFRKgtDOdPLe "17")

![
](https://lh3.googleusercontent.com/_hjqTg4fmaYyDLlOCcHI3Ppc7F-qgkGVZ1pSC-XFRgLyAaRYb4w18cSwxrSXuyCP72JQ_V3nbidq "32")

## References
[[1]](https://arxiv.org/abs/1702.08502) Panqu Wang, Pengfei Chen, Ye Yuan, Ding Liu, Zehua Huang, Xiaodi Hou, and Garrison W. Cottrell. Understanding convolution for semantic segmentation. CoRR, abs/1702.08502, 2017


[[2]](https://arxiv.org/abs/1512.03385) Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015.

[[3]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. ”Imagenet classification with deep convolutional neural networks.” In Advances in neural information processing systems, pp. 1097-1105. 2012

[[4]](https://arxiv.org/abs/1409.4842) Szegedy, Christian, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. ”Going deeper with convolutions.” In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1-9. 2015

[[5]](https://arxiv.org/abs/1502.03167) Ioffe, Sergey, and Christian Szegedy. ”Batch normalization: Accelerating deep network training by reducing internal covariate shift.” In International Conference on Machine Learning, pp. 448-456. 2015

[[6]](http://jmlr.org/papers/v15/srivastava14a.html) Srivastava, Nitish, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. ”Dropout: a simple way to prevent neural networks from overfitting.”Journal of machine learning research 15, no. 1 (2014): 1929-1958

[[7]](https://arxiv.org/abs/1409.1556) Simonyan, Karen, and Andrew Zisserman. ”Very deep convolutional networks

[8] Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006

[[9]](http://www.deeplearningbook.org) Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press,2016

[[10]](https://www.tensorflow.org/) TensorFlow

[[11]](https://keras.io/) Keras

[[12]]([https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html))  Pytorch

Return to main [README](www.github.com/medal-iitb/LungSegmentation/README.md) .


