# Introduction
This report describes the usage of **VGG Unet** architecture for **medical image segmentation**.

We divide the article into the following parts

 - [Dataset](#dataset)
 - [VGG Unet Architecture](#vgg-unet)
 - [Results](#results)
 - [References](#references)

# Dataset
## Montgomory Dataset

The dataset contains Chest X-Ray images. We use this dataset to perform lung segmentation. 
>The dataset can be found [here](http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip)

  

Structure:

We make the following structure of the given data set:

![](https://lh4.googleusercontent.com/J-kHm2BX9ywKISMuY_BaCaFf--UuPJOKlFYLO89gYgvjmqlM9RrFive2wOU30X8N7bzI03uwMCtnb_oCHDPaobyxTMEFlfsTSNXALS629uuAkSUZfm9y-lUv5FORquPe1P8CPp4p)

## Data Preprocessing
We apply random rotations as the only augmentation technique, as any other technique like center crop or flip will distort the data, and the results won’t be as expected

Since each image is approx. `4000X4000`, we resize the images to a manageable size of `512X512` as we were limited by the GPU memory.
  

# VGG Unet

## Introduction

We use **Ternaus-Net** *([Vladimir Iglovikov](https://arxiv.org/search?searchtype=author&query=Iglovikov%2C+V), [Alexey Shvets](https://arxiv.org/search?searchtype=author&query=Shvets%2C+A))* a network that is used to train on medical images to segment the image according to a given mask, that uses a `VGG 11` pretrained encoder.    
  
## Implementation of the network

The VGG 11 is implemented as `configuration A` specified in the following image

> ![enter image description here](https://qph.ec.quoracdn.net/main-qimg-30abbdf1982c8cb049ac65f3cf9d5640)
> (source: https://www.quora.com/How-does-VGG-network-architecture-work)
we remove the FC, last maxpool and soft max layers and then the upsampling layers are mirrored to match the channels of each VGG layer. Skip connections are added per layer. The entire network can be visualized as

![](https://camo.githubusercontent.com/cf2ff198ddd4f4600726fa0f2844e77c4041186b/68747470733a2f2f686162726173746f726167652e6f72672f776562742f68752f6a692f69722f68756a696972767067706637657377713838685f783761686c69772e706e67)
(Source: https://github.com/ternaus/TernausNet)

VGG net can be visualized as:

![](https://www.cs.toronto.edu/~frossard/post/vgg16/vgg16.png)
(Source: https://www.cs.toronto.edu/~frossard/post/vgg16/)
  

## Working

We input the images to the network, which is first passed through the VGG 11 encoder, this outputs a 512 channel feature map. This feature map is then upsampled back to the orignal size using transposed convolutions

The Transposed Convolution can be mathematically shows in the simplest form as an operation to upsample a given feature map to the desired dimentions.

  

![](https://lh5.googleusercontent.com/qOJ46aQEsUShQswuF9m7Sj7ZVocttxzxZHBm1jzhpb80gE8VSDpzBayc2KGnaCC2INmoUbrXu3-HUXNfzWRngfj3fewcnQ0aZzqSMVO5LDu7UQwlIuaMjaTs-0YlUkrKH_kQCohR)
(Source: https://datascience.stackexchange.com/a/20176)

At each upsampling layer, a skip connection to its corresponding layer in the encoder is added, the channels from both layers are concatenated and this is used as input for the next upsampling layer.

Finally, on the final layer, sigmoid activation is applied and the resulting feature map is the segmented image. 

## Loss Functions used 
We use two loss functions here, viz. `Binary Cross Entropy` and `Dice loss`

### Binary Cross Entropy Loss
Cross Entropy measures the probability of an item belonging to a particular class, binary cross entropy is the same concept, except that here there are only 2 classes

It is defined mathematically as
In binary classification, where the number of classes  $M$  equals 2, cross-entropy can be calculated as:

$$
-(y * log(p) + (1-y)* log(1-p) ) 
$$
here  $p$ is the prediction value and $y$ is the ground

### Dice Coefficient Loss
The dice coefficient loss is used to measure the `intersection over union` of the output and target image. 

Mathematically, Dice Score is 
$$\frac{2 * |X \cap Y|}{|X| + |Y|}$$


The dice loss is defined in code as :

    
	class SoftDiceLoss(nn.Module):
	    def __init__(self, weight=None, size_average=True):
	        super(SoftDiceLoss, self).__init__()

	    def forward(self, logits, targets):
	        smooth = 1
	        num = targets.size(0)
	        probs = F.sigmoid(logits)
	        m1 = probs.view(num, -1)
	        m2 = targets.view(num, -1)
	        intersection = (m1 * m2)

	        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
	        score = 1 - score.sum() / num
	        return score
    
  
  > NOTE: The reason why intersection is implemented as a multiplication and the cardinality as `sum()` on axis 1 (each 3 channels sum) is because predictions and targets are one-hot encoded vectors

## Results
|      Loss     | Validation Scores | Validation Scores | Test Scores | Test Scores |
|:------------:|:------------:|-------------------|-------------------|-------------|-------------|
|              |     mIoU   |    mDice     |   mIoU    |    mDice    |
|        BCE     |       0.9403      |       0.9692      |      -      |      -      |
|        BCE+DCL   |       0.9426      |       0.9704      |      -      |      -      |
|     BCE+DCL+IDCL |       0.9665      |       0.9829      |    0.9295   |    0.9623   |
The results with this network are good, and the some of the best ones are shown here

Input Image
![prediction](https://imgur.com/MwOoEno.png)

Correspoding Segmented Image
![Segmented Image](https://i.imgur.com/ak3Aa2M.png)

<!--On the testing set we got the Dice score of
> 0.95
>-->



# References


[[1]](https://arxiv.org/abs/1801.05746) V. Iglovikov and A. Shvets   - TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation, 2018 
- The code for our uses was taken from 
	> https://github.com/ternaus/TernausNet
	
[[2]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. ”Imagenet classification with deep convolutional neural networks.” In Advances in neural information processing systems, pp. 1097-1105. 2012

[[3]](https://arxiv.org/abs/1409.4842) Szegedy, Christian, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. ”Going deeper with convolutions.” In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1-9. 2015

[[4]](https://arxiv.org/abs/1502.03167) Ioffe, Sergey, and Christian Szegedy. ”Batch normalization: Accelerating deep network training by reducing internal covariate shift.” In International Conference on Machine Learning, pp. 448-456. 2015

[[5]](http://jmlr.org/papers/v15/srivastava14a.html) Srivastava, Nitish, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. ”Dropout: a simple way to prevent neural networks from overfitting.”Journal of machine learning research 15, no. 1 (2014): 1929-1958

[[6]](https://arxiv.org/abs/1409.1556) Simonyan, Karen, and Andrew Zisserman. ”Very deep convolutional networks

[7] Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006

[[8]](http://www.deeplearningbook.org) Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press,2016

[[9]](https://www.tensorflow.org/) TensorFlow

[[10]](https://keras.io/) Keras

[[11]]([https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html))  Pytorch

Return to main [README](www.github.com/medal-iitb/LungSegmentation/README.md) .
