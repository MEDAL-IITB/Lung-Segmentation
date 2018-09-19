# Introduction
This report describes the usage of **SegNet** and **U-Net** architechtures for **medical image segmentation**.

We divide the article into the following parts

 - [Dataset](#dataset)
 - [SegNet](#segnet)
 - [U-Net](#u-net)
 - [Loss Function Used](#loss-function-used)
 - Results
 -   References
 - Further Help

# Dataset
## Montgomory Dataset

  

The dataset contains Chest X-Ray images. We use this dataset to perform a lung segmentation. 
>The dataset can be found [here](http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip)

  

Structure:

We make the following structure of the given data set:

![](https://lh4.googleusercontent.com/J-kHm2BX9ywKISMuY_BaCaFf--UuPJOKlFYLO89gYgvjmqlM9RrFive2wOU30X8N7bzI03uwMCtnb_oCHDPaobyxTMEFlfsTSNXALS629uuAkSUZfm9y-lUv5FORquPe1P8CPp4p)

## Data Preprocessing
![](https://lh4.googleusercontent.com/TTLhU_8UxfxPWPURwLsqNbJu09EfPTReyCXHH9mX7saLzfK6aLgxK_NQd1VNeL7u1acwVnppg2pOZeLO9S4hoxpxjRSoXUHRlK8OAo6peOHpvv_zzTv2g43Wy4HMmk_i-aoATdEG)
Figure

The Montgomery dataset contains images from the Department of Health and Human Services, Montgomery County, Maryland, USA. The dataset consists of

138 CXRs, including 80 normal patients and 58 patients with manifested tuberculosis (TB). The CXR images are 12-bit gray-scale images of dimension 4020 × 4892 or 4892 × 4020 . Only the two lung masks annotations are available which were combined to a single image in order to make it easy for the network to learn the task of segmentation (Fig 1).To make all images of symmetric dimensions we padded the pictures to the maximum dimension in their height or width such that images are of 4892 x 4892, this is done to preserve the aspect ratio of CXR while resizing. We scale all images to 1024 x 1024 pixels, which retains sufficient visual details for vascular structures in the lung fields and this could be the maximum size that could be accommodated in, along with U-Net in Graphics Processing Unit (GPU). We scaled all pixel values to 0-1 . Data augmentation was applied by flipping around the vertical axis and adding gaussian noise with mean 0 and a variance of 0.01. Also rotation about the centre to subtle angles of 5-10 degrees during runtime were performed to make the model more robust.
  

# SegNet

### Introduction
![SegNet](http://mi.eng.cam.ac.uk/projects/segnet/images/segnet.png)

SegNet has an encoder network and a corresponding decoder network, followed by a final pixelwise classification layer. This architecture is illustrated in the above figure. The encoder network consists of 13 convolutional layers which correspond to the first 13 convolutional layers in the VGG16 network designed for object classification. We can therefore initialize the training process from weights trained for classification on large datasets. We can also discard the fully connected layers in favour of retaining higher resolution feature maps at the deepest encoder output. This also reduces the number of parameters in the SegNet encoder network significantly (from 134M to 14.7M) as compared to other recent architectures. Each encoder layer has a corresponding decoder layer and hence the decoder network has 13 layers. The final decoder output is fed to a multi-class soft-max classifier or for a binary classification task, to a sigmoid activation function to produce class probabilities for each pixel independently. Each encoder in the encoder network performs convolution with a filter bank to produce a set of feature maps. These are then batch normalized. Then an element-wise rectified- linear non-linearity (ReLU) max (0, x) is applied. Following that, max-pooling with a 2 × 2 window and stride 2 (non-overlapping window) is performed and the resulting output is sub-sampled by a factor of 2. Max-pooling is used to achieve translation invariance over small spatial shifts in the input image. Sub-sampling results in a large input image context (spatial window) for each pixel in the feature map. While several layers of max-pooling and sub-sampling can achieve more translation invariance for robust classification correspondingly there is a loss of spatial resolution of the feature maps. The increasingly lossy (boundary detail) image representation is not beneficial for segmentation where boundary delineation is vital.


# Loss Function Used 
We use two loss functions here, viz. `Binary Cross Entropy` and `Dice loss`

#### Binary Cross Entropy Loss
Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of `.012` when the actual observation label is `1` would be bad and result in a **high loss** value. A perfect model would have a log loss of `0`.

It is defined mathematically as
In binary classification, where the number of classes  $M$  equals 2, cross-entropy can be calculated as:

$$
-\frac{1}{N}\sum_{i=1}^N(y_{i}\log(p_{i}) + (1-y_{i})\log(1-p_{i}))
$$

#### Dice Coefficient Loss
The dice coefficient loss is used to measure the `intersection over union` of the output and target image. 

Mathematically, Dice Score is 
$$\frac{2 |P \cap R|}{|P| + |R|}$$

and the corresponding loss is
$$1-\frac{2 |P\cap R|}{|P| + |R|}$$

$$1- \frac{2\sum_{i=0}^Np_{i}r_{i}+\epsilon}{\sum_{i=0}^Np_{i}+ \sum_{i=0}^Nr_{i}+\epsilon}\quad p_{i}\space\epsilon\space P,\space r_{i}\space\epsilon\space R$$

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
   
 #### Inverted Dice Coefficient Loss
 The formula below calculates the measure of overlap after inverting the image or in this case taking the complement.


Mathematically, Inverted Dice Score is
$$\frac{2|\overline{P}\cap\overline{R}|}{|\overline{P}| +|\overline{R}| }$$
and the corresponding loss is
$$1-\frac{2|\overline{P}\cap\overline{R}|}{|\overline{P}| +|\overline{R}| }$$
$$1- \frac{2\sum_{i=0}^N(1-p_{i})(1-r_{i})+\epsilon}{\sum_{i=0}^N(1-p_{i})+ \sum_{i=0}^N(1-r_{i})+\epsilon}\quad p_{i}\space\epsilon\space P,\space r_{i}\space\epsilon\space R$$


	class SoftInvDiceLoss(nn.Module):
	    def __init__(self, weight=None, size_average=True):
	        super(SoftDiceLoss, self).__init__()

	    def forward(self, logits, targets):
	        smooth = 1
	        num = targets.size(0)
	        probs = F.sigmoid(logits)
	        m1 = probs.view(num, -1)
	        m2 = targets.view(num, -1)
	        m1, m2 = 1.-m1, 1.-m2
	        intersection = (m1 * m2)

	        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
	        score = 1 - score.sum() / num
	        return score


  > NOTE: The reason why intersection is implemented as a multiplication and the cardinality as `sum()` on axis 1 (each 3 channels sum) is because predictions and targets are one-hot encoded vectors

# Results

|     Loss     | Validation Scores | Validation Scores | Test Scores | Test Scores |
|:------------:|:------------:|-------------------|-------------------|-------------|-------------|
|              |     mIoU   |    mDice     |   mIoU    |    mDice    |     mDice       |     mIoU    |    mDice    |
|        BCE     |       0.8867      |       0.9396      |      -      |      -      |
|      BCE+DCL   |       0.9011      |       0.9477      |      -      |      -      |
|    BCE+DCL+IDCL |       0.9234      |       0.9600      |    0.8731   |    0.9293   |

The results with this network are good, and the some of the best ones are shown here
![
](https://lh3.googleusercontent.com/IYCDXq8yFP5drY8miu-IYgBUE9HfKWDTJxLxBlPdYXPCQtF4Gwrhk-EnKoRftNpE-Z4paZ90VFBA "results")


# References
[[1]](https://arxiv.org/abs/1511.00561) Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. "Segnet: A deep convolutional encoder-decoder architecture for image segmentation". CoRR, abs/1511.00561, 2015

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
