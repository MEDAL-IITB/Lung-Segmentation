># Global Convolutional Network
*   [Introduction](#introduction)
*   [Dataset](#dataset)
    *   [Montgomery Dataset](#montgomery-dataset)
    *   [Data Preprocessing](#data-preprocessing)
*   [Architecture](#architecture)
* [Training](#training)
* * [Loss Function](#loss-function)
* * [Evaluation Metrics](#evaluation-metrics)
* [Results](#results)
*   [References](#references)

## Introduction
GCN Architecture is proposed in the paper "Large Kernel Matters ——
Improve Semantic Segmentation by Global Convolutional Network"[$^{[1]}$](https://arxiv.org/abs/1703.02719)

A GCN based architecture, called ResNet-GCN, is used for the purposes of lung segmentation from chest x-rays.

## Dataset
 ### Montgomery Dataset
  This architecture is proposed to segment out lungs from a chest radiograph (colloquially know as chest X-Ray, CXR).  The dataset is known as the [Montgomery County X-Ray Set](https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/), which contains 138 posterior-anterior x-rays. The motivation being that this information can be further used to detect chest abnormalities like shrunken lungs or other structural deformities. This is especially useful in detecting tuberculosis in patients.
  
### Data Preprocessing
The x-rays are 4892x4020 pixels big. Due to GPU memory limitations, they are resized to 1024x1024.

The dataset is augmented by randomly rotating and flipping the images.
 
## Architecture
 
### Intuition
 
The Global Convolution Network or GCN is an architecture proposed for the task of segmenting images. An image segmenter has to perform 2 tasks: classification as well as localization. This has an inherent challenge as both tasks have inherent diametrically opposite demands.
While a classifier has to be transformation and rotation invariant, a localizer has to sensitive to the same. The GCN architecture finds a balance of the two demands with the following properties:

 1. To retain spatial information, no fully connected layers are used and a FCN framework is adopted
 2. For better classification, a large kernel size is adopted to enable dense connections in feature maps
 
For segmentation to have semantic context, local context obtained from simple CNN architectures is not sufficient; a bigger view (i.e. global context) is critical. 
This architecture, coined ResNet-GCN, is basically a modified ResNet model with additional GCN blocks obtaining the required global view and the Boundary Refinement Blocks further improving the segmentation performance near object boundaries.

The entire pipeline of this architecture is visualized below: 

![enter image description here](https://lh3.googleusercontent.com/jma3XKGwaLnS4-0TYajAsD8gNYg0_uJ0W81Xj5ssOjub3DdEhkjxhrcUEAoTJEyZ6_l7VBCmPybM "ResNet-GCN Pipeline")
### GCN Block
  The GCN Block is essentially a kx1 followed by 1xk convolution summed with a parallely computed 1xk followed by kx1 convolution.  This results in a large kxk kernel with dense connections.
  NOTE: the blocks are acting on feature maps and so channel width is larger than 3
  
### Boundary Refinement Block 
  
  The BR block improves the segmentation near the boundaries of objects, where segmentation is less like a pure classification problem. It's design is inspired by that of ResNets and is basically a parallel branch of Conv+ReLU, followed by another conv. layer added to the input.
  
![enter image description here](https://lh3.googleusercontent.com/b-WoF5ESCbTOWeR1mvHd6LTd-I0HAZ1V2pFX1E1NnSnTZhPb_eDnHevCPnUwTCb3aH6ituCTFz-_ "GCN and BR Block") 

## Training 

A pretrained ResNet-50 [$^{[2]}$](https://arxiv.org/abs/1512.03385) is used and is later fine-tuned. The rationale being that while medical images are vastly different from natural images, the ResNet is a good feature extractor (eg. edges, blobs, etc.) It is further augmented by the fact that many components in a medical image have features that resemble that of natural images eg. nuclei looks similar to balls.
 
Refer to http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006 for the ResNet50 Architecture and https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py for the torchvision.model code

57 CXRs, with their corresponding masks, were used to train the model while 20 were used for validation purposes (hold-out cross validation). Another 61 images have been reserved as test set.

### Loss Function
 
A linear combination of Soft Dice Loss, Soft Inverse Dice Loss, and Binary Cross-Entropy Loss (with logits) is used to train the model end-to-end. The best performance was obtained by weighing the three criteria at 0.25:0.5:0.25 (respectively).

#### Binary Cross-Entropy Loss (with logits)

This is calculated by passing the output of the network through a sigmoid activation before applying cross-entropy loss.

> The sigmoid and cross entropy calculations  are done in one class to exploit the log-sum-exp trick for greater numerical stability (as compared to sequentially applying sigmoid activation and then using vanilla BCE).

<!---$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top$ --->

$$l_n = - w_n \left[ t_n \cdot \log \sigma(x_n) + (1 - t_n) \cdot \log (1 - \sigma(x_n)) \right],$$

$$ L(x,y) = \sum_{i=1}^{N}l_i$$

#### Soft Dice Loss
 
Dice Loss gives a measure of how accurate the overlap of the mask and ground truth is.
The Sørensen–Dice coefficient is calculated as: $\frac{2. X\cap Y}{|X| + |Y|} = \frac{2. TP}{2. TP + FP + FN}$  and the Dice Loss is simply 1 - Dice coeff.
For Soft version of the loss, the output of the network is passed through a sigmoid before standard dice loss is evaluated.

 #### Soft Inverse Dice Loss

Inverse Dice loss checks for how accurately the background is masked. This penalizes the excess areas in the predicted mask. It is found by inverting the output before using the soft dice loss. This is added to account for true-negatives in our prediction.
  
 ### Evaluation Metrics
 
 Three metrics were used to evaluate the trained models;
  - Intersection over Union (IoU)
 - Dice Index
 - Inverse Dice Index

  #### Intersection over Union (IoU)
   
   IoU measures the accuracy of the predicted mask. It rewards better overlap of the prediction with the ground truth.
  $$\text{IoU} =\frac{P\cap GT}{P\cup GT} = \frac{TP}{TP+FP+FN} $$ 
	

> P stands for Predicted Mask while GT is ground truth.
<!--> The $\epsilon$ is added to ensure bounded ratios -->

   
   #### Dice Index
   The Dice index (also known as Sørensen–Dice similarity coefficient) has been discussed earlier. 
   Like IoU, Dice Index gives a measure of accuracy.
   > While for a single inference, both Dice and IoU are functionally equialent, over an average both have different inferences. 
   > 
   > While the Dice score is a measure of the average performance of the model, the IoU score is harsher towards errors and is a measure of the worst performance of the model. [$^{[\dagger]}$](https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou)
  
   #### Inverse Dice Index
As mentioned before, the Inverse dice index is obtained by inverting the masks and ground truth before calculating their dice score. 
>Due to the relatively smaller area of lung compared to the background, Inverse Dice score is large for every model.

 ## Results 
After 35 epochs of training, with learning rate = $10^{-3}$, scheduled to decrease by a factor of 5 after $15^{th}$ and $30^{th}$ epoch.

The model performed as follows:

Mean IoU: `0.8313548250035635`  
Mean Dice: `0.9072525421846304`  
Mean Inv. Dice: `0.9705243499345569`

### Examples of output
![
](https://lh3.googleusercontent.com/r5vPdjpPavjGm8lHB5P_HbgrqKxqowjK4xe0Q56ooCbPTe0e2lxeHOT4v-lSCixl-RYYfzjO8b63 "gcn4")

![
](https://lh3.googleusercontent.com/o9cZnU9i9lNV2orFc6S-t_pz61Ga02bY11VgS1njUaYitMhYfeAFjE4XC0Wlk21Y08EY0JDYLM9b "gcn6")


*The red boundary denotes the ground truth while the blue shaded portion is the predicted mask*


### Observations

- The GCN architecture is comparatively lightweight (in terms of GPU consumption)
 - The GCN architecture performs remarkably well for the task of lung segmentation even with very little training.
- However, further work is required to achieve better performance. Type -1 error is particularly prevalent in the predictions

## References
[[1]](https://arxiv.org/abs/1703.02719) Chao Peng, Xiangyu Zhang, Gang Yu, Guiming Luo, and Jian Sun. Large kernel matters - improve semantic segmentation by global convolutional network. CoRR, abs/1703.02719, 2017.

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

