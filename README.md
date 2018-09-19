># Lung Segmentation
>by [MeDAL - IIT Bombay](https://github.com/MEDAL-IITB)

*   [Introduction](#introduction)
*   [Dataset](#dataset)
    *   [Montgomory Dataset](#montgomory-dataset)
    *   [Data Preprocessing](#data-preprocessing)
*   [GCN](#global-convolutional-network)
*   [VGG Unet](#vgg-unet)
* [SegNet](#segnet)
* [HDC/DUC](#hybrid-dilated-convolution-and-dense-upsampling-convolution)
* [Results](##Results)
<!--*   [References](#references)-->

## Introduction

Chest X-ray (CXR) is one of the most commonly prescribed medical imaging procedures. Such large volume of CXR scans place significant workloads on radiologists and medical practitioners. 
Organ segmentation is a crucial step to obtain effective computer-aided detection on CXR.
Future applications include
1.  Abnormal shape/size of lungs
    -   cardiomegaly (enlargement of the heart), pneumothorax (lung collapse), pleural effusion, and emphysema

2.  An initial step (preprocessing) for deeper analysis - eg. tumor detection
    
In this work, we demonstrate the effectiveness of  Fully Convolution Networks (FCN) to segment lung fields in CXR images. 
FCN incorporates a critic network, consisting primarily of an encoder and a decoder network to impose segmentation to CXR. During training, the network learns to generate a mask which then can be used to segment the organ. Via supervised learning, the FCN learns the higher order structures and guides the segmentation model to achieve realistic segmentation outcomes
## Dataset 
  This architecture is proposed to segment out lungs from a chest radiograph (colloquially know as chest X-Ray, CXR).  The dataset is known as the [Montgomery County X-Ray Set](https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/), which contains 138 posterior-anterior x-rays. The motivation being that this information can be further used to detect chest abnormalities like shrunken lungs or other structural deformities. This is especially useful in detecting tuberculosis in patients.
  
### Data Preprocessing
The x-rays are 4892x4020 pixels big. Due to GPU memory limitations, they are resized to 1024x1024(gcn) or 256x256(others)

The dataset is augmented by randomly rotating and flipping the images, and adding Gaussian noise to the images.

## Flow Chart
![
](https://lh3.googleusercontent.com/4jhBbczKqk8j4k2NyvMzljuzpdZYUMqZHpiT4OSQ4F0Z_-yvZAfNCfC1ge6wvg-BI-MAwXGKQzjD "flowchart")

## Models

### Global Convolutional Network
For details, go [here](https://github.com/MEDAL-IITB/Lung-Segmentation/tree/master/GCN/) .

### VGG Unet
For details, go [here](https://github.com/MEDAL-IITB/Lung-Segmentation/tree/master/VGG_UNet/) .

### SegNet
For details, go [here](https://github.com/MEDAL-IITB/Lung-Segmentation/tree/master/SegNet/) .

### Hybrid Dilated Convolution and Dense Upsampling Convolution
For details, go [here](https://github.com/MEDAL-IITB/Lung-Segmentation/tree/master/HDC_DUC/) .

## Results
A few of the results of the various models have been displayed below. (Scores are mean scores)
| Model  | Dice Score     |   IoU     | 
| -----  | ---------------|-----------|
|VGG UNet|    0.9623      |   0.9295  |
|SegNet  |    0.9293      |   0.8731  |
|GCN     |    0.907       |   0.8314  |
|HDC/DUC |    0.8501      |   0.7462  |
 
**U-Net Result**
![](https://lh3.googleusercontent.com/ku0vzfGUgolooGUqcYm6haipYcm_QLA33aw-ywOatslqHRX2cbat54HQsCRyX-xDpy2zkX2DuVx4 "UNet Results")

**SegNet Result**
![enter image description here](https://lh3.googleusercontent.com/2SueAM5xuMZJ99UwSgW1-Ne4mRC9-WsXt7NyCZ0mMYh3wP9QlFPt_uFd80cIpqzmtBZEzXB5vGDu "SegNet")

**HDC/DUC Result**
![
](https://lh3.googleusercontent.com/emerB7tePI6Cw90KCaHhqtPj_26Uo7R1z2yafjwlNeKgfIk2m1saP9ybWm2ChB09LiyYOCXUY9a6 "hdc")

**GCN Result**
![
](https://lh3.googleusercontent.com/VxAr3JeDDNO1yocRDYmxwqcHdjCcg1lOZraIHz7XDSXy4YVU6U3TExnEdJeWdfAOEExQiWstoQh8 "gcn4")
