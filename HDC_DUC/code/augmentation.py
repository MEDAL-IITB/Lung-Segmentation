import numpy as np
import random
import cv2
from tqdm import tqdm
import os
import sys
from PIL import Image

def sp_noise(image,var):
    row,col,ch= image.shape
    mean = 0
    #var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy
'''
    output = np.zeros(image.size,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
'''


# allimages = []
# maskimages = []


image_names_image_test = os.listdir('Image/Test/Images')
image_names_image_train = os.listdir('Image/Train/Images')
image_names_manualmask_left = os.listdir('ManualMask/leftMask')
image_names_manualmask_right = os.listdir('ManualMask/rightMask')
image_names_mask_test = os.listdir('Mask/Test/Images')
image_names_mask_train = os.listdir('Mask/Train/Images')

for images in tqdm(image_names_image_test):
    #print(images)
    im = cv2.imread('Image/Test/Images/'+images) # Only for grayscale image
    #print(type(im))
    noise_img = sp_noise(im,0.1)
    cv2.imwrite('Image/Test/Images/'+images.split('.')[0] + '_gauss.png', noise_img)

for images in tqdm(image_names_image_train):
    #print(images)
    im = cv2.imread('Image/Train/Images/'+images) # Only for grayscale image
    #print(type(im))
    noise_img = sp_noise(im,0.1)
    cv2.imwrite('Image/Train/Images/'+images.split('.')[0] + '_gauss.png', noise_img)

for images in tqdm(image_names_manualmask_left):
    #print(images)
    im = cv2.imread('ManualMask/leftMask/'+images) # Only for grayscale image
    #print(type(im))
    noise_img = sp_noise(im,0.0)
    cv2.imwrite('ManualMask/leftMask/'+images.split('.')[0] + '_gauss.png', noise_img)

for images in tqdm(image_names_manualmask_right):
    #print(images)
    im = cv2.imread('ManualMask/rightMask/'+images) # Only for grayscale image
    #print(type(im))
    noise_img = sp_noise(im,0.0)
    cv2.imwrite('ManualMask/rightMask/'+images.split('.')[0] + '_gauss.png', noise_img)

for images in tqdm(image_names_mask_test):
    #print(images)
    im = cv2.imread('Mask/Test/Images/'+images) # Only for grayscale image
    #print(type(im))
    noise_img = sp_noise(im,0.0)
    cv2.imwrite('Mask/Test/Images/'+images.split('.')[0] + '_gauss.png', noise_img)

for images in tqdm(image_names_mask_train):
    #print(images)
    im = cv2.imread('Mask/Train/Images/'+images) # Only for grayscale image
    #print(type(im))
    noise_img = sp_noise(im,0.0)
    cv2.imwrite('Mask/Train/Images/'+images.split('.')[0] + '_gauss.png', noise_img)
