import numpy as np
import random
import cv2
from tqdm import tqdm
import os
import sys
from PIL import Image

image_names_image_test = os.listdir('Image/Test/Images')
image_names_image_train = os.listdir('Image/Train/Images')
image_names_manualmask_left = os.listdir('ManualMask/leftMask')
image_names_manualmask_right = os.listdir('ManualMask/rightMask')
image_names_mask_test = os.listdir('Mask/Test/Images')
image_names_mask_train = os.listdir('Mask/Train/Images')

# maskimages.append(image_names_manualmask_right)
# maskimages.append(image_names_manualmask_left)
# maskimages.append(image_names_mask_test)
# maskimages.append(image_names_mask_train)

# allimages.append(image_names_image_train)
# allimages.append(image_names_image_test)

for images in tqdm(image_names_image_train):
    img = cv2.imread('Image/Train/Images/'+images) # Only for grayscale image
    horizontal_img = img.copy()
    vertical_img = img.copy()
    horizontal_img = cv2.flip( img, 0 )
    vertical_img = cv2.flip( img, 1 )
    cv2.imwrite('Image/Train/Images/'+images.split('.')[0] + '_horizontal.png', horizontal_img)
    cv2.imwrite('Image/Train/Images/'+images.split('.')[0] + '_vertical.png', vertical_img)

for images in tqdm(image_names_image_test):
    img = cv2.imread('Image/Test/Images/'+images) # Only for grayscale image
    horizontal_img = img.copy()
    vertical_img = img.copy()
    horizontal_img = cv2.flip( img, 0 )
    vertical_img = cv2.flip( img, 1 )
    cv2.imwrite('Image/Test/Images/'+images.split('.')[0] + '_horizontal.png', horizontal_img)
    cv2.imwrite('Image/Test/Images/'+images.split('.')[0] + '_vertical.png', vertical_img)

for images in tqdm(image_names_manualmask_left):
    img = cv2.imread('ManualMask/leftMask/'+images) # Only for grayscale image
    horizontal_img = img.copy()
    vertical_img = img.copy()
    horizontal_img = cv2.flip( img, 0 )
    vertical_img = cv2.flip( img, 1 )
    cv2.imwrite('ManualMask/leftMask/'+images.split('.')[0] + '_horizontal.png', horizontal_img)
    cv2.imwrite('ManualMask/leftMask/'+images.split('.')[0] + '_vertical.png', vertical_img)

for images in tqdm(image_names_manualmask_right):
    img = cv2.imread('ManualMask/rightMask/'+images) # Only for grayscale image
    horizontal_img = img.copy()
    vertical_img = img.copy()
    horizontal_img = cv2.flip( img, 0 )
    vertical_img = cv2.flip( img, 1 )
    cv2.imwrite('ManualMask/rightMask/'+images.split('.')[0] + '_horizontal.png', horizontal_img)
    cv2.imwrite('ManualMask/rightMask/'+images.split('.')[0] + '_vertical.png', vertical_img)

for images in tqdm(image_names_mask_test):
    img = cv2.imread('Mask/Test/Images/'+images) # Only for grayscale image
    horizontal_img = img.copy()
    vertical_img = img.copy()
    horizontal_img = cv2.flip( img, 0 )
    vertical_img = cv2.flip( img, 1 )
    cv2.imwrite('Mask/Test/Images/'+images.split('.')[0] + '_horizontal.png', horizontal_img)
    cv2.imwrite('Mask/Test/Images/'+images.split('.')[0] + '_vertical.png', vertical_img)

for images in tqdm(image_names_mask_train):
    img = cv2.imread('Mask/Train/Images/'+images) # Only for grayscale image
    horizontal_img = img.copy()
    vertical_img = img.copy()
    horizontal_img = cv2.flip( img, 0 )
    vertical_img = cv2.flip( img, 1 )
    cv2.imwrite('Mask/Train/Images/'+images.split('.')[0] + '_horizontal.png', horizontal_img)
    cv2.imwrite('Mask/Train/Images/'+images.split('.')[0] + '_vertical.png', vertical_img)
