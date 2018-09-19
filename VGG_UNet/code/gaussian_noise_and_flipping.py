import numpy as np
import os
import cv2
from skimage import io
from skimage.util import random_noise
from skimage.filters import gaussian
from tqdm import tqdm

image_names  = os.listdir("Images_padded1/Train/")

for name in tqdm(image_names):
    img = io.imread('Images_padded1/Train/'+name)
    mask = io.imread('Mask_padded1/Train/'+name)

    noise_image_01 = random_noise(img, mode='gaussian', seed=None, clip=True, var = 0.01)
    noise_image_01 = noise_image_01*255.
    noise_image_01 = noise_image_01.astype('uint8')
    io.imsave('Images_padded1/Train/'+name[:-4]+'_01.png', noise_image_01)
    io.imsave('Mask_padded1/Train/'+name[:-4]+'_01.png', mask)

'''
image_names  = os.listdir("Images_padded1/Train/")

for name in tqdm(image_names): 
	img = cv2.imread('Images_padded1/Train/'+name,0)
	mask = cv2.imread('Mask_padded1/Train/'+name,0)
 
# copy image to display all 4 variations
	horizontal_img = img.copy()
	horizontal_mask = mask.copy()

 
# flip img horizontally, vertically,
# and both axes with flip()
	horizontal_img = cv2.flip( img, 1 )
	horizontal_mask = cv2.flip(mask, 1)

 
# display the images on screen with imshow()
	cv2.imwrite( 'Images_padded1/Train/'+name[:-4]+'_flip.png', horizontal_img )
	cv2.imwrite( 'Mask_padded1/Train/'+name[:-4]+'_flip.png', horizontal_mask )

'''
'''
img = io.imread('MCUCXR_0006_0.png')


noise_image_1 = random_noise(img, mode='gaussian', seed=None, clip=True, var = 0.001)
noise_image_1 = noise_image_1*255.
noise_image_1 = noise_image_1.astype('uint8')

io.imsave('MCUCXR_0006_0_noise_1.png', noise_image_1)
'''
'''
blur  = gaussian(img, sigma=5, output=None, mode='nearest', cval=0, multichannel=None, preserve_range=False, truncate=4.0)
blur = blur*255.
blur = blur.astype('uint16')
io.imsave('MCUCXR_0006_0_vlur.png', blur)
'''
