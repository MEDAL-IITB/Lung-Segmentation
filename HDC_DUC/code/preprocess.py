from skimage import io
import os
from tqdm import tqdm


image_list = os.listdir('test_image')

for images in tqdm(image_list):
	left_mask = io.imread('test_mask/left/'+images)
	right_mask = io.imread('test_mask/right/'+images)
	mask = left_mask + right_mask
	io.imsave('test_mask/'+images, mask)


