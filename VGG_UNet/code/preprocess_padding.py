
import os

from PIL import Image
import numpy as np
from tqdm import tqdm


img_names = os.listdir('test_image/')


for name in tqdm(img_names):
	
	img  = np.asarray(Image.open('test_image/'+name))
	max_dim_img = max(img.shape[0], img.shape[1])
	row_img  = max_dim_img - img.shape[0]
	cols_img = max_dim_img - img.shape[1]
	padded_img = np.pad(img, ((row_img//2,row_img//2), (cols_img//2,cols_img//2)), mode ='constant')
	img = Image.fromarray(padded_img)
	img.save('Images_padded_test/'+name)

	mask = np.asarray(Image.open('test_mask/Mask/'+name))
	max_dim_mask = max(mask.shape[0], mask.shape[1])
	row_mask = max_dim_mask- mask.shape[0]
	cols_mask = max_dim_mask - mask.shape[1]
	padded_mask = np.pad(mask, ((row_mask//2,row_mask//2), (cols_mask//2,cols_mask//2)), mode ='constant')
	mask = Image.fromarray(padded_mask)
	mask.save('Mask_padded_test/'+name)

