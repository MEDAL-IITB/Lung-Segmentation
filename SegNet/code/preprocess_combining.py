import os
import numpy as np
from skimage import io, exposure



def make_masks():
    path = 'test_image/'
    for i, filename in enumerate(os.listdir(path)):
        left = io.imread('test_mask/left/' + filename[:-4] + '.png')
        right = io.imread('test_mask/right/' + filename[:-4] + '.png')
        io.imsave('test_mask/Mask/' + filename[:-4] + '.png', np.clip(left + right, 0, 255))
        print ('Mask', i, filename)

make_masks()
    
