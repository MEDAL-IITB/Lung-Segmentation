from tqdm import tqdm
import numpy as np
import cv2
import os
import sys
from PIL import Image


        
        
def get_IoU(Gi,Si):
    #print(Gi.shape, Si.shape)
    intersect = 1.0*np.sum(np.logical_and(Gi,Si))
    union = 1.0*np.sum(np.logical_or(Gi,Si))
    return intersect/union
#check  cv2.connectedComponents and what it returns, alongwith channels first or last
def generate_list(G,S):
    G = G.astype('uint8')
    S = S.astype('uint8')
    #print(np.unique(G))
    gland_obj_cnt,gland_obj = cv2.connectedComponents(G,connectivity=8)
    seg_obj_cnt,seg_obj = cv2.connectedComponents(S,connectivity=8)
    gland_obj_list = []
    seg_obj_list = []
    for i in range(1,gland_obj_cnt):
      gland_obj_list.append( (gland_obj==(i)).astype('int32') )
    for i in range(1,seg_obj_cnt):
      seg_obj_list.append( (seg_obj==(i)).astype('int32') )
    gland_obj_list = np.array(gland_obj_list)
    seg_obj_list = np.array(seg_obj_list)
    return gland_obj_list,seg_obj_list

####Find why channel parameter was passed
def AGI_core(gland_obj_list,seg_obj_list,channel='last'):
    C = 0.0
    U = 0.0
        ##check below:
    '''
    Swapping is not required. 
    if(channel=='last'):
            # make channels first
        gland_obj_list = np.swapaxes( np.swapaxes( gland_obj_list , 0,2) , 1 , 2 )
        seg_obj_list = np.swapaxes( np.swapaxes( seg_obj_list , 0,2) , 1 , 2 )
    '''
    #print(gland_obj_list.shape)
    seg_nonused = np.ones(len(seg_obj_list))
    for gi in gland_obj_list:
        iou = np.multiply( [get_IoU(gi,si) for si in seg_obj_list] , seg_nonused )
        max_iou = np.max(iou)
        j = np.argmax(iou)
        C = C + np.sum(np.logical_and(gi,seg_obj_list[j]) )
        U = U + np.sum(np.logical_or(gi,seg_obj_list[j]) )
        seg_nonused[j] = 0
    for ind in range(len(seg_obj_list)):
        if((seg_nonused[ind])==1):
            U = U + np.sum(seg_obj_list[ind])
    return C*1./U

def Acc_Jacard_Index(G,S):
    gland_obj_list,seg_obj_list = generate_list(G, S)
    print("In AJI and length is:{}".format(len(gland_obj_list)))
    return AGI_core(gland_obj_list,seg_obj_list)
  
#-----------------------------------------------

def F1_core(gland_obj_list,seg_obj_list,channel='first'):
    TP,FP,FN = 0.0,0.0,0.0
    if(channel=='last'):
        # make channels first
        gland_obj_list = np.swapaxes( np.swapaxes( gland_obj_list , 0,2) , 1 , 2 )
        seg_obj_list = np.swapaxes( np.swapaxes( seg_obj_list , 0,2) , 1 , 2 )
    seg_nonused = np.ones(len(seg_obj_list))
    gland_unhit = np.ones(len(seg_obj_list))

    for ind in range(len(gland_obj_list)):
        gi = gland_obj_list[ind]
        overlap_s = np.multiply( np.sum( seg_obj_list*gi , axis=(1,2) ) , seg_nonused )
        max_ov = np.max(overlap_s)
        percent_overlap = max_ov/np.sum(gi)
        if percent_overlap>=0.01 :
            # hit
            TP = TP +1
            j = np.argmax(overlap_s)
            seg_nonused[j] = 0
        else:
            # unhit
            FN = FN + 1

    FP = np.sum(seg_nonused)
    F1_val = (2*TP)/(2*TP + FP + FN)
    return F1_val

def F1_score(G,S):
    #y_mask = np.asarray(G[:, :, :, 0]).astype('uint8')
    #print y_mask.shape
    #y_pred = np.asarray(S[:, :, :, 0]).astype('uint8')
    #print y_pred.shape
    #print type(y_pred)
    #y_dist = K.expand_dims(G[:, :, :, 1], axis=-1)
    gland_obj_list,seg_obj_list = generate_list(G, S)
    return F1_core(gland_obj_list,seg_obj_list)
  
def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    #print(y_true.dtype)
    #print(y_pred.dtype)
    y_true = np.squeeze(y_true)/255
    y_pred = np.squeeze(y_pred)/255
    y_true.astype('bool')
    y_pred.astype('bool')
    intersection = np.logical_and(y_true, y_pred).sum()
    return ((2. * intersection.sum()) + 1.) / (y_true.sum() + y_pred.sum() + 1.)
    
    
    
    
    

    
smooth = 1

image_names = os.listdir('results_scratch_custom99/')

mean_dice = []
mean_F1 = []
aggr_jacard = []


for images in tqdm(image_names):
    
    S = np.expand_dims(np.array(Image.open('results_scratch_custom99/'+images).convert('L')),axis=-1)
    G = np.expand_dims(np.array(Image.open('/home/sahyadri/Testing/Test_40_y_HE/'+images).convert('L')),axis=-1)
    #print S.shape
    #G.shape
    #print(Acc_Jacard_Index(G,S))
    #aggr_jacard.append(Acc_Jacard_Index(G,S))
    #mean_F1.append(F1_score(G,S))
    mean_dice.append(Dice(G, S))
    
        
print ('Mean_Dice = ', np.mean(np.array(mean_dice)))
#print ('Mean_F1 = ', np.mean(np.array(mean_F1)))
#print (len(aggr_jacard), aggr_jacard)
#print ('Mean_Aggr_Jacard = ', np.mean(np.array(aggr_jacard)))
    
f = open('lung.txt','w')
a = 'Mean Dice : {}'.format(np.mean(np.array(mean_dice)))+ '\n' + 'Mean F1 : {}'.format(np.mean(np.array(mean_F1)))+ '\n' + 'Mean Aggregate Jacard : {}'.format(np.mean(np.array(aggr_jacard)))+ '\n'

f.write(str(a))
f.close()
        
