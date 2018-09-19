
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from build_model import *
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#--------------------------
class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    #property
    def avg(self):
        return self.sum / self.count
#------------------------------        
# import csv
writer = SummaryWriter()
#----------------------------------------
class SoftDiceLoss(nn.Module):
    '''
    Soft Dice Loss
    '''        
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))
#-------------------------------------------------
'''
class SoftDicescore(nn.Module):
'''
    #Soft Dice Loss
'''        
    def __init__(self, weight=None, size_average=True):
        super(SoftDicescore, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))
'''
#-------------------------------------------------        
'''
class W_bce(nn.Module):
    
    #weighted crossentropy per image
    
    def __init__(self, weight=None, size_average=True):
        super(W_bce, self).__init__()
        
    def forward(self, logits, targets):
        eps = 1e-6
        total_size = targets.view(-1).size()[0]
        #print "total_size", total_size
        ones_size = torch.sum(targets.view(-1,1)).item()
        #print "one_size", ones_size
        zero_size = total_size - ones_size
        #print "zero_size", zero_size
        #assert total_size == (ones_size + zero_size)
        #print "crossed assertion"
        loss_1 = torch.mean(-(targets.view(-1)* ( total_size/ones_size) * torch.log(torch.clamp(F.sigmoid(logits).view(-1),eps,1.-eps))))#.sum(axis=1)
        #print "crossed loss1"
        loss_0 = torch.mean(-((1.-targets.view(-1))* ( total_size/zero_size) * torch.log((1.-torch.clamp(F.sigmoid(logits).view(-1),eps,1.-eps)))))#.sum(axis=1)
        #print "crossed loss0"
        return loss_1 + loss_0
'''        
#----------------------------------                
class InvSoftDiceLoss(nn.Module):

    '''
    Inverted Soft Dice Loss
    '''   
    def __init__(self, weight=None, size_average=True):
        super(InvSoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = 1-logits.view(-1)
        tflat = 1-targets.view(-1)
        intersection = (iflat * tflat).sum()
    
    
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))
#--------------------------------------    
'''
class InvSoftDicescore(nn.Module):

'''
    #Inverted Soft Dice Loss
'''
       
    def __init__(self, weight=None, size_average=True):
        super(InvSoftDicescore, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = 1-logits.view(-1)
        tflat = 1-targets.view(-1)
        intersection = (iflat * tflat).sum()    
        return ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))
'''
#----------------------------------------
'''
class int_custom_loss(nn.Module):
'''
    #custom loss
'''
    def __init__(self, weight=None, size_average=True):
        super(int_custom_loss, self).__init__()
        
    def forward(self, logits, targets):
        loss_inv_dice = InvSoftDicescore()
        loss_dice = SoftDicescore()
        total_size = targets.view(-1).size()[0]
        ones_size = torch.sum(targets.view(-1,1)).item()
        th = 0.2 * total_size
        if(ones_size > th):
            return (- 0.8*torch.log(loss_dice(logits,targets))-0.2*torch.log(loss_inv_dice(logits, targets)))
        else:
            return(-0.2*torch.log(loss_dice(logits, targets))-0.8*torch.log(loss_inv_dice(logits, targets)))
'''
'''
class weighted_dice_invdice(nn.Module):
'''
    #custom loss
'''
    def __init__(self, weight=None, size_average=True):
        super(weighted_dice_invdice, self).__init__()
        
    def forward(self, logits, targets):
        loss_inv_dice = InvSoftDicescore()
        loss_dice = SoftDicescore()
        total_size = targets.view(-1).size()[0]
        ones_size = torch.sum(targets.view(-1,1)).item()
        zero_size = total_size - ones_size
        th = 0.2 * total_size
        return (-(zero_size/total_size)*torch.log(loss_dice(logits,targets))-(ones_size/total_size)*torch.log(loss_inv_dice(logits, targets)))
'''

#Tranformations------------------------------------------------
transformations_train = transforms.Compose([transforms.Resize((864,864)),transforms.ToTensor()])
    
transformations_val = transforms.Compose([transforms.Resize((864,864)),transforms.ToTensor()])     
#-------------------------------------------------------------                                      
      
from data_loader import LungSegTrain
from data_loader import LungSegVal
train_set = LungSegTrain(transforms = transformations_train)  
batch_size = 1   
num_epochs = 75
    
def train():
    cuda = torch.cuda.is_available()
    net = SegNet(3,1)
    if cuda:
        net = net.cuda()
    #net.load_state_dict(torch.load('Weights_BCE_Dice/cp_bce_lr_05_100_0.222594484687.pth.tar'))
    criterion1 = nn.BCEWithLogitsLoss().cuda()
    criterion2 = SoftDiceLoss().cuda()
    criterion3 = InvSoftDiceLoss().cuda()
    #criterion4 = W_bce().cuda()
    #criterion5 = int_custom_loss()
    #criterion6 = weighted_dice_invdice()     
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    #scheduler = MultiStepLR(optimizer, milestones=[2,10,75,100], gamma=0.1)
    
    print("preparing training data ...")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print("done ...")    
    val_set = LungSegVal(transforms = transformations_val)   
    val_loader = DataLoader(val_set, batch_size=batch_size,shuffle=False)
    for epoch in tqdm(range(num_epochs)):
        #scheduler.step()        
        train_loss = Average()
        net.train()
        for i, (images, masks) in tqdm(enumerate(train_loader)):
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            #writer.add_image('Training Input',images)
            #writer.add_image('Training Pred',F.sigmoid(outputs)>0.5)
            c1 = criterion1(outputs,masks) + criterion2(outputs, masks) + criterion3(outputs, masks)
            loss = c1
            writer.add_scalar('Train Loss',loss,epoch)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), images.size(0))
            for param_group in optimizer.param_groups:
                writer.add_scalar('Learning Rate',param_group['lr'])
        val_loss1 = Average()
        val_loss2 = Average()
        val_loss3 = Average()
        net.eval()
        for images, masks,_ in tqdm(val_loader):
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            outputs = net(images)
            if (epoch)%10==0:
                writer.add_image('Validation Input',images,epoch)
                writer.add_image('Validation GT ',masks,epoch)
                writer.add_image('Validation Pred0.5',F.sigmoid(outputs)>0.5,epoch)
                writer.add_image('Validation Pred0.3',F.sigmoid(outputs)>0.3,epoch)
                writer.add_image('Validation Pred0.65',F.sigmoid(outputs)>0.65,epoch)
            
            vloss1 = criterion1(outputs, masks)
            vloss2 = criterion2(outputs, masks)  
            vloss3 = criterion3(outputs, masks) #+ criterion2(outputs, masks)
            #vloss = vloss2 + vloss3
            writer.add_scalar('Validation loss(BCE)',vloss1,epoch)
            writer.add_scalar('Validation loss(Dice)',vloss2,epoch)
            writer.add_scalar('Validation loss(InvDice)',vloss3,epoch)

            val_loss1.update(vloss1.item(), images.size(0))
            val_loss2.update(vloss2.item(), images.size(0))
            val_loss3.update(vloss3.item(), images.size(0))

        print("Epoch {}, Training Loss(BCE+Dice): {}, Validation Loss(BCE): {}, Validation Loss(Dice): {}, Validation Loss(InvDice): {}".format(epoch+1, train_loss.avg(), val_loss1.avg(), val_loss2.avg(), val_loss3.avg()))

        # with open('Log.csv', 'a') as logFile:
        #     FileWriter = csv.writer(logFile)
        #     FileWriter.writerow([epoch+1, train_loss.avg, val_loss1.avg, val_loss2.avg, val_loss3.avg])        
        	
        torch.save(net.state_dict(), 'Weights_BCE_Dice_InvDice/cp_bce_flip_lr_04_no_rot{}_{}.pth.tar'.format(epoch+1, val_loss2.avg()))                
    return net

def test(model):
    model.eval()



if __name__ == "__main__":
    train()
