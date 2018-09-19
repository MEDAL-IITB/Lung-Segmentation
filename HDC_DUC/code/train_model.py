import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from build_model import ResNetwithHDCDUC
import csv

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
    
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

class InverseSoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(InverseSoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = 1-logits.view(-1)
        tflat = 1-targets.view(-1)
        intersection = (iflat * tflat).sum()
    
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))
    
img_size = (256,256)    
transformations_train = transforms.Compose([transforms.Resize(img_size),
                                      transforms.RandomRotation(10),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
    
transformations_test = transforms.Compose([transforms.Resize(img_size),
                                      transforms.ToTensor()])     
                                      
    
  
from data_loader import LungSeg
from data_loader import LungSegTest
train_set = LungSeg(transforms = transformations_train)  
test_set = LungSegTest(transforms = transformations_test)  
batch_size = 5
num_epochs = 100
    
class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count
loss_list = []
vloss_list = []

def train():
    cuda = torch.cuda.is_available()
    net = ResNetwithHDCDUC(1)
    if cuda:
        net = net.cuda()
    criterion1 = nn.BCEWithLogitsLoss().cuda()
    criterion2 = SoftDiceLoss().cuda()
    criterion3 = InverseSoftDiceLoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    print("preparing training data ...")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print("done ...")

    test_set = LungSegTest(transforms = transformations_test) 
    test_loader = DataLoader(test_set, batch_size=batch_size)
    for epoch in range(num_epochs):
        train_loss = Average()
        net.train()

        if epoch<20:
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        elif 20<=epoch<30:
            optimizer = torch.optim.Adam(net.parameters(), lr=3.33e-5)
        elif 30<=epoch<40:
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
        elif 40<=epoch<100:
            optimizer = torch.optim.Adam(net.parameters(), lr=3.33e-6)
        elif 150<=epoch<201:
            optimizer = torch.optim.Adam(net.parameters(), lr=1.e-6)    

            
        for i, (images, masks) in enumerate(train_loader):
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            loss = (criterion2(outputs, masks) + criterion1(outputs, masks) + 1.5*criterion3(outputs, masks))/3.5
            #loss_list.append(loss)

            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), images.size(0))

        val_loss = Average()
        net.eval()
        for images, masks in test_loader:
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            outputs = net(images)
            vloss = criterion2(outputs, masks)
            #vloss_list.append(vloss) 
            val_loss.update(vloss.item(), images.size(0))

        print("Epoch {}, Loss: {}, Validation Loss: {}".format(epoch+1, train_loss.avg, val_loss.avg))
        loss_list.append(train_loss.avg)
        vloss_list.append(val_loss.avg)

        with open('Log.csv', 'a') as logFile:
            FileWriter = csv.writer(logFile)
            FileWriter.writerow([epoch+1, train_loss.avg, val_loss.avg])
        torch.save(net.state_dict(), 'Weights/cp_{}_{}.pth.tar'.format(epoch+1, val_loss.avg))    

    
    torch.save(net.state_dict(), 'cp.pth')
    """
    key = list(range(2))
    plt.plot(key, loss_list, 'r')
    plt.plot(key, vloss_list, 'b')
    plt.show()      
    """        
    return net, loss_list, vloss_list

def test(model):
    model.eval()



if __name__ == "__main__":
    train()
    #print("done!")
    key = list(range(100))
    plt.plot(key, loss_list, 'r')
    plt.plot(key, vloss_list, 'b')
    plt.show()
    plt.savefig('Losses plot')
    
"""
        alpha0 = 1e-5
        k = 9.0/num_epochs

        optimizer = torch.optim.Adam(net.parameters(), lr=alpha0/(1+k*epoch))
"""
