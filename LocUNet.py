from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from scipy import ndimage
import scipy.io
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os

from lib import loader, modules



simName = "DPM" # Options: DPM, ZSDPMtoIRT2, DPMtoIRT2, DPMcars, IRT2carsCDPM, IRT2carsCDPMtoIRT, from top to bottom in Table II of paper
  
inp = 16
Loc_train = loaderLocDenNoEpsCorCarsC.locDL(phase="train",dir_dataset="dataset/",cityMap="true",carsMap="false",simulation=simName,TxMaps="true")
Loc_val = loaderLocDenNoEpsCorCarsC.locDL(phase="val",dir_dataset="dataset/",cityMap="true",carsMap="false",simulation=simName,TxMaps="true")    
    
Type = "noEps",epsNormTrnc=0.3)

image_datasets = {
    'train': Loc_train, 'val': Loc_val
}

batch_size = 15
batch_sizeTest = 1

dataloaders = {
    'train': DataLoader(Loc_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True),
    'val': DataLoader(Loc_val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
}

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
model = modulesLargerVar.LocUNet(inputs=inp)
model.cuda()


import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from datetime import datetime
import copy
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn

def my_loss(output, target):
    loss = torch.sum((output - target)**2,1)
    loss = torch.sqrt(loss)
    loss = torch.mean(loss)
    return loss


def calc_loss_dense(pred, target, metrics):
    loss = my_loss(pred, target)# *256*256
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss



def print_metrics(metrics, epoch_samples, phase):
    outputs1 = []
    outputs2 = []
    for k in metrics.keys():
        outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    with open('resultsWCityWTxWCarsLr3LargerIRT2carsOnlineDSNoEpsCorVarCarsCDPM/Log.txt', 'a') as f:
        print("{}: {}".format(phase, ", ".join(outputs1)), file=f)

def train_model(model, optimizer, scheduler, num_epochs=50):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
#    print('selam1')
    for epoch in range(num_epochs):
        with open('resultsWCityWTxWCarsLr3LargerIRT2carsOnlineDSNoEpsCorVarCarsCDPM/Log.txt', 'a') as f:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), file=f)
            print('-' * 10, file=f)
        with open('resultsWCityWTxWCarsLr3LargerIRT2carsOnlineDSNoEpsCorVarCarsCDPM/OutputTrue.txt', 'a') as f:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), file=f)
            print('-' * 10, file=f)
        with open('resultsWCityWTxWCarsLr3LargerIRT2carsOnlineDSNoEpsCorVarCarsCDPM/TrainingLoss.txt', 'a') as f:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), file=f)
            print('-' * 10, file=f)    

        since = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    with open('resultsWCityWTxWCarsLr3LargerIRT2carsOnlineDSNoEpsCorVarCarsCDPM/Log.txt', 'a') as f:
                        print("learning rate", param_group['lr'], file=f)

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)

                    # zero the parameter gradients

                optimizer.zero_grad()

                    # forward
                    # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs1 = model(inputs)

                    loss = calc_loss_dense(outputs1.float(), targets.float(), metrics)
                   

                        # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # statistics
                epoch_samples += inputs.size(0)
            

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        with open('resultsWCityWTxWCarsLr3LargerIRT2carsOnlineDSNoEpsCorVarCarsCDPM/Log.txt', 'a') as f:
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=f)
            now = datetime.now()
            print("now =", now, file=f)
    with open('resultsWCityWTxWCarsLr3LargerIRT2carsOnlineDSNoEpsCorVarCarsCDPM/Log.txt', 'a') as f:
        print('Best val loss: {:4f}'.format(best_loss), file=f)
    
    #Save model from last epoch
    stringer = 'resultsWCityWTxWCarsLr3LargerIRT2carsOnlineDSNoEpsCorVarCarsCDPM/WCityWTx' + 'Epoch50.pt'

    torch.save(model.state_dict(), stringer)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[10]:

# #load the saved model from first epoch to continue (cityMap, Tx)
# model.load_state_dict(torch.load('resultsWCityWTxWCarsLr3LargerIRT2carsOnlineDSNoEpsCorVarCarsCDPM/WCityWTxEpoch1.pt'))

#looks good, CoM NaN corrected (v1, denom 0 olmadi ama bi bokluk oldu, print in txt), run epoch, 
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler)


# In[ ]:


#Save Model with best val loss
stringer = 'resultsWCityWTxWCarsLr3LargerIRT2carsOnlineDSNoEpsCorVarCarsCDPM/BestModel.pt'
torch.save(model.state_dict(), stringer)

# #load the saved model above to continue
# model.load_state_dict(torch.load('../data/trainedModels/WorksEpoch0.pt'))
# model.to(device)#accuracy test for 5eps CoM Epoch 1-2



