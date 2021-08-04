from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
warnings.filterwarnings("ignore")
import scipy.io


    
class locDL(Dataset):
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="dataset/",
                 numTx=5,   
                 numTrials=50,
                 numRx=200,
                 simulation="DPM",
                 cityMap="false", 
                 carsMap="false", 
                 TxMaps="false",
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the dataset.
            numTx: Number of transmitters per map. Default and maximum numTx = 5.  
            numTrials: Number of sets of numTx transmitters per map
            simulation:"DPM", "IRT2", "DPMtoIRT2", "DPMcars". Default= "DPM"
            cityMap: . Default cityMap="false"
            TxMaps: Images of Tx. Defaul TxMaps="false"
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The LocUNet inputs.  
            RXlocr: Pixel row of true location
            RXlocc: Pixel column of true location
            
        """
       
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,99,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=68
        elif phase=="val":
            self.ind1=69
            self.ind2=83
        elif phase=="test":
            self.ind1=84
            self.ind2=98
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx =  numTx                
        self.numTrials =  numTrials
        self.numRx = numRx
        self.simulation=simulation
        self.cityMap=cityMap
        self.carsMap=carsMap
        self.TxMaps=TxMaps
        self.transform= transform
        
        self.height = 256
        self.width = 256
        
        if simulation=="DPM":
            self.dir_gainTrue=self.dir_dataset+"DPM/true/"
            self.dir_gainEst=self.dir_dataset+"DPM/estimate/"
        elif simulation=="ZSDPMtoIRT2":
            self.dir_gainTrue=self.dir_dataset+"IRT2/true/"
            self.dir_gainEst=self.dir_dataset+"DPM/estimate/"
        elif simulation=="DPMtoIRT2":
            self.dir_gainTrue=self.dir_dataset+"IRT2/true/"
            self.dir_gainEst=self.dir_dataset+"DPMtoIRT2/estimate/"    
        elif simulation=="DPMcars" :
            self.dir_gainTrue=self.dir_dataset+"DPMcars/true/"
            self.dir_gainEst=self.dir_dataset+"DPMcars/estimate/"                  
        elif simulation=="IRT2carsCDPM" :
            self.dir_gainTrue=self.dir_dataset+"IRT2cars/true/"
            self.dir_gainEst=self.dir_dataset+"DPM/estimate/"    
        elif simulation=="IRT2carsCDPMtoIRT2" :
            self.dir_gainTrue=self.dir_dataset+"IRT2cars/true/"
            self.dir_gainEst=self.dir_dataset+"DPMtoIRT2/estimate/"        
                          
        self.dir_buildings=self.dir_dataset+"buildings/"
        self.dir_cars=self.dir_dataset+"cars/"        
        self.dir_Tx = self.dir_dataset+ "png/antennas/"         

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTrials*self.numRx
    
    def __getitem__(self, idx):
        numMapPhase = self.ind2-self.ind1+1
        idxMap,idxTrial,idxRx = np.unravel_index(idx,(numMapPhase,self.numTrials,self.numRx))
        dataset_map_ind=self.maps_inds[idxMap+self.ind1]
        #names of files that depend only on the map:
        nameMap = str(dataset_map_ind) + ".png"

        #Load true (reported) radio maps for Txs:
        mat = np.load('my_fileCorr.npy',allow_pickle='TRUE').item()
        rxx = mat['rxx']
        rxy = mat['rxy']
        RXr = rxx[dataset_map_ind,idxRx] - 1
        RXc = rxy[dataset_map_ind,idxRx] - 1
        antList = mat['antList']
        TXlist = antList[idxTrial,dataset_map_ind,:]
            
        
        inputEstMaps = []
        for m in range(self.numTx):
            name2 = str(dataset_map_ind) + "_" + str(TXlist[m]-1) + ".png"
            img_name_gainTrue = os.path.join(self.dir_gainTrue, name2)  
            image_gainTrue = np.asarray(io.imread(img_name_gainTrue))/255
            img_name_gainEst = os.path.join(self.dir_gainEst, name2)  
            image_gainEst = np.asarray(io.imread(img_name_gainEst))/255                
            inputEstMaps.append(image_gainEst) 
            gainTrue = image_gainTrue[RXr,RXc]
            imgGainTrue = gainTrue*np.ones(np.shape(image_gainEst))
            inputEstMaps.append(imgGainTrue) 
        inputs = inputEstMaps    
                
                 
        #Load Tx maps
        if self.TxMaps == "true":    
            antX = mat['antX']
            antY = mat['antY']
            TXr = antX[idxTrial,:,dataset_map_ind]
            TXc = antY[idxTrial,:,dataset_map_ind]
            inputTxMaps = []
            for m in range(self.numTx):
                imTx = np.zeros((256,256))
                imTx[TXr[m],TXc[m]] = 1
                inputs.append(imTx)
                       
        
        #Load buildings:
        if self.cityMap == "true":
            img_name_buildings = os.path.join(self.dir_buildings, nameMap)        
            image_buildings = np.asarray(io.imread(img_name_buildings))/255  
            inputs.append(image_buildings)
           
            
        #Load cars:
        if self.carsMap == "true":
            img_name_cars = os.path.join(self.dir_cars, nameMap)        
            image_cars = np.asarray(io.imread(img_name_cars))/255  
            inputs.append(image_cars)  
        inputs = np.asarray(inputs, dtype=np.float32)   
        inputs = np.transpose(inputs, (1, 2, 0))
            
        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)      
  
        #True coordinates      
        RXlocr = torch.from_numpy(np.asarray(RXr, dtype=np.float32))
        RXlocc = torch.from_numpy(np.asarray(RXc, dtype=np.float32))

        return [inputs, torch.stack((RXlocr, RXlocc), dim=0)]

    
    

