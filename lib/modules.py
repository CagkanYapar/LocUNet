import torch
import torch.nn as nn
from torchvision import models
from scipy import ndimage

def convrelu(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.LeakyReLU(0.2, True),
        nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False) 
    )

def convreluT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Upsample(mode='bilinear', scale_factor=2),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.LeakyReLU(0.2, True)
    )

def get_centers_of_mass(tensor):
#taken from:https://gitlab.liu.se/emibr12/wasp-secc/blob/cb02839115da475c2ad593064e3b9daf2531cac3/utils/tensor_utils.py    
    """
    Args:
        tensor (Tensor): Size (*,height,width)
    Returns:
        Tuple (Tensor): Tuple of two tensors of sizes (*)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    width = tensor.size(-1)
    height = tensor.size(-2)

    x_coord_im = torch.linspace(0,255,width).repeat(height,1).to(device)
    y_coord_im = torch.linspace(0,255,height).unsqueeze(0).transpose(0,1).repeat(1,width).to(device)

    x_mean = torch.mul(tensor,x_coord_im).sum(-1).sum(-1)/torch.add(tensor.sum(-1).sum(-1),0.0000001)
    y_mean = torch.mul(tensor,y_coord_im).sum(-1).sum(-1)/torch.add(tensor.sum(-1).sum(-1),0.0000001)

    return (y_mean, x_mean)
   
class LocUNet(nn.Module):

    def __init__(self,inputs=11):
        super().__init__()
        
        self.inputs=inputs

        self.layer00 = convrelu(inputs, 20, 3, 1,1) #256
        self.layer0 = convrelu(20, 50, 5, 2,2) #128       
        self.layer1 = convrelu(50, 60, 5, 2,2) #64 
        self.layer10 = convrelu(60, 70, 5, 2,1) #64 
        self.layer11 = convrelu(70, 90, 5, 2,2)  #32
        self.layer110 = convrelu(90, 100, 5, 2,1) #32 
        self.layer2 = convrelu(100, 120, 5, 2,2) #16    
        self.layer20 = convrelu(120, 120, 3, 1,1) #16
        self.layer3 = convrelu(120, 135, 5, 2,1) #16
        self.layer31 = convrelu(135, 150, 5, 2,2) #8
        self.layer4 =convrelu(150, 225, 5, 2,1) #8
        self.layer41 =convrelu(225, 300, 5, 2,2) #4
        self.layer5 =convrelu(300, 400, 5, 2,1) #4
        self.layer51 =convrelu(400, 500, 5, 2,2) #2
        self.conv_up51 =convreluT(500, 400, 4, 1) #4
        self.conv_up5 =convrelu(400+400, 300, 5, 2, 1) #4
        self.conv_up41 = convreluT(300+300, 225, 4, 1) #8
        self.conv_up4 = convrelu(225+225, 150, 5, 2, 1) #8
        self.conv_up31 = convreluT(150 + 150, 135, 4, 1) #16
        self.conv_up3 = convrelu(135 + 135, 120, 5, 2, 1) #16
        self.conv_up20 = convrelu(120 + 120, 120, 3, 1, 1) #16
        self.conv_up2 = convreluT(120 + 120, 100, 6, 2) #32
        self.conv_up110 = convrelu(100 + 100, 90, 5, 2, 1)#32
        self.conv_up11 = convreluT(90 + 90, 70, 6, 2)#64
        self.conv_up10 = convrelu(70 + 70, 60, 5, 2, 1) #64
        self.conv_up1 = convreluT(60 + 60, 50, 6, 2)#128
        self.conv_up0 = convreluT(50 + 50, 20, 6, 2) #256
        self.conv_up00 = convrelu(20+20+inputs, 20, 5, 2,1)#256        
        self.conv_up000 = convrelu(20+inputs, 1, 5, 2,1)#256
        
    def forward(self, input):
        
        input0=input[:,0:self.inputs,:,:]
        layer00 = self.layer00(input0)
        layer0 = self.layer0(layer00)
        layer1 = self.layer1(layer0)
        layer10 = self.layer10(layer1)
        layer11 = self.layer11(layer10)
        layer110 = self.layer110(layer11)
        layer2 = self.layer2(layer110)
        layer20 = self.layer20(layer2)
        layer3 = self.layer3(layer20)
        layer31 = self.layer31(layer3)
        layer4 = self.layer4(layer31)
        layer41 = self.layer41(layer4)
        layer5 = self.layer5(layer41)
        layer51 = self.layer51(layer5)
        layer5u = self.conv_up51(layer51)
        layer5u = torch.cat([layer5u, layer5], dim=1)
        layer41u = self.conv_up5(layer5u)
        layer41u = torch.cat([layer41u, layer41], dim=1)
        layer4u = self.conv_up41(layer41u)
        layer4u = torch.cat([layer4u, layer4], dim=1)
        layer31u = self.conv_up4(layer4u)
        layer31u = torch.cat([layer31u, layer31], dim=1)
        layer3u = self.conv_up31(layer31u)
        layer3u = torch.cat([layer3u, layer3], dim=1)
        layer20u = self.conv_up3(layer3u)
        layer20u = torch.cat([layer20u, layer20], dim=1)
        layer2u = self.conv_up20(layer20u)
        layer2u = torch.cat([layer2u, layer2], dim=1)
        layer110u = self.conv_up2(layer2u)
        layer110u = torch.cat([layer110u, layer110], dim=1)
        layer11u = self.conv_up110(layer110u)
        layer11u = torch.cat([layer11u, layer11], dim=1)
        layer10u = self.conv_up11(layer11u)
        layer10u = torch.cat([layer10u, layer10], dim=1)
        layer1u = self.conv_up10(layer10u)
        layer1u = torch.cat([layer1u, layer1], dim=1)
        layer0u = self.conv_up1(layer1u)
        layer0u = torch.cat([layer0u, layer0], dim=1)
        layer00u = self.conv_up0(layer0u)
        layer00u = torch.cat([layer00u, layer00], dim=1)
        layer00u = torch.cat([layer00u,input0], dim=1)
        layer000u  = self.conv_up00(layer00u)
        layer000u = torch.cat([layer000u,input0], dim=1)      
        output  = self.conv_up000(layer000u)
        [outputRo, outputCo] = get_centers_of_mass(output)
        output = torch.cat((outputRo, outputCo), 1)

        return output

    
    
    
    
    
  