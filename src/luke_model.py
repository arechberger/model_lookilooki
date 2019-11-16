import torch
import torch.nn as nn

class LukeConvBlock(nn.Module):
    def __init__(self, insize, outsize, batchnorm):
        super().__init__()
        self.insize = insize
        self.outsize = outsize
        self.batchnorm = batchnorm
        block  = []
        block.append(nn.Conv2d(insize, outsize, kernel_size=3, padding=1))
        block.append(nn.ReLU(inplace=True))
        if batchnorm:
            block.append(nn.BatchNorm2d(outsize))
        block.append(nn.Conv2d(outsize, outsize, kernel_size=3, padding=1))
        block.append(nn.ReLU(inplace=True))
        if batchnorm:
            block.append(nn.BatchNorm2d(outsize))
        block.append(nn.MaxPool2d(2, stride=0))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)       
        
                     
class Luke(nn.Module):
    

    
    def __init__(self, fl_features=16, channels=1, depth=3, batchnorm=True):
        super().__init__()
        
        model_seq = []
        model_seq.append(LukeConvBlock(channels,fl_features,batchnorm))
        for i in range(depth-1):
            model_seq.append(LukeConvBlock(fl_features*2**(i),fl_features*2**(i+1),batchnorm))
            
        self.model_seq = nn.Sequential(*model_seq)
        self.adaptpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.nfeat_afterconv = 7*7*fl_features*2**(i+1)
        classifier = []
        classifier.append(nn.Linear(self.nfeat_afterconv,2000))
        classifier.append(nn.ReLU(inplace=True))
        if batchnorm:
            classifier.append(nn.BatchNorm1d(2000))
        classifier.append(nn.Linear(2000,1000))
        classifier.append(nn.ReLU(inplace=True))
        if batchnorm:
            classifier.append(nn.BatchNorm1d(1000))
        classifier.append(nn.Linear(1000,4))
        self.classifier = nn.Sequential(*classifier)
        
    def forward(self, x):
        x = self.model_seq(x)
        x = self.adaptpool(x)
        x = x.view(-1,self.nfeat_afterconv)
        return self.classifier(x)