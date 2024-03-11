import torch.nn as nn
import numpy as np
import torch    
 

class MyRnnNet2Layers(nn.Module):
    def __init__(self, params):
        super(MyRnnNet2Layers, self).__init__()
        self.inputSize = params["inputSize"]
        self.hiddenSize_L1 = params["hL1"]
        self.hiddenSize_L2 = params["hL2"]
        self.hiddenSize_L3 = params["hLinear"]
        self.outSize = params["outputSize"]
        
        self.l2 = nn.LSTMCell(self.inputSize, self.hiddenSize_L1)
        self.l3 = nn.LSTMCell(self.hiddenSize_L1, self.hiddenSize_L2)
        self.l4 = nn.Linear(self.hiddenSize_L2, self.hiddenSize_L3)
        self.relu = nn.ReLU()
        self.l5 = nn.Linear(self.hiddenSize_L3, self.outSize)  
    
    def forward(self, x, hx1, cx1, hx2, cx2):
        hx1, cx1 = self.l2(x, (hx1, cx1))
        hx2, cx2 = self.l3(hx1, (hx2, cx2))
        out = self.l4(hx2)
        out = self.relu(out)
        out = self.l5(out)
        # no activation and no softmax at the end
        return out, hx1, cx1, hx2, cx2

