import torch
import torch.nn as nn


model = torch.load('Auto-PyTorch/examples/basics/TOPCAT_search.pt')


class Appended_Model(nn.Module):
    def __init__(self, last_in=79, last_out=2):
        super(Appended_Model, self).__init__()
        
        self.backbone = model[:-1]
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        #self.linear_main = nn.Linear(last_in, last_out)
        self.linear_main = model[-1]
        #self.linear = nn.Linear(last_in, last_out)
        self.linear =  model[-1]
        
        self.sm = nn.Softmax(1)
        
    def forward(self, x, main_fc=False):
        
        x = self.backbone(x)
        if main_fc:
            output = self.linear_main(x)
        else:
            output = self.linear(x)
            
        #output = torch.sigmoid(output) 
        output = self.sm(output)
        
        return output