import torch
import torch.nn as nn
import torch.nn.functional as F

## Define the NN architecture
class Net(nn.Module):
    def __init__(self, h_sizes, drop_r, out_size):
        super(Net, self).__init__()
        
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.hidden.append(nn.BatchNorm1d(h_sizes[k+1]))
            
            
        self.dropout = nn.Dropout(drop_r)
        self.linear = nn.Linear(h_sizes[-1], out_size)
        self.linear_main = nn.Linear(h_sizes[-1], out_size)
        


    def forward(self, x, main_fc=False):

        for idx, layer in enumerate(self.hidden):
            if idx % 2 == 0:
                x = F.relu(layer(x))
            else:
                x = layer(x)
            
        #x = self.dropout(x)
        
        if main_fc:
            output = torch.sigmoid(self.linear_main(x))
        else:
            output = torch.sigmoid(self.linear(x))
            
        return output