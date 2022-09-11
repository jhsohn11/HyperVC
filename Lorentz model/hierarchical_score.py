import torch.nn as nn
import torch

class get_k_from_hscore(nn.Module):
    def __init__(self, t_hscore):
        super(get_k_from_hscore, self).__init__()
        self.t_hscore = t_hscore

        self.linear1_k = nn.Linear(1, 1)
        self.relu = nn.ReLU()
        self.linear2_k = nn.Linear(1, 1)


    def forward(self, t_list):
        hs_list = []
        for i in t_list.tolist():
            hs_list.append(self.t_hscore[int(i)])
        hs = torch.FloatTensor(hs_list).cuda().unsqueeze(1)
        l1 = self.linear1_k(hs)
        act = self.relu(l1)
        k = self.linear2_k(act)

        return k.squeeze(1)
        
    def predict(self, t):
        hs = self.t_hscore[int(t)]
        l1 = self.linear1_k(torch.FloatTensor([hs]).cuda().unsqueeze(1))
        k = self.linear2_k(self.relu(l1))
        
        return k.squeeze(1)