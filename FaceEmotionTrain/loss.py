import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class CELoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss().to(device)

    def forward(self, pred, label):
        return self.criterion(pred, label)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, device='cpu'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.device=device

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight).to(self.device)

def compute_adjustment(train_loader, device, tro=1.0):
    ## calculate count of labels
    label_count = {}
    for _, target in tqdm(train_loader):
        target = target.to(device)
        for j in target:
            key = int(j.item())
            label_count[key] = label_count.get(key, 0) + 1
    label_count = dict(sorted(label_count.items()))
    label_count_array = np.array(list(label_count.values()))
    label_count_array = label_count_array / label_count_array.sum()
    adj = np.log(label_count_array ** tro + 1e-12)
    adj = torch.from_numpy(adj)
    adj = adj.to(device)
    return adj