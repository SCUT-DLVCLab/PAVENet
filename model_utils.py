# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelectivePool1d(nn.Module):
    def __init__(self,in_features,d_head,num_heads,tau=1.0):
        super().__init__()
        self.keys = nn.Parameter(torch.Tensor(num_heads,d_head),requires_grad=True)
        self.W_q = nn.Conv1d(in_features,d_head * num_heads,kernel_size=1)
        self.norm = 1 / np.sqrt(d_head)
        self.d_head = d_head
        self.num_heads = num_heads
        self.cnt = 0
        self.weights_init()

    def weights_init(self):
        nn.init.orthogonal_(self.keys,gain=1)
        nn.init.kaiming_normal_(self.W_q.weight,a=1)
        nn.init.zeros_(self.W_q.bias)

    def orthogonal_norm(self):
        keys = F.normalize(self.keys,dim=1)
        corr = torch.mm(keys,keys.transpose(0,1))
        return torch.sum(torch.triu(corr,1).abs_())

    def forward(self,x,mask):
        N,_,L = x.shape # (N,C,L)
        q = v = self.W_q(x).transpose(1,2).view(N,L,self.num_heads,self.d_head)
        if mask != None:
            mask = mask.to(x.device)
            attn = F.softmax(torch.sum(q * self.keys,dim=-1) * self.norm - (1. - mask).unsqueeze(2) * 1000,dim=1) 
            # (N,L,num_heads)
        else:
            attn = F.softmax(torch.sum(q * self.keys,dim=-1) * self.norm,dim=1)
        y = torch.sum(v * attn.unsqueeze(3),dim=1).view(N,-1) # (N,d_head * num_heads)
        return y

def get_len_mask(features_lens):
    batch_size = len(features_lens)
    max_len = torch.max(features_lens)
    mask = torch.zeros((batch_size,max_len),dtype=torch.float32) # 必须要减
    for i in range(batch_size):
        mask[i,0:features_lens[i]] = 1.0
    return mask

class SEBlock(nn.Module):
    def __init__(self,d_in,d_hidden):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_in,d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden,d_in),
            nn.Sigmoid())

    def forward(self,x): # x: (n,l,c)
        n,_,c = x.size()
        y = self.avgpool(x.permute(0,2,1)).squeeze()
        y = self.fc(y).view(n,1,c)
        return x * y.expand_as(x)

class SEBlock2(nn.Module): # 通道不一样
    def __init__(self,d_in,d_hidden):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_in,d_hidden,kernel_size=1,padding=0,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_hidden,d_in,kernel_size=1,padding=0,stride=1),
            nn.Sigmoid())

    def forward(self,x): # x: (n,c,l)
        y = self.fc(x)
        return x * y.expand_as(x)

def neighbor_max(x,kernel_size=3,num_pattern=8):
    xt = x.sum(dim=1,keepdim=True)
    hmax = F.max_pool1d(xt,kernel_size=kernel_size,stride=1,padding=1) # 八近邻最大值
    # x_points = torch.where(xt == hmax,xt,torch.full_like(xt,-torch.finfo(x.dtype).max))
    x_points = torch.where(xt == hmax,xt,torch.full_like(xt,0))
    scores,indices = torch.topk(x_points,k=num_pattern,sorted=False)
    # print(indices.cpu().numpy())
    indices = indices.squeeze()
    pattern_len = x.size(2) // 4 // num_pattern
    pattern_range = torch.tensor([*range(0,pattern_len)]).to(x.device)
    indices = indices.unsqueeze(-1) + pattern_range.view(1,1,-1) - pattern_len // 2
    # print(indices.cpu().numpy())
    indices = indices.view(xt.shape[0],-1)
    indices = torch.clamp(indices,min=0,max=x.size(2) - 1)
    y = x.transpose(1,2)
    components = torch.cat([y[i:i + 1,indices[i],:] for i in range(x.shape[0])],dim=0)
    mask = torch.full_like(y,-3)
    # for i,idx in enumerate(indices):
        # mask[i,idx,:] = y[i:i + 1,indices[i],:]
    for i,idx in enumerate(indices):
        mask[i,idx,:] = 2
    # # mask = torch.softmax(mask,dim=1)
    mask = torch.sigmoid(mask)
    return mask,components

class PatternAwareNormalization(nn.Module):
    def __init__(self,d_feat,num_pattern,eps=1e-8):
        super().__init__()
        self.d_feat = d_feat
        self.num_pattern = num_pattern
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_feat))
        self.beta = nn.Parameter(torch.zeros(d_feat))

    def forward(self,x):
        mask,components = neighbor_max(x.transpose(1,2),num_pattern=self.num_pattern)
        cmean = torch.mean(components,dim=(-1,-2),keepdim=True)
        cstd = torch.std(components,dim=(-1,-2),keepdim=True)
        y = self.gamma * (x - cmean) / (cstd + self.eps) + self.beta
        y = y * mask
        return y

class ResCRB(nn.Module):
    def __init__(self,d_in,kernel_size=1,stride=1,padding=0,dilation=0,bias=False,scale=4):
        super().__init__()
        self.scale = scale
        self.d_in = d_in // scale
        self.num_layers = scale if scale == 1 else scale - 1
        self.conv = nn.ModuleList([nn.Conv1d(self.d_in,self.d_in,kernel_size,stride,padding,
            dilation=dilation,bias=bias) for _ in range(self.num_layers)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(self.d_in) for _ in range(self.num_layers)])

    def forward(self,x):
        y = []
        spx = torch.split(x,self.d_in,1)
        for i in range(self.num_layers):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.bn[i](F.relu(self.conv[i](sp)))
            y.append(sp)
        if self.scale != 1:
            y.append(spx[self.num_layers])
        y = torch.cat(y,dim=1)
        return y

class CRB(nn.Module):
    def __init__(self,d_in,d_out,kernel_size=1,stride=1,padding=0,bias=False):
        super().__init__()
        self.conv = nn.Conv1d(d_in,d_out,kernel_size,stride=stride,padding=padding,bias=bias)
        self.bn = nn.BatchNorm1d(d_out)

    def forward(self,x):
        return self.bn(F.relu(self.conv(x)))

class SEResBlock(nn.Module):
    def __init__(self,d_in,kernel_size,stride,padding,dilation,scale):
        super().__init__()
        self.net = nn.Sequential(
            CRB(d_in,d_in,kernel_size=1,stride=1,padding=0),
            ResCRB(d_in,kernel_size,stride=stride,padding=padding,dilation=dilation,scale=scale),
            CRB(d_in,d_in,kernel_size=1,stride=1,padding=0),
            SEBlock2(d_in,d_in // 8),
        )

    def forward(self,x):
        return self.net(x)