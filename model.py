# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import SelectivePool1d,get_len_mask,PatternAwareNormalization,SEResBlock,CRB

class PAVENet(nn.Module):
    def __init__(self,d_in,num_pattern=8,d_hidden=512,num_classes=202):
        super().__init__()
        self.layer1 = CRB(d_in=d_in,d_out=512,kernel_size=7,stride=1,padding=3)
        self.layer2 = SEResBlock(d_in=512,kernel_size=3,stride=1,padding=2,dilation=2,scale=4)
        self.layer3 = SEResBlock(d_in=512,kernel_size=3,stride=1,padding=3,dilation=3,scale=4)
        self.layer4 = SEResBlock(d_in=512,kernel_size=3,stride=1,padding=4,dilation=4,scale=4)
        self.rnn = nn.LSTM(input_size=d_hidden,hidden_size=d_hidden,num_layers=2,
            bias=False,dropout=0,batch_first=True,bidirectional=False)
        # self.rnn = nn.TransformerEncoderLayer(512,nhead=4,dim_feedforward=512)
        # self.rnn = TCN(d_in=512,num_channels=[512])
        self.pattern_norm = PatternAwareNormalization(512,num_pattern)
        self.sel_pool1 = SelectivePool1d(d_hidden,d_head=32,num_heads=10)
        self.sel_pool2 = SelectivePool1d(d_hidden,d_head=32,num_heads=10)
        self.head = nn.Sequential(
            nn.Linear(640,d_hidden * 2,bias=False),
            nn.BatchNorm1d(d_hidden * 2),
            nn.SELU(True),
            nn.Linear(d_hidden * 2,num_classes,bias=False)
        )
        nn.init.kaiming_normal_(self.head[0].weight,a=1)
        nn.init.kaiming_normal_(self.head[3].weight,a=1)

    def forward(self,x,feature_lens):
        x = x.transpose(1,2)
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y1 + y2)
        y4 = self.layer4(y1 + y2 + y3)
        y4 = y4.transpose(1,2)
        y5 = self.rnn(y4)[0]
        # y5 = self.rnn(y4)
        score = F.softmax(y5,dim=-1)
        y5 = y4 * score + y4
        y6 = self.pattern_norm(y4)
        y4 = y4 + y6 # (n,l,c)
        mask = get_len_mask(feature_lens)
        f1 = self.sel_pool1(y4.permute(0,2,1),mask)
        f2 = self.sel_pool2(y5.permute(0,2,1),mask) # (n,c)
        y_vector = torch.cat([f1,f2],dim=1) # (n,2c)
        y_prob = self.head(y_vector) # (n,num_classes)
        return y_vector,y_prob