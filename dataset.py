# -*- coding:utf-8 -*-

import torch
import numpy as np
from utils import extract_features,path_drop
from torch.utils.data import Dataset

class Numbers(Dataset):
    def __init__(self,handwriting_info,train=True):
        super().__init__()
        self.users = handwriting_info.keys()
        self.users_cnt = len(self.users)
        self.train = train
        self.features = []
        self.genuine_cnt = np.zeros(self.users_cnt,dtype=np.int32)
        self.forgery_cnt = np.zeros(self.users_cnt,dtype=np.int32)
        for i,k in enumerate(self.users):
            extract_features(handwriting_info[k]['genuine'],self.features)
            extract_features(handwriting_info[k]['forgery'],self.features)
            self.genuine_cnt[i] = len(handwriting_info[k]['genuine'])
            self.forgery_cnt[i] = len(handwriting_info[k]['forgery'])
        self.features_cnt = len(self.features)
        accu_indices = np.cumsum(self.genuine_cnt + self.forgery_cnt)
        self.accu_indices = np.roll(accu_indices,1)
        self.accu_indices[0] = 0
        self.feature_dims = np.shape(self.features[0])[1] # 12
        self.each_feature_len = []
        for f in self.features:
            self.each_feature_len.append(f.shape[0])
        user_interval = self.genuine_cnt[0] + self.forgery_cnt[0]
        self.user_labels = []
        for i in range(self.users_cnt):
            self.user_labels.extend([i] * user_interval)
        self.binary_labels = ([1] * self.genuine_cnt[0] + [0] * self.forgery_cnt[0]) * self.users_cnt

    @property
    def config(self):
        return {
            'users_cnt':self.users_cnt,
            'accu_indices':self.accu_indices,
            'forgery_cnt':self.forgery_cnt,
            'genuine_cnt':self.genuine_cnt,
        }
        
    def __len__(self):
        return self.features_cnt

    def __getitem__(self,idx):
        feature = self.features[idx]
        return feature,len(feature),self.user_labels[idx],self.binary_labels[idx]

class TestSampler:
    def __init__(self,users_cnt,genuine_sample=5,forgery_sample=5):
        self.users_cnt = users_cnt
        self.users_indices = np.arange(0,self.users_cnt,dtype=np.int32)
        self.user_sample = genuine_sample + forgery_sample # 每个batch的用户个数
        self.genuine_sample = genuine_sample
        self.forgery_sample = forgery_sample

    def __len__(self):
        return self.users_cnt

    def __iter__(self):
        np.random.shuffle(self.users_indices)
        for i in range(self.users_cnt): # 每次都取一个用户，每个都拿全部20个，然后55个就取完了
            batch_indices = np.arange(i * self.user_sample,(i + 1) * self.user_sample,dtype=np.int32)
            yield batch_indices
 
def collate_fn(batch):
    batch_size = len(batch)
    handwriting = [i[0] for i in batch]
    hw_len = np.array([i[1] for i in batch],dtype=np.float32)
    user_labels = np.array([i[2] for i in batch])
    binary_labels = np.array([i[3] for i in batch])
    max_len = int(np.max(hw_len))
    time_function_cnts = np.shape(handwriting[0])[1]
    handwriting_padded = np.zeros((batch_size,max_len,time_function_cnts),dtype=np.float32)
    for i,hw in enumerate(handwriting):
        handwriting_padded[i,:hw.shape[0]] = hw # 就是后面补零
    return handwriting_padded,hw_len,user_labels,binary_labels