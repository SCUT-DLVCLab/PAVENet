# -*- coding:utf-8 -*-

import os,pickle,glob
import numpy as np
from copy import deepcopy
from natsort import natsorted
from utils import centernorm_size,norm_pressure,interpolate_torch

def extract_single_session(session_num='session1'):
    train_num = 202
    random_number_train = {}
    random_number_test = {}

    for u in natsorted(os.listdir(f'data/HRDS4BV/{session_num}'))[:train_num]:
        random_number_train[u] = {'genuine':[],'forgery':[]}
        files = glob.glob(f'data/HRDS4BV/{session_num}/{u}/*')
        genuine = natsorted(list(filter(lambda x:'g_' in x,files)))
        forgery = natsorted(list(filter(lambda x:'f_' in x,files)))
        for g in genuine:
            with open(g,'r',encoding='utf-8') as f:
                lines = f.readlines()[1:]
            cur_info = [[float(i) for i in line.split()[:3]] for line in lines]
            cur_info = np.array(cur_info,float).astype('float32')
            cur_info = centernorm_size(cur_info)
            cur_info = norm_pressure(cur_info)
            cur_info = interpolate_torch(cur_info,2)
            random_number_train[u]['genuine'].append(cur_info)
        for fo in forgery:
            with open(fo,'r',encoding='utf-8') as f:
                lines = f.readlines()[1:]
            cur_info = [[float(i) for i in line.split()[:3]] for line in lines]
            cur_info = np.array(cur_info,float).astype('float32')
            cur_info = centernorm_size(cur_info)
            cur_info = norm_pressure(cur_info)
            cur_info = interpolate_torch(cur_info,2)
            random_number_train[u]['forgery'].append(cur_info)

    for u in natsorted(os.listdir(f'data/HRDS4BV/{session_num}'))[train_num:]:
        random_number_test[u] = {'genuine':[],'forgery':[]}
        files = glob.glob(f'data/HRDS4BV/{session_num}/{u}/*')
        genuine = natsorted(list(filter(lambda x:'g_' in x,files)))
        forgery = natsorted(list(filter(lambda x:'f_' in x,files)))
        for g in genuine:
            with open(g,'r',encoding='utf-8') as f:
                lines = f.readlines()[1:]
            cur_info = [[float(i) for i in line.split()[:3]] for line in lines]
            cur_info = np.array(cur_info,float).astype('float32')
            cur_info = centernorm_size(cur_info)
            cur_info = norm_pressure(cur_info)
            cur_info = interpolate_torch(cur_info,2)
            random_number_test[u]['genuine'].append(cur_info)
        for fo in forgery:
            with open(fo,'r',encoding='utf-8') as f:
                lines = f.readlines()[1:]
            cur_info = [[float(i) for i in line.split()[:3]] for line in lines]
            cur_info = np.array(cur_info,float).astype('float32')
            cur_info = centernorm_size(cur_info)
            cur_info = norm_pressure(cur_info)
            cur_info = interpolate_torch(cur_info,2)
            random_number_test[u]['forgery'].append(cur_info)

    os.makedirs('./data',exist_ok=True)
    with open(f'./data/hrds4bv-{session_num}-train.pkl','wb') as f:
        pickle.dump(random_number_train,f)
    with open(f'./data/hrds4bv-{session_num}-test.pkl','wb') as f:
        pickle.dump(random_number_test,f)

def merge_seq():
    for mode in ['train','test']:
        first = pickle.load(open(f'./data/hrds4bv-session1-{mode}.pkl','rb'),encoding='iso-8859-1')
        second = pickle.load(open(f'./data/hrds4bv-session2-{mode}.pkl','rb'),encoding='iso-8859-1')
        assert list(first.keys()) == list(second.keys())
        across = deepcopy(first)
        for k in across.keys():
            across[k]['genuine'].extend(second[k]['genuine'])
            across[k]['forgery'].extend(second[k]['forgery'])
        with open(f'./data/hrds4bv-across-{mode}.pkl','wb') as f:
            pickle.dump(across,f)

if __name__ == '__main__':
    extract_single_session('session1')
    extract_single_session('session2')
    merge_seq()