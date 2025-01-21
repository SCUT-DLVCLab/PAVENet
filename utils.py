# -*- coding:utf-8 -*-

from scipy import signal
import numpy as np
import logging,sys,time,os
from functools import wraps,lru_cache
from termcolor import colored
import torch
import torch.nn.functional as F

class ButterWorthLPF: # 巴特沃斯低通滤波器
    def __init__(self,order=3,half_pnt=15.0,fnyquist=100.0):
        fM = 0.5 * fnyquist
        half_pnt /= fM
        b,a = signal.butter(order,half_pnt,'low')
        self.b = b # 分子
        self.a = a # 分母

    def __call__(self,x): # 将data通过零相位滤波器，零相位的意思就是输入和输出信号的相位完全相同，相移为0
        return signal.filtfilt(self.b,self.a,x)

lpf = ButterWorthLPF()

def difference(x): # 差分，两跨点之间相减
    delta_x = np.convolve(x,[0.5,0,-0.5],mode='same')
    delta_x[0] = delta_x[1]
    delta_x[-1] = delta_x[-2]
    return delta_x

def difference_theta(x):
    delta_x = np.zeros_like(x)
    delta_x[1:-1] = x[2:] - x[:-2]
    delta_x[-1] = delta_x[-2]
    delta_x[0] = delta_x[1]
    t = np.where(np.abs(delta_x) > np.pi)
    delta_x[t] = np.sign(delta_x[t]) * 2 * np.pi
    delta_x *= 0.5
    return delta_x

def extract_features(handwrittings,features,num=2):
    '''
        paths: 路径列表，第一维应该是点的个数
        features: 这个就是所有特征的列表，是单个feature append进去的
        num: 使用的信息个数，比如x,y,pressure...,2就是只用012前三个
    '''
    for handwritting in handwrittings:
        pressure = handwritting[:,num]
        handwriting = handwritting[:,0:num] # (x,y,pressure)
        handwritting[:,0] = lpf(handwritting[:,0])
        handwritting[:,1] = lpf(handwritting[:,1])
        delta_x = difference(handwritting[:,0])
        delta_y = difference(handwritting[:,1])
        v = np.sqrt(delta_x ** 2 + delta_y ** 2) # 速度
        theta = np.arctan2(delta_y,delta_x)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        delta_v = difference(v)
        delta_theta = np.abs(difference_theta(theta))
        log_curve_radius = np.log((v + 0.05) / (delta_theta + 0.05)) # log的曲线弧度
        delta_v2 = np.abs(v * delta_theta)
        acceleration = np.sqrt(delta_v ** 2 + delta_v2 ** 2)

        # None在这里的作用是升维，比如说[2,2]会变成[2,1,2],concat起来就是[2,x,2]
        single_feature = np.concatenate((delta_x[:,None],delta_y[:,None],v[:,None],
            cos_theta[:,None],sin_theta[:,None],theta[:,None],log_curve_radius[:,None],
            acceleration[:,None],delta_v[:,None],delta_v2[:,None],delta_theta[:,None],
            pressure[:,None]),axis=1).astype(np.float32)
        single_feature[:,:-1] = (single_feature[:,:-1] - np.mean(single_feature[:,:-1],axis=0)) / \
            np.std(single_feature[:,:-1],axis=0)
        features.append(single_feature)

def interpolate_torch(org_info,interp_ratio):
    l = len(org_info)
    org_info = torch.tensor(org_info).view(1,1,l,-1)
    new_info = F.interpolate(org_info,size=(l * interp_ratio,3),mode='bicubic').squeeze().numpy()
    return new_info

def path_drop(path,low=0.05,high=0.075):
    seq_len = path.shape[0]
    r = (high - low) * np.random.random_sample() + low
    drop_len = int(r * seq_len)
    drop_idx = np.random.choice(np.arange(1,seq_len),drop_len,replace=False)
    path = np.delete(path,drop_idx,axis=0)
    return path

def clock(func):
    @wraps(func)
    def impl(*args,**kwargs):
        start = time.perf_counter()
        res = func(*args,**kwargs)
        end = time.perf_counter()
        args_list = []
        if args:
            args_list.extend([repr(arg) for arg in args])
        if kwargs:
            args_list.extend([f'{key}={value}' for key,value in kwargs.items()])
        args_str = ','.join(i for i in args_list)
        print(f'[executed in {(end - start):.5f}s, '
            f'{func.__name__}({args_str}) -> {res}]')
        return res
    return impl

def save_ckpt(epoch,model,optimizer,lr_scheduler,logger,save_folder,subname):
    save_state = {
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'lr_scheduler':lr_scheduler.state_dict(),
        'epoch':epoch
    }
    save_path = f'{save_folder}/ckpt-{epoch}-{subname}.pth'
    logger.info(f'{save_path} saving checkpoint......')
    torch.save(save_state,save_path)
    logger.info(f'{save_path} successfully saved.')

def load_ckpt(model,pretrained_root,device,logger): 
    state_dict = torch.load(pretrained_root,map_location=device)
    print(model.load_state_dict(state_dict))

    # sd = {k.replace('module.',''):v for k,v in state_dict['model'].items()}
    # model.load_state_dict(sd)
    logger.info(f'mode: "testing" {pretrained_root} successfully loaded.')

@lru_cache()
def create_logger(log_root,name='',test=False):
    os.makedirs(log_root,exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]','green') + \
        colored('(%(filename)s %(lineno)d)','yellow') + ': %(levelname)s %(message)s'
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO) # 分布式的等级
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt,datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    date = time.strftime('%Y-%m-%d') if not test else time.strftime('%Y-%m-%d') + '_test'
    file_handler = logging.FileHandler(f'{log_root}/log_{date}.txt',mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt,datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    return logger

def l2_norm(x): # x:(batch_size,seq_len)
    org_size = x.size()
    x_pow = torch.pow(x,2)
    x_pow = torch.sum(x_pow,1).add_(1e-6)
    x_pow = torch.sqrt(x_pow)
    y = torch.div(x,x_pow.view(-1,1).expand_as(x)).view(org_size)
    return y

def centernorm_size(handwriting,coord_idx=[0,1]):
    # coord_idx其实是下标，就是说在handwriting这个二维数组里面是下标0和1分别是x和y
    assert len(coord_idx) == 2
    pos = handwriting[:,coord_idx]
    minx = np.min(pos,axis=0)
    maxn = np.max(pos,axis=0)
    pos = (pos - (maxn + minx) / 2.) / np.max(maxn - minx) # 不知道为什么这样除，经验值
    handwriting[:,coord_idx] = pos
    return handwriting

def norm_pressure(handwriting,pressure_idx=2): # 单纯变0到1，但是其实可以不用
    pressure = handwriting[:,pressure_idx]
    maxn = np.max(pressure)
    pressure /= maxn
    handwriting[:,pressure_idx] = pressure
    return handwriting

if __name__ == '__main__':
    weights = torch.load('weights/ckpt-300-ECSEVer.pth',map_location='cpu')
    sd = {k.replace('module.',''):v for k,v in weights['model'].items()}
    torch.save(sd,'weights/model.pth')
