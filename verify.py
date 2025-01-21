# -*- coding:utf-8 -*-

import numpy as np

np.set_printoptions(threshold=1e4)
output_root = './output'

def select_template(dist_matrix): # (template_num,template_num)
    template_num = len(dist_matrix)
    if template_num == 1:
        return None,1,1,1,1
    dist_matrix = (dist_matrix + dist_matrix.transpose()) # 本来是上三角，转置相加之后就变成了上下三角了
    dist_avg = np.sum(dist_matrix,axis=1) / (template_num - 1)
    min_idx = np.argmin(dist_avg)
    dist_var = np.sqrt((np.sum(np.power(dist_matrix,2)) / template_num / (template_num - 1) - \
        np.power(np.sum(dist_matrix) / template_num / (template_num - 1),2)))
    dist_mean = np.sum(dist_matrix) / template_num / (template_num - 1)
    dist_temp = np.sum(dist_matrix[:,min_idx]) / (template_num - 1)
    dist_max = np.mean(np.max(dist_matrix,axis=1))
    dist_matrix[range(template_num),range(template_num)] = float('inf')
    dist_matrix[dist_matrix == 0] = float('inf')
    dist_min = np.mean(np.min(dist_matrix,axis=1))
    return min_idx,dist_temp ** 0.5,dist_max ** 0.5,dist_min ** 0.5,dist_mean ** 0.5

def getEER(FAR,FRR):
    a = (FRR <= FAR)
    s = np.sum(a)
    a[-s - 1] = 1
    a[-s + 1:] = 0
    FRR = FRR[a]
    FAR = FAR[a]
    a = [[FRR[1] - FRR[0],FAR[0] - FAR[1]],[-1,1]]
    b = [(FRR[1] - FRR[0]) * FAR[0] - (FAR[1] - FAR[0]) * FRR[0],0]
    return np.linalg.solve(a,b)

def verify(logger,template_num,genuine_sample,forgery_sample,rf=True):
    logger.info(f'template number: {template_num} rf: {rf}')
    EER_global,EER_local = [],[]
    num_users = 200
    num_test_p = genuine_sample - template_num
    num_test_n = num_users - 1 if rf else forgery_sample
    dist_genuine = np.load(f'{output_root}/dist_genuine.npy')
    dist_forgery = np.load(f'{output_root}/dist_forgery.npy')
    dist_template = np.load(f'{output_root}/dist_template.npy')
    datum_p,datum_n = [],[]
    for i in range(num_users):
        min_idx,dist_temp,dist_max,dist_min,dist_mean = select_template(
            dist_template[i * template_num:(i + 1) * template_num,0:template_num])
        dmax_p = np.max(dist_genuine[i * num_test_p:(i + 1) * num_test_p,0:template_num],axis=1) / dist_max
        dmin_p = np.min(dist_genuine[i * num_test_p:(i + 1) * num_test_p,0:template_num],axis=1) / dist_min
        dmean_p = np.mean(dist_genuine[i * num_test_p:(i + 1) * num_test_p,0:template_num],axis=1) / dist_mean

        dmax_n = np.max(dist_forgery[i * num_test_n:(i + 1) * num_test_n,0:template_num],axis=1) / dist_max
        dmin_n = np.min(dist_forgery[i * num_test_n:(i + 1) * num_test_n,0:template_num],axis=1) / dist_min
        dmean_n = np.mean(dist_forgery[i * num_test_n:(i + 1) * num_test_n,0:template_num],axis=1) / dist_mean
        datum_p.append(np.concatenate((dmax_p[:,None],dmin_p[:,None],dmean_p[:,None]),axis=1) / 10.)
        datum_n.append(np.concatenate((dmax_n[:,None],dmin_n[:,None],dmean_n[:,None]),axis=1) / 10.)
    datum_p = np.concatenate(datum_p,axis=0)
    datum_n = np.concatenate(datum_n,axis=0)

    EER_local_temp,EER_global_temp = [],[]
    for i in range(num_users):
        k = 1
        t = np.arange(0,50,0.01)[None,:]
        FRR = 1. - np.sum(np.sum(datum_p[i * num_test_p:(i + 1) * num_test_p,1:] * [1,1 / k],
            axis=1)[:,None] - t <= 0,axis=0) / float(num_test_p)
        FAR = 1. - np.sum(np.sum(datum_n[i * num_test_n:(i + 1) * num_test_n,1:] * [1,1 / k],
            axis=1)[:,None] - t >= 0,axis=0) / float(num_test_n)
        EER_local_temp.append(getEER(FAR,FRR)[0] * 100)
    k = 1
    t = np.arange(0,50,0.01)[None,:]
    FRR = 1. - np.sum(np.sum(datum_p[:,1:] * [1,1 / k],axis=1)[:,None] - t <= 0,axis=0) / float(len(datum_p))
    FAR = 1. - np.sum(np.sum(datum_n[:,1:] * [1,1 / k],axis=1)[:,None] - t >= 0,axis=0) / float(len(datum_n))
    EER_global_temp.append(getEER(FAR,FRR)[0] * 100)

    EER_local.append(EER_local_temp)
    EER_global.append(EER_global_temp)
    if rf:
        logger.info('random forgery:')
    else:
        logger.info('skilled forgery:')
    logger.info(f'global threshold: {np.mean(EER_global):.5f}')
    logger.info(f'local threshold: {np.mean(EER_local):.5f}')