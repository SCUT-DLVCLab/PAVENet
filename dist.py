# -*- coding:utf-8 -*-

import numpy as np

def dist_skilled_forgery(features,template_num,nf,genuine_num,forgery_num):
    dist_positive,dist_negative,dist_template = [],[],[]
    for i,feat in enumerate(features):
        # feat_a = feat[0:template_num]
        # feat_p = feat[(template_num + nf):(genuine_num + nf)]

        l = len(feat)
        first = template_num // 2 # 第一轮拿这么多个，偶数拿一半，奇数拿少一个
        second = template_num - first # 第二轮拿这么多个
        feat_a = feat[0:first]
        feat_a = np.concatenate((feat_a,feat[l // 4:l // 4 + second]),axis=0)
        feat_p = np.concatenate((feat[first:l // 4],feat[l // 4 + second:l // 2]),axis=0)
        
        feat_n = feat[(genuine_num + nf):]
        dist_p = np.zeros((genuine_num - template_num,template_num))
        dist_n = np.zeros((forgery_num,template_num))
        dist_t = np.zeros((template_num,template_num))
        for i in range(template_num):
            fa1 = feat_a[i]
            for j in range(i + 1,template_num):
                fa2 = feat_a[j]
                dist = np.sqrt(np.sum(np.power(fa2 - fa1,2),axis=0))
                dist_t[i,j] = dist # 对角线为0，上三角矩阵
        for i in range(genuine_num - template_num):
            fp = feat_p[i]
            for j in range(template_num):
                fa = feat_a[j]
                dist = np.sqrt(np.sum(np.power(fp - fa,2),axis=0))
                dist_p[i,j] = dist
        for i in range(forgery_num):
            fn = feat_n[i]
            for j in range(template_num):
                fa = feat_a[j]
                dist = np.sqrt(np.sum(np.power(fn - fa,2),axis=0))
                dist_n[i,j] = dist
        dist_positive.append(dist_p)
        dist_negative.append(dist_n)
        dist_template.append(dist_t)
    dist_positive = np.concatenate(dist_positive,axis=0)
    dist_negative = np.concatenate(dist_negative,axis=0)
    dist_template = np.concatenate(dist_template,axis=0)
    return dist_positive,dist_negative,dist_template

def dist_random_forgery(features,template_num,nf,genuine_num,forgery_num):
    dist_positive,dist_negative,dist_template = [],[],[]
    features_anchor,features_positive = [],[]
    for i,feat in enumerate(features):
        # feat_a = feat[0:template_num]
        # feat_p = feat[(template_num + nf):(genuine_num + nf)]

        l = len(feat)
        first = template_num // 2 # 第一轮拿这么多个，偶数拿一半，奇数拿少一个
        second = template_num - first # 第二轮拿这么多个
        feat_a = feat[0:first]
        feat_a = np.concatenate((feat_a,feat[l // 4:l // 4 + second]),axis=0)
        feat_p = np.concatenate((feat[first:l // 4],feat[l // 4 + second:l // 2]),axis=0)
        
        features_anchor.append(feat_a)
        features_positive.append(feat_p)
    for i,feat_a in enumerate(features_anchor): # i是user的下标
        feat_p = features_positive[i] # 这里是会变的，主要改变的是这里
        feat_n = []
        for j in range(len(features_anchor)): # j也是user的下标，一个user有4个anchor16个genuine
            if i != j:
                feat_n.append(features_positive[j][3]) # 随便选个genuine而已，不能只选一轮的genuine啊
        dist_p = np.zeros((len(feat_p),template_num))
        dist_n = np.zeros((len(feat_n),template_num))
        dist_t = np.zeros((template_num,template_num))
        for j in range(template_num):
            fa1 = feat_a[j] # (640,)
            for k in range(j + 1,template_num):
                fa2 = feat_a[k] # anchor与anchor之间
                dist = np.sqrt(np.sum(np.power(fa2 - fa1,2),axis=0))
                dist_t[j,k] = dist # 对角线为0，上三角矩阵
        for j in range(len(feat_p)):
            fp = feat_p[j]
            for k in range(template_num):
                fa = feat_a[k] # anchor与anchor之间
                dist = np.sqrt(np.sum(np.power(fp - fa,2),axis=0))
                dist_p[j,k] = dist # 每个user都有4个anchor，每个anchor与每个positive的距离
        for j in range(len(feat_n)):
            fn = feat_n[j] # (16,640)
            for k in range(template_num):
                fa = feat_a[k] # anchor与negative之间
                # dist = np.mean(np.sqrt(np.sum(np.power(fn - np.expand_dims(fa,axis=0),2),axis=1)),axis=0)
                dist = np.sqrt(np.sum(np.power(fn - fa,2),axis=0))
                dist_n[j,k] = dist # 每个user都有4个anchor，每个anchor与每个positive的距离
        dist_positive.append(dist_p)
        dist_negative.append(dist_n)
        dist_template.append(dist_t)
    dist_positive = np.concatenate(dist_positive,axis=0)
    dist_negative = np.concatenate(dist_negative,axis=0)
    dist_template = np.concatenate(dist_template,axis=0)
    return dist_positive,dist_negative,dist_template
