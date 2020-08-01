# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/1 9:15 AM
@Auth ： LiuYun ZhaoYing
@File ：KData.py
@IDE ：PyCharm Community Edition

"""
from getData import excel2Pd
from keras import backend as K
import pandas as pd
import numpy as np




def getKData(all_data,all_targets,k):
    num_val_samples = len(all_data) // k
    k_train_data = []
    k_train_targets = []
    k_val_data = []
    k_val_targets = []
    for i in range(k):

        val_data = all_data[i*num_val_samples: (i+1) * num_val_samples]
        val_targets = all_targets[i*num_val_samples: (i+1) * num_val_samples]

        partial_train_data = np.concatenate(
            [all_data[:i * num_val_samples],
            all_data[(i+1)  * num_val_samples:]],
        axis=0
        )

        partial_train_targets = np.concatenate(
            [all_targets[:i * num_val_samples],
            all_targets[(i+1)  * num_val_samples:]],
        axis=0
        )

        k_train_data.append(partial_train_data)
        k_train_targets.append(partial_train_targets)
        k_val_data.append(val_data)
        k_val_targets.append(val_targets)
    return k_train_data,k_train_targets,k_val_data,k_val_targets


if __name__ == '__main__':
    inputFile = './data/沪深300指数.xlsx'
    inputPd = excel2Pd(inputFile)
    k = 4

    inputPd.replace('None', 0)
    all_data = K.cast_to_floatx(inputPd[['收盘价(元)', '最高价(元)', '最低价(元)', '成交额(百万)']][:-1].values)
    all_targets = K.cast_to_floatx(inputPd[['收盘价(元)']][1:].values)

    partial_train_data, partial_train_targets, val_data, val_targets = getKData(all_data, all_targets, k)
    print(val_targets)
    for i in range(k):
        print(partial_train_data[i].shape)
        print(partial_train_targets[i].shape)
        print(val_data[i].shape)
        print(val_targets[i].shape)
