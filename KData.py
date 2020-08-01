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


inputFile = './data/沪深300指数.xlsx'
inputPd = excel2Pd(inputFile)
k = 4

inputPd.replace('None',0)
all_data = K.cast_to_floatx(inputPd[['收盘价(元)','最高价(元)','最低价(元)','成交额(百万)']][:-1].values)
all_targets = K.cast_to_floatx(inputPd[['收盘价(元)']][1:].values)
num_val_samples = len(all_data) // k

for i in range(k):
    print('processing fold #', i)
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

    print(partial_train_data[-1])
    print(partial_train_targets)
    print(val_data[-1])
    print(val_targets)
    break

# train_data = K.cast_to_floatx(inputPd[['收盘价(元)','最高价(元)','最低价(元)','成交额(百万)']][1:1999].values)
# train_targets = inputPd['收盘价(元)'][2:2000].values
#
# test_data = K.cast_to_floatx(inputPd[['收盘价(元)','最高价(元)','最低价(元)','成交额(百万)']][2001:2420].values)
# test_targets = inputPd['收盘价(元)'][2002:2421].values
#
# predict_data = K.cast_to_floatx(inputPd[['收盘价(元)','最高价(元)','最低价(元)','成交额(百万)']][2428:2429].values)


