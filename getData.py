# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/1 9:15 AM
@Auth ： LiuYun ZhaoYing
@File ：getData.py
@IDE ：PyCharm Community Edition

"""

import pandas as pd
import matplotlib.pyplot as plt
import logging
def use_logging(func):
    def wrapper(*args, **kwargs):
        logging.warning("%s is running" % func.__name__)
        return func(*args, **kwargs)
    return wrapper

@use_logging
def excel2Pd(input):
    inputFile = input
    inputPd = pd.read_excel(inputFile)
    print(str(inputFile)+' all data shape is '+str(inputPd.shape))
    return inputPd

@use_logging
def cav2Pd(input):
    inputFile = input
    inputPd = pd.read_csv(inputFile,encoding='gbk')
    print((inputFile)+' all data shape is '+ str(inputPd.shape))
    return inputPd
@use_logging
def plotData(inputPd):
    plt.plot(inputPd['日期'],inputPd['收盘价(元)'])
    plt.xlabel('date')
    plt.ylabel('price')
    plt.show()



if __name__ == '__main__':
    inputFile = './data/沪深300指数.xlsx'
    resPd = excel2Pd(inputFile)
    print(resPd.shape)