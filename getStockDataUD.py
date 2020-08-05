# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/5 10:34 PM
@Auth ： LiuYun ZhaoYing
@File ：getStockDataUD.py
@IDE ：PyCharm Community Edition

"""
from getData import excel2Pd


def getUDData(inputPd):
    lastPrice = 0
    inputPd['UD'] = 0
    #print(inputPd[['日期','沪深300']])
    for i in inputPd[['日期','沪深300']].values:
        if i[1] > lastPrice:
            inputPd['UD'].loc[inputPd['日期']==i[0]] = 1
        elif i[1] <= lastPrice:
            inputPd['UD'].loc[inputPd['日期'] == i[0]] = 0
        lastDay = i[0]
        lastPrice = i[1]
    return inputPd

if __name__ == '__main__':
    inputFile = './data/沪深300指数.xlsx'
    inputPd = excel2Pd(inputFile)
    inputPd.replace('None', 0)
    inputPd.replace('True', 1)
    inputPd.replace('False', 0)

    UDData = getUDData(inputPd)
    print(UDData['UD'])
