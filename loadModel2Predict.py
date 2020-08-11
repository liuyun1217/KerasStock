# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/5 9:46 PM
@Auth ： LiuYun ZhaoYing
@File ：loadModel2Predict.py
@IDE ：PyCharm Community Edition

"""

from keras.models import load_model
import numpy as np
import os
from getData import excel2Pd
from keras import backend as K
import re

if __name__ == '__main__':
    models_save_path = './models'

    # #需要指定模型的训练时间
    # model_time = np.datetime64('2020-08-07')
    # model_name_pre = str(model_time)+'_MAE'

    # 下面是ZhaoYing修改添加训练数据的地方----------------------------------------------------------------------------------
    # 需要指定数据里的最新时间
    new_time = np.datetime64('2020-08-10')
    #选择MAE小于多少的模型进行预测
    model_mae = 40
    # 上面是ZhaoYing修改添加训练数据的地方----------------------------------------------------------------------------------

    model_nmaeMae = '_MAE'+str(model_mae)
    model_list = [f for f in os.listdir(models_save_path) if f.endswith('.h5') and 'MAE' in f]

    inputFile = './data/沪深300指数.xlsx'
    inputPd = excel2Pd(inputFile)
    inputPd.replace('None', 0)
    inputPd.replace('True', 1)
    inputPd.replace('False', 0)

    # 模型保存的路径
    models_save_path = './models'


    # 指定用于训练的列名
    col_data = ['市盈率', '市盈率(TTM)', '市净率', '开盘价', '最高价', '最低价', '前收盘价', '涨跌幅', '振幅', '换手率', '指数成分上涨数量', '指数成分下跌数量',
                '近期创阶段新高', '近期创阶段新低', '连涨天数', '连跌天数', '向上有效突破5日均线', '向下有效突破5日均线', '向上有效突破10日均线', '向下有效突破10日均线',
                '向上有效突破20日均线', '向下有效突破20日均线', '向上有效突破60日均线', '向下有效突破60日均线', '均线多头排列', '均线空头排列', 'BBI多空指数', 'DMI趋向指标',
                'DMA平均线差', 'MACD', 'TRIX三重指数平滑平均', 'KDJ', 'RSI', 'VROC量变动速率', 'ARBR人气意愿指标', 'PSY心理指标', 'VR成交量比率',
                'MFI资金流向指标', '多空布林线', '量比']
    # 预测数据，会自动选择最好的模型来预测下一个交易日的收盘价
    predict_data = K.cast_to_floatx(inputPd[col_data].loc[inputPd['日期'] == new_time].values)
    all_predict_res = {}

    for m in model_list:
        thisModelMae = int(re.split('_MAE|.h5',m)[1])
        #if model_name_pre in m:
        if model_mae > thisModelMae:
            model_name = models_save_path + '/' + m
            model = load_model(model_name)
            #print("Using loaded model %s to predict..." %m)
            res = model.predict(predict_data)
            all_predict_res[m] = res
    for k in all_predict_res.keys():
        print(str(k)+':'+inputPd['日期'].loc[inputPd['日期']==new_time].astype(str)+'下1日: '+str(all_predict_res[k]))
    print(inputPd['日期'].loc[inputPd['日期'] == new_time].astype(str) + "的收盘价： " + inputPd['沪深300'].loc[
        inputPd['日期'] == new_time].astype(str))



