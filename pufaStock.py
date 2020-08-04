# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/1 9:15 AM
@Auth ： LiuYun ZhaoYing
@File ：pufaStock.py
@IDE ：PyCharm Community Edition

"""
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from getData import excel2Pd
import matplotlib.pyplot as plt
from KData import getKData
import datetime


#定义模型的结构
def build_model(allDataShape):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(allDataShape)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.compile(optimizer='rmsprop', loss='mae', metrics=['mae','mape','mse'])
    return model

if __name__ == '__main__':
    # 处理excel文件转化为pandas能处理的df格式
    inputFile = './data/沪深300指数.xlsx'
    inputPd = excel2Pd(inputFile)
    inputPd.replace('None', 0)
    inputPd.replace('True',1)
    inputPd.replace('False', 0)

    #下面是ZhaoYing修改添加训练数据的地方----------------------------------------------------------------------------------
    #需要指定数据里的最新时间
    new_time = np.datetime64('2020-08-03')
    #指定用于训练的列名
    col_data = ['市盈率', '市盈率(TTM)', '市净率', '开盘价', '最高价', '最低价', '前收盘价', '涨跌幅', '振幅', '换手率', '指数成分上涨数量', '指数成分下跌数量',
                '近期创阶段新高', '近期创阶段新低', '连涨天数', '连跌天数', '向上有效突破5日均线', '向下有效突破5日均线', '向上有效突破10日均线', '向下有效突破10日均线',
                '向上有效突破20日均线', '向下有效突破20日均线', '向上有效突破60日均线', '向下有效突破60日均线', '均线多头排列', '均线空头排列', 'BBI多空指数', 'DMI趋向指标',
                'DMA平均线差', 'MACD', 'TRIX三重指数平滑平均', 'KDJ', 'RSI', 'VROC量变动速率', 'ARBR人气意愿指标', 'PSY心理指标', 'VR成交量比率',
                'MFI资金流向指标', '多空布林线', '量比']
    #训练的轮数，先用1轮来跑通程序，然后改成10，50，100甚至更多来让训练更准确（也更慢）,目前最多设置过500，模型还在优化还能更多
    num_epochs = 10
    #预测的几天后的数据
    predict_day = 1
    #上面是ZhaoYing修改添加训练数据的地方----------------------------------------------------------------------------------


    #程序会自动分割训练数据和测试数据
    all_data = K.cast_to_floatx(inputPd[col_data].loc[inputPd['日期']<new_time].values)[:-predict_day]
    all_targets = K.cast_to_floatx(inputPd[['沪深300']].loc[inputPd['日期']<new_time].values)[predict_day:]
    #预测数据，会自动选择最好的模型来预测下一个交易日的收盘价
    predict_data = K.cast_to_floatx(inputPd[col_data].loc[inputPd['日期']==new_time].values)



    allDataShape = [all_data.shape[1],]
    #构建模型网络结构
    model = build_model(allDataShape)

    #使用KData.py里的K-折线方法切割数据为多组训练数据和测试数据
    k = 4
    k_partial_train_data,k_partial_train_targets,k_val_data,k_val_targets = getKData(all_data,all_targets,k)

    #开始训练
    all_scores = []
    all_mae_histories = []
    all_predict_res = {}
    for i in range(1,k):
        print('processing fold #', i)
        partial_train_data  = k_partial_train_data[i]
        partial_train_targets = k_partial_train_targets[i]
        val_data = k_val_data[i]
        val_targets = k_val_targets[i]
        print('本模型的：训练数据量%d，测试数据量%d' %(len(partial_train_targets),len(val_targets)))

        history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=8, verbose=0,
                            validation_data=(val_data,val_targets))
        t,test_mae_score, test_mape_score,test_mse_score = model.evaluate(val_data, val_targets)
        print("历史预测误差MAE：" + str(test_mae_score))
        print("历史预测误差百分比MAPE：" + str(test_mape_score))
        print("历史预测均方差MSE：" + str(test_mse_score))
        mae_history = history.history['mae']
        all_mae_histories.append(mae_history)
        all_scores.append(test_mae_score)
        #break
        res = model.predict(predict_data)
        all_predict_res[str(res)] = test_mae_score

    #找出表现最好（mape最小）的模型
    min_mae = min(all_predict_res.values())
    #算出mae的平均值，用于画图展示模型训练的趋势
    mean_mae = np.mean(all_scores)
    #print(all_scores)
    print('K次训练模型误差百分比MAE平均值为 %f' % mean_mae)
    print('挑选出最好的模型误差MAE为 %f' %min_mae)
    #用表现最好的模型去预测下一日的收盘价
    best_predict_res = list(all_predict_res.keys())[list(all_predict_res.values()).index(min_mae)]
    print('该模型预测的'+inputPd['日期'].loc[inputPd['日期']==new_time].astype(str)+"下"+str(predict_day)+"个交易日的收盘价： "+ str(best_predict_res))
    print(inputPd['日期'].loc[inputPd['日期']==new_time].astype(str)+"的收盘价： "+ inputPd['沪深300'].loc[inputPd['日期']==new_time].astype(str))


    #
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    plt.plot(range(1, len(average_mae_history) +1),average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()

#测试git push忽略文件夹