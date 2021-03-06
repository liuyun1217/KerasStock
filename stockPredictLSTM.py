# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/6 10:22 PM
@Auth ： LiuYun ZhaoYing
@File ：stockPredictLSTM.py
@IDE ：PyCharm Community Edition

"""

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
from keras.layers import Dense, LSTM, BatchNormalization
from keras.layers import Embedding, SimpleRNN
from getStockDataUD import getUDData



def build_LSTMmodel(allDataShape):
    model = models.Sequential()
    #model.add(Embedding(2,32))
    model.add(LSTM(64, activation="tanh", return_sequences=True))
    model.add(LSTM(32, activation="tanh", return_sequences=True))
    #model.add(SimpleRNN(32))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',  # 还可以通过optimizer = optimizers.RMSprop(lr=0.001)来为优化器指定参数
                  loss='binary_crossentropy',  # 等价于loss = losses.binary_crossentropy
                  metrics=['accuracy'])  # 等价于metrics = [metircs.binary_accuracy]

    return model



if __name__ == '__main__':
    # 处理excel文件转化为pandas能处理的df格式
    inputFile = './data/沪深300指数.xlsx'
    inputPd = excel2Pd(inputFile)
    inputPd.replace('None', 0)
    inputPd.replace('True',1)
    inputPd.replace('False', 0)
    UDData = getUDData(inputPd)

    #模型保存的路径
    models_save_path = './models'

    #下面是ZhaoYing修改添加训练数据的地方----------------------------------------------------------------------------------
    #需要指定数据里的最新时间
    new_time = np.datetime64('2020-08-05')
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
    all_data = K.cast_to_floatx(UDData[col_data].loc[UDData['日期']<new_time].values)[:-predict_day]
    all_targets = K.cast_to_floatx(UDData[['UD']].loc[UDData['日期']<new_time].values)[predict_day:]
    #预测数据，会自动选择最好的模型来预测下一个交易日的收盘价
    predict_data = K.cast_to_floatx(UDData[col_data].loc[UDData['日期']==new_time].values)
    predict_data = np.reshape(predict_data, (predict_data.shape[0], predict_data.shape[1], 1))


    allDataShape = [all_data.shape[1],]
    model = build_LSTMmodel(allDataShape)

    # 使用KData.py里的K-折线方法切割数据为多组训练数据和测试数据
    k = 4
    k_partial_train_data, k_partial_train_targets, k_val_data, k_val_targets = getKData(all_data, all_targets, k)

    # 开始训练
    all_scores = []
    all_acc_histories = []
    all_dict_histories = []
    predict_res = {}
    all_predict_res = []
    for i in range(1, k):
        print('processing fold #', i)
        partial_train_data = k_partial_train_data[i]
        partial_train_targets = k_partial_train_targets[i]
        val_data = k_val_data[i]
        val_targets = k_val_targets[i]
        print('本模型的：训练数据量%d，测试数据量%d' % (len(partial_train_targets), len(val_targets)))
        partial_train_data = np.reshape(partial_train_data, (partial_train_data.shape[0], partial_train_data.shape[1], 1))
        val_data = np.reshape(val_data,(val_data.shape[0], val_data.shape[1], 1))
        print(partial_train_data.shape)
        history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=8, verbose=0,
                            validation_data=(val_data, val_targets))
        test_bct_score, test_acc_score = model.evaluate(val_data, val_targets)
        print("历史预测精确度：" + str(test_acc_score))
        model_name = models_save_path + '/' + str(new_time) + '_ACC' + str(float(test_acc_score)) + '.h5'
        model.save(model_name)
        acc_history = history.history['val_accuracy']
        history_dict = history.history
        all_dict_histories.append(history_dict)
        all_acc_histories.append(acc_history)
        all_scores.append(test_acc_score)
        # break
        res = np.argmax(model.predict(predict_data), axis=1)
        predict_res[str(res)] = test_acc_score
        all_predict_res.append(predict_res)

    # 找出表现最好（mape最小）的模型
    #max_acc = max(all_predict_res.values())
    print(all_predict_res)
    # 算出acc的平均值，用于画图展示模型训练的趋势
    #mean_acc = np.mean(all_scores)
    # print(all_scores)
    #print('K次训练模型精确度平均值为 %f' % mean_acc)
    #print('挑选出最好的模型精确度为 %f' % max_acc)
    # 用表现最好的模型去预测下一日的收盘价
    #best_predict_res = list(all_predict_res.keys())[list(all_predict_res.values()).index(max_acc)]
    print('该模型预测的' + inputPd['日期'].loc[inputPd['日期'] == new_time].astype(str) + "下" + str(
        predict_day) + "个交易日的涨跌为： " + str(all_predict_res))
    print(inputPd['日期'].loc[inputPd['日期'] == new_time].astype(str) + "的收盘价： " + inputPd['沪深300'].loc[
        inputPd['日期'] == new_time].astype(str))

    acc_values = [np.mean([x['accuracy'][i] for x in all_dict_histories]) for i in range(num_epochs)]
    val_acc_values = [np.mean([x['val_accuracy'][i] for x in all_dict_histories]) for i in range(num_epochs)]
    plt.clf()
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()
