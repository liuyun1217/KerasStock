from keras.datasets import boston_housing
from keras import models
from keras import layers

import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from getData import excel2Pd

inputFile = './data/沪深300指数.xlsx'
inputPd = excel2Pd(inputFile)


inputPd.replace('None',0)

train_data = K.cast_to_floatx(inputPd[['收盘价(元)','最高价(元)','最低价(元)','成交额(百万)']][1:1999].values)
train_targets = inputPd['收盘价(元)'][2:2000].values

test_data = K.cast_to_floatx(inputPd[['收盘价(元)','最高价(元)','最低价(元)','成交额(百万)']][2001:2420].values)
test_targets = inputPd['收盘价(元)'][2002:2421].values

predict_data = K.cast_to_floatx(inputPd[['收盘价(元)','最高价(元)','最低价(元)','成交额(百万)']][2428:2429].values)


print(train_data.shape)
#print(test_data)

#
# plt.plot(inputPd['日期'][501:1500],train_data)
# plt.xlabel('date')
# plt.ylabel('换手率')
# plt.show()

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


model = build_model()
modelRes = model
modelRes.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = modelRes.evaluate(test_data, test_targets)
print("历史预测误差MAE：" + str(test_mae_score))
print("历史预测均方差MSE：" + str(test_mse_score))

res = modelRes.predict(predict_data)
print(str(inputPd['日期'][2429])+"下一个交易日的收盘价： "+ str(res[0]))
# git test
#test liuyun