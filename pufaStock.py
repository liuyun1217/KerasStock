from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from getData import excel2Pd
import matplotlib.pyplot as plt
inputFile = './data/沪深300指数.xlsx'
inputPd = excel2Pd(inputFile)


inputPd.replace('None',0)

# train_data = K.cast_to_floatx(inputPd[['收盘价(元)','最高价(元)','最低价(元)','成交额(百万)']][1:1999].values)
# train_targets = inputPd['收盘价(元)'][2:2000].values
#
# test_data = K.cast_to_floatx(inputPd[['收盘价(元)','最高价(元)','最低价(元)','成交额(百万)']][2001:2420].values)
# test_targets = inputPd['收盘价(元)'][2002:2421].values




def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(all_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model



k = 4
num_epochs = 100
#只需要填写all_data，程序会自动分割训练数据和测试数据
all_data = K.cast_to_floatx(inputPd[['收盘价(元)','最高价(元)','最低价(元)','成交额(百万)']][:-1].values)
all_targets = K.cast_to_floatx(inputPd[['收盘价(元)']][1:].values)

#输入需要预测数据的条件，会自动选择最好的模型来预测下一个交易日的收盘价
predict_data = K.cast_to_floatx(inputPd[['收盘价(元)','最高价(元)','最低价(元)','成交额(百万)']][-2:-1].values)

model = build_model()

num_val_samples = len(all_data) // k
all_scores = []
all_mae_histories = []
all_predict_res = {}
for i in range(k):
    print('model processing fold #', i)
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

    print(partial_train_data.shape)
    print(partial_train_targets.shape)
    print(val_data.shape)
    print(val_targets.shape)

    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=16, verbose=0,
                        validation_data=(val_data,val_targets))
    test_mse_score, test_mae_score = model.evaluate(val_data, val_targets)
    print("历史预测误差MAE：" + str(test_mae_score))
    print("历史预测均方差MSE：" + str(test_mse_score))
    mae_history = history.history['mae']
    all_mae_histories.append(mae_history)
    all_scores.append(test_mae_score)
    #break
    res = model.predict(predict_data)
    all_predict_res[str(res)] = test_mae_score

print(all_predict_res)
min_mae = min(all_predict_res.values())
print(min_mae)
best_predict_res = list(all_predict_res.keys())[list(all_predict_res.values()).index(min_mae)]
print(inputPd['日期'][-2:-1].astype(str)+"下一个交易日的预测收盘价： "+ str(best_predict_res + ', mae误差为：'+ str(min_mae)))
#
# average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# plt.plot(range(1, len(average_mae_history) +1),average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()