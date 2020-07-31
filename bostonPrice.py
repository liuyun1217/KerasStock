from keras.datasets import boston_housing
from keras import models
from keras import layers

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(type(train_data))
print(type(train_targets))

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# model = build_model()
# modelRes = model
# modelRes.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
# test_mse_score, test_mae_score = modelRes.evaluate(test_data, test_targets)
#
# print(test_mse_score)