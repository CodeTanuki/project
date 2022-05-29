# tensorflow2.0을 써서 간단한 Neural Network를 구성해 보자. (두개의 수를 더한 값을 추측하는 뉴럴넷)
# datasets -> build (model) -> compile -> train and validate -> predict 

import numpy as np          # py -m pip install numpy
from random import random   # py -m pip install random
from sklearn.model_selection import train_test_split  # Datasets을 학숩용과 테스트용으로 분리 
# Scikit-learn(사이킷런)은 머신러닝용 다양한 함수 및 데이터셋 제공, 참조: Keggle, UCI...
# py -m pip install scikit-learn
import tensorflow as tf     # py -m pip install tensorflow==2.0
from keras import layers, models   # py -m pip install keras==2.3.1
import matplotlib.pyplot as plt

############################  Datasets 준비 ############################
# x(입력):   0에서 0.5사이의 랜덤값을 가진 num_samplesX2의 배열을 만듦
# y(출력):   배열 각행의 0번째, 1번째 column값을 더함
# 예) x: array([[0.4, 0.3], [0.1, 0.2]])  각각 더하면 ->  y:  array([0.7, 0.3])
# Datasets중에서 test_size값(비율, 0.2, 0.3..)으로 테스트셋을 만듦, 나머지는 학습용
def generate_dataset(num_samples, test_size):
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)]) 
    y = np.array([[i[0] + i[1]] for i in x])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.2)

############################# Build Model  #############################
# NN model구축 input:2, hidden:5, output:1 
model = models.Sequential()
model.add(layers.Dense(5, input_dim=2, activation="sigmoid"))
model.add(layers.Dense(1, activation="sigmoid"))

############################# Compile Model #############################
model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.1), loss="mean_squared_error")
model.summary()

######################### Train and Validate Model #######################
# verbose 0:silent, 1:progress bar, 2:one line per epoch
# model.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_test, y_test))
hist = model.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_test, y_test))

######################### Loss Function Plotting  ########################
fig, loss_fn = plt.subplots()

loss_fn.plot(hist.history['loss'], 'y', label='train loss')
loss_fn.set_xlabel('epoch')
loss_fn.set_ylabel('loss')
loss_fn.legend(loc='upper left')

plt.show()

############################# Make Prediction ############################
data = np.array([[0.4, 0.3], [0.1, 0.2]])
predictions = model.predict(data)
print("\nPredictions:")
for d, p in zip(data, predictions):
    print("{} + {} = {}".format(d[0], d[1], p[0]))