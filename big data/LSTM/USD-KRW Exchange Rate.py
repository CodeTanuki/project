# LSTM에 의한 환율 예측 프로그램, 원-달러화 환율 사이트에서 퍼왔음
# https://kr.investing.com/currencies/usd-krw-historical-data
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler   # py -m pip install sklearn
from sklearn.metrics import mean_squared_error
# pandas를 이용해 데이터를 읽어와서 데이터셋을 만듦
# csv의 데이터는 불필요한 줄을 지우고, 급격한 변화를 줄이기 위해 자연대수로 처리(ln)
dataframe = pandas.read_csv('USD_KRW rate change.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32') 
# 재현성 확보를 위해 seed설정(fix random seed for reproducibility)
np.random.seed(7)    # 난수발생을 위한 시드, 시드가 같으면 같은 난수가 발생됨
# 데이터셋의 정규화 (normalize the dataset) 0과 1사이의 숫자로 바꾸어 줌
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset) 
# 데이터셋을 트레인셋과 테스터셋으로 분리(split into train and test sets)
train_size = int(len(dataset) * 0.7)        # 전체 데이터의 앞쪽 70%를 훈련용 데이터셋으로 설정 
test_size = len(dataset) - train_size       # 나머지 뒷부분 구간을 테스트용 데이터셋으로 설정
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# print(len(train), len(test))  # 트레인셋과 테스터셋의 길이를 확인해 보자

# 데이터셋 배열로 변환 (convert an array of values into a dataset matrix)
def create_dataset(dataset, maxlen):    # maxlen은 다음 시간영역을 예측하기 위한 앞쪽 시간대의 스텝수
    dataX, dataY = [], []               # maxlen=3이면 X는 t-2, t-1, t의 환율, Y는 t+1의 환율
    for i in range(len(dataset)-maxlen-1):
        a = dataset[i:(i+maxlen), 0]
        dataX.append(a)
        dataY.append(dataset[i + maxlen, 0])
    return np.array(dataX), np.array(dataY)
 
# reshape into X=t and Y=t+maxlen
maxlen = 3   # 스텝수를 20정도로 늘리면 어떻게 바뀔까?
trainX, trainY = create_dataset(train, maxlen)
testX, testY = create_dataset(test, maxlen)
 
print (trainX[:10,:])
print (trainY[:10])
 
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX[:10,:])    # 데이터셋의 형태가 어떻게 바뀌었는지 확인해 보자

# create and fit the LSTM network
# 1개의 입력(가시층), 4개의 LSTM블럭을 가지는 은닉층, 단일치 예측을 하는 출력층(Dense(1))
model = Sequential()
model.add(LSTM(4, input_shape=(1, maxlen)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam') # 디폴트 활성화 함수: Sigmoid
model.fit(trainX, trainY, epochs=30, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
# 예측된 데이터를 원래의 배열형태로 바꾸고, inverse_transform()으로 오차계산을 함
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate RMSE(root mean squared error) 
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
 
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[maxlen:len(trainPredict)+maxlen, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(maxlen*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), color ="g", label = "row")
plt.plot(trainPredictPlot,color="b", label="trainpredict")
plt.plot(testPredictPlot,color="m", label="testpredict")
plt.title('Prediction with USD-KRW Exchange Rate') 
plt.legend()
plt.show()