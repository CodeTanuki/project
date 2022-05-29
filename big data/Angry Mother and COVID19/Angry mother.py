# 2X3X1의 인공 신경망을 만들어 보자. 
# 입력부에 1을 추가하여 bias대체
# 출력부 활성화 함수로 항등함수(identity function)사용

# 게임, 공부할 때의 엄마의 기분을 학습해 보자. 
# 게임시간    공부시간       엄마의 기분(y)  
#    0           0           -1 (살짝 나쁨) 
#    0           1            3 (아주 좋음)
#    1           0           -3 (아주 나쁨)
#    1           1            1 (살짝 좋음)
#    0.5        0.5              ??
#   게임도 공부도 평소보다 덜 했을 때 엄마의 기분은 과연? 

import numpy as np
from random import random

alpha = 0.3   # 학습률(learning rate)
epoch = 1000

# 가중치의 초기화 (initializing of weight) 
wt = []    # 빈 list를 만들고, 나중에 append를 통해 추가 

for i in range(13):  # 2X3X1에서 총 13개의 가중치값이 필요
    w = np.random.rand()
    wt.append(w)

# sigmoid 활성화 함수
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

# 입력(input)값과 정답(teaching data)
input_data = np.array([[0,0], [0,1], [1,0], [1,1]])
teaching_data = np.array([[-1], [3], [-3], [1]])

# 입력값과 정답을 통해 학습을 시작
for n in range(1, epoch+1): # 1부터 epoch까지 반복
    for i in range(len(input_data)): 
        x1 = input_data[i][0]   # i번째 행의 첫번째 숫자
        x2 = input_data[i][1]   # i번째 행의 두번째 숫자
        t  = teaching_data[i]   # i번째 행의 숫자
        ########## 순방향 계산 #########
        u1 = sigmoid(wt[0]*x1 + wt[3]*x2 + wt[6])
        u2 = sigmoid(wt[1]*x1 + wt[4]*x2 + wt[7])
        u3 = sigmoid(wt[2]*x1 + wt[5]*x2 + wt[8])
        y  = wt[9]*u1 + wt[10]*u2 + wt[11]*u3 + wt[12]
        ######## 역방향 계산 (오차 역전파) ########
        E = 0.5 * (y - t)**2
        dE_dw_0 = (y-t)*wt[9]* (1-u1)*u1*x1
        dE_dw_1 = (y-t)*wt[10]*(1-u2)*u2*x1
        dE_dw_2 = (y-t)*wt[11]*(1-u3)*u3*x1
        dE_dw_3 = (y-t)*wt[9]* (1-u1)*u1*x2
        dE_dw_4 = (y-t)*wt[10]*(1-u2)*u2*x2
        dE_dw_5 = (y-t)*wt[11]*(1-u3)*u3*x2
        dE_dw_6 = (y-t)*wt[9]* (1-u1)*u1
        dE_dw_7 = (y-t)*wt[10]*(1-u2)*u2 
        dE_dw_8 = (y-t)*wt[11]*(1-u3)*u3 
        dE_dw_9 =  (y-t)*u1 
        dE_dw_10 = (y-t)*u2 
        dE_dw_11 = (y-t)*u3
        dE_dw_12 = (y-t)
        ########## 가중치 업데이트 (경사하강법) #########
        wt[0] = wt[0] - alpha * dE_dw_0
        wt[1] = wt[1] - alpha * dE_dw_1
        wt[2] = wt[2] - alpha * dE_dw_2
        wt[3] = wt[3] - alpha * dE_dw_3
        wt[4] = wt[4] - alpha * dE_dw_4
        wt[5] = wt[5] - alpha * dE_dw_5
        wt[6] = wt[6] - alpha * dE_dw_6
        wt[7] = wt[7] - alpha * dE_dw_7
        wt[8] = wt[8] - alpha * dE_dw_8
        wt[9] = wt[9] - alpha * dE_dw_9
        wt[10] = wt[10] - alpha * dE_dw_10
        wt[11] = wt[11] - alpha * dE_dw_11
        wt[12] = wt[12] - alpha * dE_dw_12

    print("{} EPOCH-ERROR: {}".format(n, E))

# Test: 입력값 x에 대하여 본 신경망으로 예측(순방향 계산)
x1 = 0.5
x2 = 0.8
u1 = sigmoid(wt[0]*x1 + wt[3]*x2 + wt[6])
u2 = sigmoid(wt[1]*x1 + wt[4]*x2 + wt[7])
u3 = sigmoid(wt[2]*x1 + wt[5]*x2 + wt[8])
y  = wt[9]*u1 + wt[10]*u2 + wt[11]*u3 + wt[12]
print("엄마의 기분은???")
print("게임:{}시간, 공부:{}시간 --> 엄마기분:{}".format(x1, x2, y))
print("")