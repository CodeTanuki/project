# 2X3X2의 NN를 만들고, 2개 관절각에 대한 끝점위치 x, y를 계산해 보자. 
# 위치값과 관절각을 정규화(nomalize)하여 표현해보자. 
# Min-Max Nomalization (최소-최대 정규화)를 사용 
# x = (x-min)/(max-min)

import numpy as np
from random import random

alpha = 1.0   # learning rate
epoch = 10000

# initializing of weight and bias 
wt = []    # vacant array for weights
bs = []    # vacant array for bias
for i in range(12):  # 2X3X2에서는 총 12개의 가중치값이 필요
    w = np.random.rand()
    wt.append(w)
for i in range(5):   # 2X3X2에서는 총 5개의 bias값이 필요
    w = np.random.rand()
    bs.append(w)

# sigmoid activation function 
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

def normalize(lst):
    normalized = []
    for value in lst:
        normal_num = (value - min(lst))/(max(lst) - min(lst))
        normalized.append(normal_num)
    return normalized

# Kinematics calculation
L1 = 1       # 첫번째 Link의 길이 (theta1과 theta2사이의 거리)
L2 = 2       # 두번째 Link의 길이 (theta2와 끝점 사이의 거리)
# 각도(degree)를 radian으로 바꾸기 위해 pi/180을 곱했음. 1 degree = π radian/180
def kinematics(theta1, theta2):
    x = L1 * np.cos(theta1*np.pi/1.8) + L2 * np.cos(theta2*np.pi/1.8 - theta1*np.pi/1.8)
    y = L1 * np.cos(theta1*np.pi/1.8) - L2 * np.cos(theta2*np.pi/1.8 - theta1*np.pi/1.8)
    return x, y

# input and teaching data
# 앞부분 theta1의 각도(0-60degree), 뒷부분 theta2의 각도 (0-30degree)
input_data = [[0, 0],   [0.3, 0],    [0.6, 0],
              [0, 0.3], [0.3, 0.3],  [0.6, 0.3]]

xy = []      # 끝점의 x, y좌표에 대한 vacant list를 만듦
for th1, th2 in input_data:      
    x, y = kinematics(th1, th2)  # 운동학 계산, th1: theta1, th2:theta2
    xy.append(x)               # 끝점의 x좌표를 더해 감
    xy.append(y)               # 끝점의 y좌표를 더해 감

norm_xy = normalize(xy)      # 정규화 시킴
norm_xy = np.array(norm_xy)  # reshape를 이용하기 위해, numpy배열로 바꾸어 줌
teaching_data = norm_xy.reshape((-1,2)) # 펼쳐진 배열을 다시 30X2(x,y)배열로 만듦
print(teaching_data)

# train with input and teaching data
for n in range(1, epoch+1): # 1부터 epoch까지 반복
    for i in range(len(input_data)): 
        x1 = input_data[i][0]   # i번쨰 행의 첫번째 숫자
        x2 = input_data[i][1]   # i번쨰 행의 두번째 숫자
        t1 = teaching_data[i][0]
        t2 = teaching_data[i][1]
        ########## forward #########
        u1 = sigmoid(wt[0]*x1 + wt[1]*x2 + bs[0])
        u2 = sigmoid(wt[2]*x1 + wt[3]*x2 + bs[1])
        u3 = sigmoid(wt[4]*x1 + wt[5]*x2 + bs[2])
        y1 = sigmoid(wt[6]*u1 + wt[7]*u2 + wt[8]*u3 + bs[3])
        y2 = sigmoid(wt[9]*u1 + wt[10]*u2 + wt[11]*u3 + bs[4])
        ########## backward ########
        E = 0.5 * (y1 - t1)**2 + 0.5 * (y2 - t2)**2  
        dE_dw_0 = ((y1-t1)*(1-y1)*y1*wt[6] + (y2-t2)*(1-y2)*y2*wt[9])* (1-u1)*u1*x1
        dE_dw_1 = ((y1-t1)*(1-y1)*y1*wt[7] + (y2-t2)*(1-y2)*y2*wt[10])*(1-u2)*u2*x1
        dE_dw_2 = ((y1-t1)*(1-y1)*y1*wt[8] + (y2-t2)*(1-y2)*y2*wt[11])*(1-u3)*u3*x1
        dE_dw_3 = ((y1-t1)*(1-y1)*y1*wt[6] + (y2-t2)*(1-y2)*y2*wt[9])* (1-u1)*u1*x2
        dE_dw_4 = ((y1-t1)*(1-y1)*y1*wt[7] + (y2-t2)*(1-y2)*y2*wt[10])*(1-u2)*u2*x2
        dE_dw_5 = ((y1-t1)*(1-y1)*y1*wt[8] + (y2-t2)*(1-y2)*y2*wt[11])*(1-u3)*u3*x2
        dE_dw_6 =  (y1-t1)*(1-y1)*y1*u1  # error * s'(out_put) * hidden_output
        dE_dw_7 =  (y1-t1)*(1-y1)*y1*u2  # delta rule: w=w+alpha*delta*input
        dE_dw_8 =  (y1-t1)*(1-y1)*y1*u3  # delta=error*(1-y)y=(y-t)(1-y)y
        dE_dw_9 =  (y2-t2)*(1-y2)*y2*u1 
        dE_dw_10 = (y2-t2)*(1-y2)*y2*u2 
        dE_dw_11 = (y2-t2)*(1-y2)*y2*u3

        dE_db_0 = ((y1-t1)*(1-y1)*y1*wt[6] + (y2-t2)*(1-y2)*y2*wt[9])*  (1-u1)*u1 
        dE_db_1 = ((y1-t1)*(1-y1)*y1*wt[7] + (y2-t2)*(1-y2)*y2*wt[10])* (1-u2)*u2
        dE_db_2 = ((y1-t1)*(1-y1)*y1*wt[8] + (y2-t2)*(1-y2)*y2*wt[11])* (1-u3)*u3
        dE_db_3 = (y1-t1)*(1-y1)*y1
        dE_db_4 = (y2-t2)*(1-y2)*y2
        ########## weight&bias update by BP #########
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
        bs[0] = bs[0] - alpha * dE_db_0
        bs[1] = bs[1] - alpha * dE_db_1
        bs[2] = bs[2] - alpha * dE_db_2
        bs[3] = bs[3] - alpha * dE_db_3
        bs[4] = bs[4] - alpha * dE_db_4

    print("{} EPOCH-ERROR: {}".format(n, E))

# Test: 입력값 x에 대하여 본 뉴럴넷으로 예측된 값과 정답값 비교
th1 = 0.6
th2 = 0.2
u1 = sigmoid(wt[0]*x1 + wt[1]*x2 + bs[0])
u2 = sigmoid(wt[2]*x1 + wt[3]*x2 + bs[1])
u3 = sigmoid(wt[4]*x1 + wt[5]*x2 + bs[2])
y1 = sigmoid(wt[6]*u1 + wt[7]*u2 + wt[8]*u3 + bs[3])
y2 = sigmoid(wt[9]*u1 + wt[10]*u2 + wt[11]*u3 + bs[4])
print("Forward Kinematics for a robot with 2 DoF")
print("Angle:[{}, {}] --> End Point: [{}, {}]".format(th1, th2, y1, y2))
print("")