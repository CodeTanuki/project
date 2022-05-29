# 일반화된 3층 뉴럴넷의 프로그램을 이용하여 운동학을 풀어보자

import numpy as np
import math
import random
import matplotlib.pyplot as plt

class NeuralNetwork:    # 뉴럴넷 클래스를 정의
    # constructor  # 초기 weights값 설정
    def __init__(self, n_input, n_hidden, n_output):   # 1이 더해진 것은 bias를 의미함
        self.hidden_weight = np.random.random_sample((n_hidden, n_input + 1))  
        self.output_weight = np.random.random_sample((n_output, n_hidden + 1)) 

# X: input, T:teaching(정답), alpha:learning rate, epoch: No of iteration for train
    def train(self, X, T, alpha, epoch):  
        self.error = np.zeros(epoch)
        N = X.shape[0]                          # N:입력 갯수
        for epo in range(epoch):
            for i in range(N):
                x = X[i, :]                     # x: 입력값 처음부터 끝까지
                t = T[i, :]                     # t: 출력값 처음부터 끝까지
                self.__update_weight(x, t, alpha)
            self.error[epo] = self.__calc_error(X, T)

    def predict(self, X):  # 학습이 끝난 후, forward를 하면 자동적으로 에측이 됨
        N = X.shape[0]
        Y = np.zeros((N, X.shape[1]))
        for i in range(N):
            x = X[i, :]
            z, y = self.__forward(x)

            Y[i] = y
        return Y

    def error_graph(self):  # 학습이 진행됨(epoch)에 따른 에러값 변화를 가시화
        plt.ylim(0.0, 2.0)
        plt.plot(np.arange(0, self.error.shape[0]), self.error)
        plt.show()

# define sigmoid activation function
    def __sigmoid(self, arr): # sigmoid함수를 lamda fn으로 표현
        return np.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(arr)

    def __forward(self, x):  # 가중치와 입력값을 곱해서 더해가며 출력값을 계산함
        z = self.__sigmoid(self.hidden_weight.dot(np.r_[np.array([1]), x])) # hidden layer 출력 
        y = self.__sigmoid(self.output_weight.dot(np.r_[np.array([1]), z])) # output layer 출력
        return (z, y)

    def __update_weight(self, x, t, alpha):   # 이 부분이 가장 중요. 델타의 법칙을 익히자. 
        z, y = self.__forward(x)    # 최종 출력값

        # update output_weight
        output_delta = (y - t) * y * (1.0 - y)     # output_delta = output_errors * final_output * (1-final_output)
        _output_weight = self.output_weight        
        self.output_weight -= alpha * output_delta.reshape((-1, 1)) * np.r_[np.array([1]), z]
        # self.output_weight += alpha * np.dot(output_delta, np.transpose(z))  로 표현해도 됨

        # update hidden_weight
        hidden_delta = (self.output_weight[:, 1:].T.dot(output_delta)) * z * (1.0 - z)
        # hidden_delta = hidden_errors * hidden_outputs * (1-hidden_outputs)
        _hidden_weight = self.hidden_weight
        self.hidden_weight -= alpha * hidden_delta.reshape((-1, 1)) * np.r_[np.array([1]), x]
        # self.hidden_weight += alpha * np.dot(hidden_delta, np.transpose(x))  로 표현해도 됨

    def __calc_error(self, X, T):
        N = X.shape[0]
        err = 0.0
        for i in range(N):
            x = X[i, :]
            t = T[i, :]

            z, y = self.__forward(x)
            err += (y - t).dot((y - t).reshape((-1, 1))) / 2.0              
        return err

# Test: Forward Kinematics용 학습데이터 마련 및 훈련, 예측
if __name__ == '__main__':

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
    # 각도는 1/100 스케일로 표시하였음
    X = np.array([[0, 0],    [0.3, 0],    [0.6, 0],    [0.9, 0],
                  [0, 0.3],  [0.3, 0.3],  [0.6, 0.3],  [0.9, 0.3],
                  [0, 0.6],  [0.3, 0.6],  [0.6, 0.6],  [0.9, 0.6],
                  [0, 0.9],  [0.3, 0.9],  [0.6, 0.9],  [0.9, 0.9]])

    xy = []      # 끝점의 x, y좌표에 대한 vacant list를 만듦
    for th1, th2 in X:      
        _x, _y = kinematics(th1, th2)  # 운동학 계산, th1: theta1, th2:theta2
        xy.append(_x)                  # 끝점의 x좌표를 더해 감
        xy.append(_y)                  # 끝점의 y좌표를 더해 감

    norm_xy = normalize(xy)      # 정규화 시킴
    norm_xy = np.array(norm_xy)  # reshape를 이용하기 위해, numpy배열로 바꾸어 줌
    T = norm_xy.reshape((-1,2))  # 펼쳐진 배열을 다시 mX2(x,y)배열로 만들어 teaching값으로 씀

    N = X.shape[0]   # number of data                      # 입력 데이터 갯수: 16개
    input_size = X.shape[1]                                # 입력 노드수: 2개 
    hidden_size = 6                                        # 은닉층 노드수
    output_size = 2                                        # 출력 노드수
    alpha = 0.5                                            # 학습률
    epoch = 3000                                           # 학습 횟수

    nn = NeuralNetwork(input_size, hidden_size, output_size) # NeuralNetwork Class의 Instance
    nn.train(X, T, alpha, epoch)                             # instance에 의한 method호출
    nn.error_graph()                                         # epoch에 따른 오차값 그래프

    Y = nn.predict(X)

    for i in range(N):
        x = X[i, :]                             # 입력값을 처음부터 끝까지
        y = Y[i]                                # 계산된 출력값을 가지고 옴
        t = T[i]                                # 실제 운동학으로 계산된 값

        print("Input : {}".format(x))           # 입력값(theta1/100, theta2/100)      
        print("Output: {}".format(y))           # 예측 출력값(x좌표,y좌표) 
        print("Real_V: {}".format(t))           # 실제 계산값(x좌표,y좌표)
        print("Error : {}".format(y-t))         # 예측값과 계산값 차이
        print("")