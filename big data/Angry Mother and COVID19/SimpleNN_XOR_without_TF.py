# 곱셈 뉴럴넷을 변경하여 XOR문제용으로 만들어 보자. 
# 본 프로그램의 뉴럴넷은 바이어스를 0으로 설정하였다.

import numpy as np
from random import random 

# save activations and derivatives 
# implement backpropagation
# implement gradient descent
# implement train
# train a neural net with some datasets
# make some prediction

class NeuralNet(object):  # constructor for the NN (each numbers are arbitrary)
    def __init__(self, num_inputs=3, hidden_layers=[3,3], num_outputs=2):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        
        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]
    
        # create random connection weights for the layers
        weights = []    # set weights with initial values
        for i in range(len(layers)-1):  # No need weights in the input layer
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        activations = []   # set vacant activations temporarily
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []   # set vacant derivatives temporarily
        for i in range(len(layers)-1):  # No need derivatives in the input layer
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):  # compute forward propagation of the NN
        # the input layer activation is just the input itself
        activations = inputs
        self.activations[0] = inputs  # activations of input layer is just the inputs

        # iterate through the NN layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)
            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
            # example:  a_3 = s(h_3) ,  h_3 = a_2 * W_2 

        # return output layer activation 
        return activations

    def back_propagate(self, error, verbose=False):
        # dE/dW_i = (y - a_[i+1]) s'(h_[i+1]) a_i
        # s'(h_[i+1]) = s(h_[i+1])(1-s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1]

        # dE/dW_[i-1] = (y - a_[i+1]) s'(h_[i+1]) W_i s'(h_i) a_[i-1]
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations) # ndarray([0.1, 0.2]) -> ndarray([[0.1, 0.2]])
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i] # ndarray([0.1, 0.2]) -> ndarray([0.1], [0.2])
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}:{}".format(i, self.derivatives[i]))

        return error
    
    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            # print("Original W{} {}".format(i, weights))
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate
            # print("Updated W{} {}".format(i, weights))


    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):

                    # forward propagation
                    output = NN.forward_propagate(input)

                    # calcaulate error
                    error = target - output

                    # back propagation
                    self.back_propagate(error)

                    # apply gradient descent
                    self.gradient_descent(learning_rate)

                    sum_error += self._mse(target, output)

            # report error
            print("Error: {} at epoch {}".format(sum_error/len(inputs), i))

    def _mse(self, target, output):
        return np.average((target - output)**2)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _sigmoid(self, x):
        y = 1.0/(1 + np.exp(-x))
        return y

if __name__ == "__main__":
    
    # create a dataset to train a network for the sum operation
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]]) 
    
    # creat a NeuralNet
    NN = NeuralNet(2, [10], 1) # input=2, hidden=10, output=1
    
    # train a NeuralNet
    NN.train(inputs, targets,15000, 0.1)  # epochs=15000, learning_rate=0.1

    # create test data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])         # 입력값(2진 2게이트 입력)

    for i in range(4):
        x = X[i, :]       # 테스트 데이터 처음부터 끝까지
        Y = NN.forward_propagate(X)
        y = Y[i]          # NN 순방향 계산된 값 (즉, 예측값) 
        print("{} XOR {} = {}".format(x[0], x[1], y))           
