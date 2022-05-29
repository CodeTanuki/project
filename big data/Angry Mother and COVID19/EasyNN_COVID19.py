# Corona Virus 감염여부 예측
#  발열, 미후각, 기침, 가슴통증    감염(1), 비감염(0)    
#   1      0      0      1            1
#   1      0      0      0            1
#   0      0      1      1            0
#   0      1      0      0            0
#   1      1      0      0            1
#   0      0      1      1            1
#   0      0      0      1            0
#   0      0      1      0            0

import numpy as np

# Define input features :
input_data = np.array([[1,0,0,1],[1,0,0,0],[0,0,1,1],
                      [0,1,0,0],[1,1,0,0],[0,0,1,1],
                      [0,0,0,1],[0,0,1,0]])
                           # shape: 8X4
# Define target output :
target = np.array([[1,1,0,0,1,1,0,0]]) # 1X8
# Reshaping our target output into vector :
target = target.reshape(8,1)

# Define weights :
weights = np.array([[0.1],[0.2],[0.3],[0.4]])

# Bias weight :
bias = 0.3
# Learning Rate :
lr = 0.05

# Sigmoid function :
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid function :
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

####### Building a model for neural network #####
for epoch in range(10000):
    inputs = input_data
    # Forward input :
    cell_in = np.dot(inputs, weights) + bias
    # Forward output :
    cell_out = sigmoid(cell_in)

############### Backpropogation ################# 
    # Calculating error
    error = cell_out - target

    # Going with the formula
    x = error.sum()
    print(x)

    # Calculating derivative
    z = sigmoid_der(cell_out)

    # Multiplying individual derivatives
    # dE/dw = dE/dy * dy/dw = (y-t) * z  
    z_delta = error * z

    # Multiplying with the 3rd individual derivative
    inputs = input_data.T
    # w => w - alpha * dE/dw * x
    weights -= lr * np.dot(inputs, z_delta)

    # Updating the bias weight value
    # b => b - alpha * dE/db , dE/db = (y-t) * z
    for i in z_delta:
        bias -= lr * i

# Taking inputs
test_data = np.array([1,0,0,1])
# 1st step
result1 = np.dot(test_data, weights) + bias
# 2nd step
result2 = sigmoid(result1)
# Print final result
print("Fever, Pain --> Corona Virus ? : {} % ".format(result2*100))

# Taking inputs
test_data = np.array([0,0,1,0])
# 1st step
result1 = np.dot(test_data, weights) + bias
# 2nd step
result2 = sigmoid(result1)
# Print final result
print("Cough Only ---> Corona Virus ? : {} % ".format(result2*100))

# Taking inputs
test_data = np.array([1,0,1,0])
# 1st step
result1 = np.dot(test_data, weights) + bias
# 2nd step
result2 = sigmoid(result1)
# Print final result
print("Fever, Cough -> Corona Virus ? : {} % ".format(result2*100))