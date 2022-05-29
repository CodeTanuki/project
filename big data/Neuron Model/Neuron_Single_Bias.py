# Neuron에 있어 weight와 bias가 미치는 영향을 살펴보자

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def neuron(x, w, b):
   return sigmoid(w * x - b)

x = np.arange(-5, 5, 0.1) 

# Red Line: weight=1, bias=0
y = neuron(x, 1, 0)
plt.plot(x, y, color="r", label='w:1, b:0') 

# Blue Line: weight=1, bias=-1
y = neuron(x, 1, -1)
plt.plot(x, y, color="b", label='w:1, b:-1')

# Green Line: weight=1, bias=1
y = neuron(x, 1, 1)
plt.plot(x, y, color="g", label='w:1, b:1') 

plt.ylim(-0.5, 1.5) 
plt.legend(loc='upper left')
plt.show()