# 뉴런의 갯수가 증가하면 어떤 효과가 있는지 알아보자. 

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def neuron(x, w, b):
   return sigmoid(w * x - b)

x = np.arange(-5, 5, 0.1) 

# Red Line: weight=2, bias=-2
y = neuron(x, 2, -2) 
plt.plot(x, y, color="r", label='w:2.0, b:-2') 

# Blue Line: weight=0.5, bias=1
y = neuron(x, 0.5, 1)
plt.plot(x, y, color="b", label='w:0.5, b:1') 

# Orange Line: weight=2, bias=4
y = neuron(x, 2, 4) 
plt.plot(x, y, color="orange", label='w:2.0, b:4') 

# Green Line = Red + Orange +Blue
y = neuron(x, 2, -2) + neuron(x, 0.5, 1) + neuron(x, 2, 4) 
plt.plot(x, y, "g", label='Red + Orange +Blue') 

plt.ylim(-0.5, 3)
plt.legend(loc='upper left')
plt.show()