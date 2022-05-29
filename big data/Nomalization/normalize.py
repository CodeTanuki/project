# 입력데이터는 임의의 숫자를 입력하여도 무방하나, 출력 데이터는 sigmoid를 거친 값이므로
# 0과 1사이의 값이다. 따라서 정답(teaching data)도 정규화 하여 0-1의 값으로 만들어야 한다. 

import numpy as np

def normalize(lst):
    normalized = []
    for value in lst:
        normal_num = (value - min(lst))/(max(lst) - min(lst))
        normalized.append(normal_num)
    return normalized

teaching_data = [[0, 0],   [30, 0],   [60, 0],   [90, 0],   [120, 0],   [150, 0],
                 [0, 30],  [30, 30],  [60, 30],  [90, 30],  [120, 30],  [150, 30],
                 [0, 60],  [30, 60],  [60, 60],  [90, 60],  [120, 60],  [150, 60],
                 [0, 90],  [30, 90],  [60, 90],  [90, 90],  [120, 90],  [150, 90],
                 [0, 120], [30, 120], [60, 120], [90, 120], [120, 120], [150, 120]]

data = []  # normalize하기위해 teaching_data를 옆으로 펼쳐, 1차원 배열 data를 만듦
for i in range(len(teaching_data)):        # len(teaching_data)=30
    for j in range(len(teaching_data[0])): # len(teaching_data[0])=2
        data.append(teaching_data[i][j])   # 1차원 배열의 데이터 [0,0, 30,0,....,150,120]

norm_data = normalize(data)  # 정규화 시킴
norm_data = np.array(norm_data)  # reshape를 이용하기 위해, numpy배열로 바꾸어 줌
norm_data = norm_data.reshape((30,2)) # 펼쳐진 배열을 다시 30X2(x,y)배열로 만듦
# norm_data = norm_data.reshape((-1,2)) # 위와 같은 결과가 나옴 
print(norm_data)