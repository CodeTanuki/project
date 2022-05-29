# XOR와 AND로 구성된 가산기를 만들어 보자

from Neural_XOR import *         # import SimpleNN

if __name__ == '__main__':

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])         # 입력값(2진 2게이트 입력)
    T = np.array([[0, 0], [0, 1], [0, 1], [1, 0]])         # 출력값(정답): [AND, XOR]
    N = X.shape[0]   # number of data                      # 입력 데이터 갯수: 4개
                                                           # X.shape=(4,2)
    input_size = X.shape[1]                                # 입력 노드수: 2개 
    hidden_size = 3                                        # 은닉층 노드수
    output_size = 2                                        # 출력 노드수
    alpha = 0.1                                            # 학습률
    delta = 0.5                                            # Momentum Coefficient 
    epoch = 10000                                          # 학습 횟수

    nn = Neural(input_size, hidden_size, output_size)      # Neural Class의 Instance
    nn.train(X, T, alpha, delta, epoch)                    # instance에 의한 method호출
    nn.error_graph()

    Y = nn.predict(X)

    for i in range(N):
        x = X[i, :]
        y = Y[i]

        print("Input : {}".format(x))           # 입력값       
        print("Output: {}".format(y))           # 앞부분은 AND값, 뒷부분은 XOR값

        for j in range(2):
            if y[j] > 0.5:                      # 출력값이 0.5 초과하면 1로 처리
                y[j]=1
            else:                               # 출력값이 0.5 이하이면 0로 처리
                y[j]=0  

        print("Output(Binary):{}".format(y))     # 2진 출력
        print("")