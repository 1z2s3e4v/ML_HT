import numpy as np
import random as rand

import numpy as np
from numpy.lib.function_base import average

def getDataSet(filename):
    dataSet = open(filename, 'r')
    dataSet = dataSet.readlines()
    num = len(dataSet)
    
    X = np.zeros((num,11))
    Y = np.zeros((num,1))
    for i in range(num):
        data = dataSet[i].strip().split()
        data = [np.float64(i) for i in data]
        X[i] = np.concatenate([[np.float64(1.0)], data[0:10]], axis=0)
        Y[i,0] = np.float64(data[10])
    return X,Y

def sign(x,w):
    if np.dot(x,w)[0] > 0:
        return 1
    else:
        return -1
        
def PLA(X, Y, w, speed, updates):
    iterations = 0
    num = len(X)
    flag = True
    for i in range(updates):
        flag = True
        count_rCheck = 0
        for j in range(5*num):
            r_idx = rand.randint(0,num-1)
            if sign(X[r_idx],w) != Y[r_idx,0]:
                flag = False
                w = w + speed*Y[r_idx,0]*np.matrix(X[r_idx]).T
                break
            else:
                continue
        if flag == True:
            iterations = i
            break
    return w, flag, iterations
    
filename = r"hw1_train.dat"
X, Y = getDataSet(filename)
#print("X: ", X, ", Y: ", Y)
w0 = np.zeros((11,1)) 
speed = 1
updates = 100000

w_list = []
for i in range(1000):
    w, flag, iterations = PLA(X, Y, w0, speed, updates)
    if flag == True:
        w_list.append(np.sum(np.square(w)))
    else:
        print("flag=F at iter[{}]".format(i))

w_avg = sum(w_list) / len(w_list)
print("|w_avg|^2 =", w_avg)

#print(flag)
#print(iterations)
#print(w)
