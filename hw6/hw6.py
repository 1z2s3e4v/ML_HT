import os
import sys
import random
import numpy as np
import math

def Average(lst):
	return sum(lst) / len(lst)
def sign(x):
	if (x>=0):
		return 1
	else:
		return 0

class Data:
	def __init__(self):
		self.X = [] # Xn = [0-9]
		self.Y = [] # Yn = +1/-1
		self.Trans = [] # Xn = [0-]
		self.Trans_p15 = []
	def add_data(self,data):
		self.X.append(data[0:6]) 
		self.Y.append(data[6])
	def read(self, fileName):
		f = open(fileName)
		lines = f.readlines()
		for line in lines:
			data = list(map(float, line.split()))
			self.add_data(data)
	def print_info(self):
		count = 0
		for (x, y) in zip(self.X, self.Y):
			print("x[%s]=(%s), y[%s]=%s" % (count,x,count,y))
			count += 1

'''def decision_strump(X,Y):
    global N
    N = len(Y)
    best = 99999
    index = 0
    theta = 0
    for i in range(10):
        I = [0,0,0,0,0,0,0,0,0,0]
        I[i] = 1
        x1 = np.array(X).dot(np.array(I))
        feature = sorted([[x1[k]]+[Y[k]] for k in range(len(Y))], key=lemda x:x[0])
        feature_x = np.array(feature).dot([1,0])
        feature_y = np.array(feature).dot([0,1])
        P = best_para(feature_x, feature_y, possible_theta(feature_x))
        if(P[0]<best):
            best = P[0]
            index = i
            theta = P[1]
    return (best, index, theta)
'''



def main():
    train_data = Data()
    train_data.read("./hw6_train.dat");
    test_data = Data()
    test_data.read("./hw6_test.dat");


if __name__ == "__main__":
	main()
