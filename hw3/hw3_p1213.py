import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression

def sign(x):
	return (x>=0)

def Phi(x,Q):
	data = [1]
	for q in range(1,Q+1):
		data += [xi**q for xi in x]
	return data

class Data:
	def __init__(self):
		self.X = [] # Xn = [0-9]
		self.Trans = [] # Xn = [0-]
		self.Y = [] # Yn = +1/-1
	def add_data(self,data):
		self.X.append(data[0:9]) 
		self.Y.append(data[10])
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
	def print_info2(self):
		count = 0
		for (x, y, t) in zip(self.X, self.Y, self.Trans):
			print("x[%s]=(%s), y[%s]=%s" % (count,x,count,y))
			print("trans[%s]=(%s)" % (count,t))
			count += 1
	def transform(self,Q):
		self.Trans = []
		for x in self.X:
			self.Trans.append(Phi(x,Q))

def cal_Err(y,ans_y):
	err = 0.0
	for yi,ans_yi in zip(y,ans_y):
		if sign(yi) == sign(ans_yi):
			err += 1.0
	return err/len(y)

def main():
	train_data = Data()
	train_data.read("./hw3_train.dat");
	#train_data.print_info()
	test_data = Data()
	test_data.read("./hw3_test.dat");
	#test_data.print_info()
	
	# train
	Q = 8
	train_data.transform(Q)
	lr = LinearRegression().fit(np.array(train_data.Trans), np.array(train_data.Y))
	E_in = cal_Err(lr.predict(np.array(train_data.Trans)).tolist(), train_data.Y)
	print("E_in = %s" % E_in)
	# test
	test_data.transform(Q)
	test_predict = lr.predict(np.array(test_data.Trans)).tolist()
	E_out = cal_Err(test_predict,test_data.Y)
	print("E_out = %s" % E_out)

	print("|E_in - E_out| = %s" % abs(E_in-E_out))


if __name__ == "__main__":
	main()
