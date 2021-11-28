import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
import random

def Average(lst):
	return sum(lst) / len(lst)
def sign(x):
	return (x>=0)

def Phi(x,Q):
	data = [1]
	for q in range(1,Q+1):
		data += [xi**q for xi in x]
	return data
def Phi_2(x):
	data = [1] + x
	for i in range(len(x)):
		for j in range(i,len(x)):
			data.append(x[i]*x[j])
	return data

class Data:
	def __init__(self):
		self.X = [] # Xn = [0-9]
		self.Y = [] # Yn = +1/-1
		self.Trans = [] # Xn = [0-]
		self.Trans_p15 = []
	def add_data(self,data):
		self.X.append(data[0:10]) 
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
	def transform_p14(self):
		self.Trans = []
		for x in self.X:
			self.Trans.append(Phi_2(x))
	def transform_p15(self):
		self.Trans = []
		for i in range(10):
			phi_i = []
			for x in self.X:
				phi_i.append([1]+x[0:i+1])
			self.Trans.append(phi_i)
	def transform_p16(self):
		self.Trans = []
		dim_list = [1,2,3,4,5,6,7,8,9,10]
		r = random.sample(dim_list, 5)
		r = sorted(r)
		for x in self.X:
			self.Trans.append([1] + [x[i-1] for i in r])


def cal_Err(y,ans_y):
	err = 0.0
	for yi,ans_yi in zip(y,ans_y):
		if sign(yi) == sign(ans_yi):
			err += 1.0
	return err/len(y)

def run_p12(train_data, test_data):
	# train
	Q = 2
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
	return abs(E_in-E_out)

def run_p13(train_data, test_data):
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
	return abs(E_in-E_out)
def run_p14(train_data, test_data):
	# train
	train_data.transform_p14()
	lr = LinearRegression().fit(np.array(train_data.Trans), np.array(train_data.Y))
	E_in = cal_Err(lr.predict(np.array(train_data.Trans)).tolist(), train_data.Y)
	print("E_in = %s" % E_in)
	# test
	test_data.transform_p14()
	test_predict = lr.predict(np.array(test_data.Trans)).tolist()
	E_out = cal_Err(test_predict,test_data.Y)
	print("E_out = %s" % E_out)
	print("|E_in - E_out| = %s" % abs(E_in-E_out))
	return abs(E_in-E_out)

def run_p15(train_data, test_data):
	# train
	train_data.transform_p15()
	lr = [LinearRegression().fit(np.array(train_data.Trans[i]), np.array(train_data.Y)) for i in range(10)]
	E_in = [cal_Err(lr[i].predict(np.array(train_data.Trans[i])).tolist(), train_data.Y) for i in range(10)]
	print("E_in = %s" % E_in)
	# test
	test_data.transform_p15()
	test_predict = [lr[i].predict(np.array(test_data.Trans[i])).tolist() for i in range(10)]
	E_out = [cal_Err(test_predict[i],test_data.Y) for i in range(10)]
	print("E_out = %s" % E_out)
	E_diff = [abs(E_in[i]-E_out[i]) for i in range(10)]
	print("min |E_in - E_out| = %s in index=%s" % (min(E_diff), E_diff.index(min(E_diff))))
	return min(E_diff)
def run_p16(train_data, test_data):
	E_diff = []
	for i in range(200):
		# train
		train_data.transform_p16()
		lr = LinearRegression().fit(np.array(train_data.Trans), np.array(train_data.Y))
		E_in = cal_Err(lr.predict(np.array(train_data.Trans)).tolist(), train_data.Y)
		# test
		test_data.transform_p16()
		test_predict = lr.predict(np.array(test_data.Trans)).tolist()
		E_out = cal_Err(test_predict,test_data.Y)
		E_diff.append(abs(E_in-E_out))
	
	print("avg |E_in - E_out| = %s" % Average(E_diff))
	return Average(E_diff)


def main():
	train_data = Data()
	train_data.read("./hw3_train.dat");
	#train_data.print_info()
	test_data = Data()
	test_data.read("./hw3_test.dat");
	#test_data.print_info()
	
	#run_p12(train_data, test_data)
	#run_p13(train_data, test_data)
	run_p14(train_data, test_data)
	#run_p15(train_data, test_data)
	#run_p16(train_data, test_data)

if __name__ == "__main__":
	main()
