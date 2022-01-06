import os
import sys
import random
import numpy as np
import math

from liblinear.liblinearutil import *

def Average(lst):
	return sum(lst) / len(lst)
def sign(x):
	if (x>=1/2):
		return 1
	else:
		return 0
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
def Phi_3(x):
	data = Phi_2(x)
	for i in range(len(x)):
		for j in range(i,len(x)):
			data.append(x[i]*x[i]*x[j])
	return data
def E(X,Y,W):
	Sum=0
	for i in range(len(X)):
		h = 1 / (1+math.exp(-np.array(W).dot(X[i])))
		if(sign(h) != Y[i]):
			Sum += 1
	return Sum/len(X)

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
	def print_trans_info(self):
		count = 0
		for (x, y, t) in zip(self.X, self.Y, self.Trans):
			print("x[%s]=(%s), y[%s]=%s" % (count,x,count,y))
			print("trans[%s]=(%s)" % (count,t))
			count += 1
	def transform(self):
		self.Trans = []
		for x in self.X:
			self.Trans.append(Phi_3(x))
	def write_LIBSVM(self, fileName):
		f = open(fileName,"w")
		for x,y in zip(self.Trans, self.Y):
			line = str(int(y>0))
			for i in range(len(x)):
				line += (" %d:%f" % (i+1,x[i]))
			f.write(line+"\n")

def dat2libsvm(in_dat, out_libsvm):
	train_data = Data()
	train_data.read(in_dat);
	train_data.transform() # Phi_3(train.x)
	train_data.write_LIBSVM(out_libsvm)
	#train_data.print_info()

def run_p12(train_libsvm, test_libsvm):
	lambda_list = [0.0001, 0.01, 1, 100, 10000] # log10(_lambda) = [-4,-2,0,2,4]
	train_y, train_x = svm_read_problem(train_libsvm)
	test_y, test_x = svm_read_problem(test_libsvm)
	p_acc_list = []
	for _lambda in lambda_list:
		print("------------------------------------------------")
		print("lambda* =", _lambda)
		C = 1/(2*_lambda) # lambda=0.0001 -> C=5000, lambda=0.01 -> C=50, ...
		print("C =",C)
		option = "-s 0" +  " -c " + str(C) + " -e 0.000001"
		m = train(train_y, train_x, option)
		p_label, p_acc, p_val = predict(test_y, test_x, m)
		p_acc_list.append(p_acc)
	best_lambda = round(math.log(lambda_list[p_acc_list.index(max(p_acc_list))], 10))
	print("=> Best lambda:", best_lambda)
	print("=> Max Acc:", max(p_acc_list))
def run_p13(train_libsvm, test_libsvm):
	lambda_list = [0.0001, 0.01, 1, 100, 10000] # log10(_lambda) = [-4,-2,0,2,4]
	train_y, train_x = svm_read_problem(train_libsvm)
	test_y, test_x = svm_read_problem(test_libsvm)
	p_acc_list = []
	for _lambda in lambda_list:
		print("------------------------------------------------")
		print("lambda* =", _lambda)
		C = 1/(2*_lambda) # lambda=0.0001 -> C=5000, lambda=0.01 -> C=50, ...
		print("C =",C)
		option = "-s 0" +  " -c " + str(C) + " -e 0.000001"
		m = train(train_y, train_x, option)
		p_label, p_acc, p_val = predict(train_y, train_x, m)
		p_acc_list.append(p_acc)
	best_lambda = round(math.log(lambda_list[p_acc_list.index(max(p_acc_list))],10))
	print("=> Best lambda:", best_lambda)
	print("=> Max Acc:", max(p_acc_list))
def run_p14(train_libsvm, test_libsvm):
	lambda_list = [0.0001, 0.01, 1, 100, 10000] # log10(_lambda) = [-4,-2,0,2,4]
	train_y, train_x = svm_read_problem(train_libsvm)
	val_y, val_x = (train_y[120:200], train_x[120:200])
	train_y, train_x = (train_y[:120], train_x[:120])

	p_acc_list = []
	for _lambda in lambda_list:
		print("------------------------------------------------")
		print("lambda* =", _lambda)
		C = 1/(2*_lambda) # lambda=0.0001 -> C=5000, lambda=0.01 -> C=50, ...
		print("C =",C)
		option = "-s 0" +  " -c " + str(C) + " -e 0.000001"
		m = train(train_y, train_x, option)
		p_label, p_acc, p_val = predict(val_y, val_x, m)
		p_acc_list.append(p_acc)
	best_lambda = round(math.log(lambda_list[p_acc_list.index(max(p_acc_list))],10))
	print("=> Best lambda:", best_lambda)
	print("=> Max Acc:", max(p_acc_list))
def run_p15(train_libsvm, test_libsvm):
	_lambda = 100 # log10(_lambda) = 2
	train_y, train_x = svm_read_problem(train_libsvm)
	test_y, test_x = svm_read_problem(test_libsvm)

	print("lambda* =", _lambda)
	C = 1/(2*_lambda) # lambda=0.0001 -> C=5000, lambda=0.01 -> C=50, ...
	print("C =",C)
	option = "-s 0" +  " -c " + str(C) + " -e 0.000001"
	m = train(train_y, train_x, option)
	p_label, p_acc, p_val = predict(test_y, test_x, m)
def run_p16(train_libsvm, test_libsvm):
	lambda_list = [0.0001, 0.01, 1, 100, 10000] # log10(_lambda) = [-4,-2,0,2,4]
	train_y, train_x = svm_read_problem(train_libsvm)
	D_5fold_y = [train_y[40*i:40*(i+1)] for i in range(5)]
	D_5fold_x = [train_x[40*i:40*(i+1)] for i in range(5)]
	train_5fold_y = [train_y[40:200], train_y[0:40]+train_y[80:200], train_y[0:80]+train_y[120:200], 
					train_y[0:120]+train_y[160:200], train_y[0:160]]
	train_5fold_x = [train_x[40:200], train_x[0:40]+train_x[80:200], train_x[0:80]+train_x[120:200], 
					train_x[0:120]+train_x[160:200], train_x[0:160]]
	p_acc_list = []
	for _lambda in lambda_list:
		print("------------------------------------------------")
		print("lambda* =", _lambda)
		C = 1/(2*_lambda) # lambda=0.0001 -> C=5000, lambda=0.01 -> C=50, ...
		print("C =",C)
		option = "-s 0" +  " -c " + str(C) + " -e 0.000001"
		acc_sum = 0
		for i in range(5):	
			m = train(train_5fold_y[i], train_5fold_x[i], option)
			[W,b] = m.get_decfun()
			p_label, p_acc, p_val = predict(D_5fold_y[i], D_5fold_x[i], m)
			acc, e, cal = p_acc
			acc_sum += acc
		p_acc_list.append(acc_sum/5/100)
		print("acc =", acc_sum/5/100)
	best_lambda = round(math.log(lambda_list[p_acc_list.index(max(p_acc_list))], 10))
	print("=> Best lambda:", best_lambda)
	print("=> Max Acc:", max(p_acc_list))

def main():
	# data processing
	dat2libsvm("./hw4_train.dat", "./hw4_train.libsvm")
	dat2libsvm("./hw4_test.dat", "./hw4_test.libsvm")
	
	run_p16("./hw4_train.libsvm", "./hw4_test.libsvm")

if __name__ == "__main__":
	main()
