import os
import sys
import random
import numpy as np
import math

from libsvm.svmutil import *

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


def run_p16(train_libsvm, test_libsvm):
    g_list = [0.1, 1, 10, 100, 1000]
    g_select_time = {0.1:0, 1:0, 10:0, 100:0, 1000:0}
    progress = 0
    for i in range(1000):
        train_y, train_x = svm_read_problem(train_libsvm)
        train_y = [int(y==1) for y in train_y]
        val_y = []
        val_x = []
        for j in range(200):
            r = random.randint(0, len(train_y)-1)
            val_y.append(train_y[r])
            val_x.append(train_x[r])
            del train_y[r]
            del train_x[r]
            
        best_gamma = 0.0
        best_Eval = 100
        prob = svm_problem(train_y, train_x)
        for g in g_list:
            m = svm_train(prob, "-t 2 -c 0.1 -g {}".format(g))
            p_label, p_acc, p_val = svm_predict(val_y, val_x, m)
            acc, mse, u = p_acc
            if(best_Eval > 1-acc*0.01):
                best_Eval = 1-acc*0.01
                best_gamma = g
        g_select_time[best_gamma] += 1
        progress += 1
        print("progress = {}/1000".format(progress))
        print("---------------------------------------------------------------")

    for i in range(5):
        print("gamma={}: {}".format(g_list[i], g_select_time[g_list[i]]))

def run_p15(train_libsvm, test_libsvm):
    train_y, train_x = svm_read_problem(train_libsvm)
    train_y = [int(y==1) for y in train_y]
    test_y, test_x = svm_read_problem(test_libsvm)
    test_y = [int(y==1) for y in test_y]
    
    prob = svm_problem(train_y, train_x)
    g_list = [0.1, 1, 10, 100, 1000]
    for g in g_list:
        m = svm_train(prob, "-t 2 -c 0.1 -g {}".format(g))
        p_label, p_acc, p_val = svm_predict(test_y, test_x, m)
        acc, mse, u = p_acc
        print("gamma = {}".format(g))
        print("E_out = {}".format(1-acc*0.01))
        print("--------------------------------------------------------------")

def run_p14(train_libsvm, test_libsvm):
    train_y, train_x = svm_read_problem(train_libsvm)
    train_y = [int(y==1) for y in train_y]
    test_y, test_x = svm_read_problem(test_libsvm)
    test_y = [int(y==1) for y in test_y]
    
    prob = svm_problem(train_y, train_x)
    C_list = [0.01, 0.1, 1, 10, 100]
    for C in C_list:
        m = svm_train(prob, "-t 2 -c {} -g 10".format(C))
        p_label, p_acc, p_val = svm_predict(test_y, test_x, m)
        acc, mse, u = p_acc
        print("C = {}".format(C))
        print("E_out = {}".format(1-acc*0.01))
        print("--------------------------------------------------------------")


def run_p1213(train_libsvm, test_libsvm):
    for i in range(1,7): # 1 ~ 6
        train_y, train_x = svm_read_problem(train_libsvm)
        train_y = [int(y==i) for y in train_y]
        prob = svm_problem(train_y, train_x)
        m = svm_train(prob, "-t 1 -c 10 -g 1 -d 3 -r 1")
        p_label, p_acc, p_val = svm_predict(train_y, train_x, m)
        acc, mse, u = p_acc
        print("\'%d\' vs \'not %d\'" % (i, i))
        print("E_in = %f" % (1-acc*0.01))
        print("# of sv = %d" % (m.get_nr_sv()))
        print("---------------------------------------------------------------")

def run_p11(train_libsvm, test_libsvm):
    train_y, train_x = svm_read_problem(train_libsvm)
    train_y = [int(y==5) for y in train_y]
    prob = svm_problem(train_y, train_x)
    m = svm_train(prob, "-t 0 -c 10")
    p_label, p_acc, p_val = svm_predict(train_y, train_x, m)
    ACC, MSE, SCC = evaluations(train_y, p_label)
    alpha = m.get_sv_coef() # support_vector_coefficients
    alpha = [alpha[i][0] for i in range(len(alpha))]
    support_vectors = m.get_SV()
    # fill the missing data
    sv_val = []
    for i in range(len(support_vectors)):
        for idx in range(1,37): # 1~36
            if(idx not in support_vectors[i]):
                support_vectors[i][idx] = 0
        sv_val.append([support_vectors[i][idx] for idx in range(1,37)])
    
    w = np.array(alpha).dot(sv_val)
    norm_w = np.sqrt(w.transpose().dot(w))
    print("|w| = %f" % (norm_w))
        
def main():
    run_p16("satimage.scale", "satimage.scale.t")

if __name__ == "__main__":
	main()
