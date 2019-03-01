# -*- coding: UTF-8 -*-

import numpy as np
import os

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

path = '/Users/chaoyi/Desktop/python'
training_sample = 'training_sample.txt'
testing_sample = 'testing_sample.txt'


def loadDataSet(path, file_name):
	dataList = []; labelList = []
	file = open(os.path.join(path, file_name))
	for line in file.readlines():
		lineArr = line.strip().split()
		dataList.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelList.append(int(lineArr[2]))
	return dataList, labelList

def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))

def predict(w, x):
	z = x * w
	return sigmoid(z)

# 打印交叉熵
def print_cross_entropy_error(weights, dataMatrix, labelMatrix):
	resultMat = predict(weights, dataMatrix)
	y = np.array(resultMat.T)[0]
	t = np.array(labelMatrix.T)[0]
	print('==============交叉熵==============') 
	print(cross_entropy_error(y, t))

# 交叉熵
def cross_entropy_error(y, t):
	delta = 1e-7
	return -sum(t * np.log(y + delta))

# 递度下降算法
def gradAscent(weights, dataMatrix, labelMatrix, maxCycles):
	for k in range(maxCycles):
		y = predict(weights, dataMatrix)

		error = y - labelMatrix

		alpha = 0.001 
		# 交叉熵相对于weights的偏导数
		temp = dataMatrix.transpose() * error

		weights = weights - alpha * temp
	return weights

def test_logistic_regression():
	dataList, labelList = loadDataSet(path, training_sample)

	dataMatrix = np.mat(dataList)
	labelMatrix = np.mat(labelList).transpose()

	m, n = np.shape(dataMatrix)
	weights = np.ones((n, 1))

	print('============初始weights==============')
	print(weights)

	print_cross_entropy_error(weights, dataMatrix, labelMatrix)

	weights = gradAscent(weights, dataMatrix, labelMatrix, 200)

	print('============学习后weights==============')
	print(weights)

	print_cross_entropy_error(weights, dataMatrix, labelMatrix)

	plotBestFit(weights)

def plotBestFit(weights):

	fig = plt.figure()
	ax = fig.add_subplot(121)
	drawAX(weights, ax, plt, training_sample)

	ax = fig.add_subplot(122)
	drawAX(weights, ax, plt, testing_sample)

	plt.show()

def drawAX(weights, ax, plt, file_name):
	dataList, labelList = loadDataSet(path, file_name)

	m, n = np.shape(dataList)

	x1 = []; y1 = []
	x2 = []; y2 = []

	for i in range(m):
		if int(labelList[i]) == 1:
			x1.append(dataList[i][1])
			y1.append(dataList[i][2])
		else:
			x2.append(dataList[i][1])
			y2.append(dataList[i][2])

	ax.scatter(x1, y1, s=30, c='red', marker='s')
	ax.scatter(x2, y2, s=30, c='green')

	x = np.arange(-3.0, 3.0, 0.1)

	y_ = (-weights[0]-weights[1]*x)/weights[2]

	y = np.array(y_)[0]
	
	ax.plot(x, y)

	plt.xlabel('X1'); plt.ylabel('X2');


test_logistic_regression()






