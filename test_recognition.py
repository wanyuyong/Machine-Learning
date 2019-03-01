import sys, os
import numpy as np 
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 200
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

for i in range(iters_num):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	grad = network.gradient(x_batch, t_batch)

	for key in('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]

	loss = network.loss(x_batch, t_batch)

	if i % 100 == 0:
		print('loss is : ', loss)

x_test = x_test[:20]
t_test = t_test[:20]
y = network.predict(x_test)

print('y shape = ', y.shape)
print(np.argmax(y, axis=1))
print('t_test shape = ', t_test.shape)
print(np.argmax(t_test, axis=1))
