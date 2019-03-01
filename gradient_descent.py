import numpy as np

def numerical_gradient(f, x):
	h = 1e-4
	grad = np.zeros_like(x)

	for idx in range(x.size):
		tmp = x[idx]
		
		x[idx] = tmp + h
		fxh1 = f(x)

		x[idx] = tmp - h
		fxh2 = f(x)

		grad[idx] = (fxh1 - fxh2) / (2 * h)

		x[idx] = tmp

	return grad

def function_2(x):
	return x[0]**2+x[1]**2

def gradient_descent(f, init_x, lr=0.01, step=100):
	x = init_x
	for i in range(step):
		grad = numerical_gradient(f, x)
		x -= lr * grad
	return x

init_x = np.array([-3.0, 4.0])
print gradient_descent(function_2, init_x=init_x, lr=0.1)
