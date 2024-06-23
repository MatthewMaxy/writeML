import numpy as np 

X = np.random.rand(100,4)
a, b, c, d, f = 1.5, -2, 3.1, 4.2, 6.1

Y = a*X[:,0] + b*X[:,1] + c*X[:,2] + d*X[:, 3] + f
Y = Y.reshape((len(Y), 1))
X_b = np.hstack([X, np.ones((100,1))])

eta = 0.01
n_iteration = 5000

W = np.random.rand(5,1)

for i in range(n_iteration):
    grad = 2/100 * X_b.T.dot(X_b.dot(W)-Y)
    W -= eta * grad

print(W)