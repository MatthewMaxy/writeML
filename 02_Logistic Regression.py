import numpy as np
import matplotlib.pyplot as plt 

X = np.array([[3.4, 2.8],
     [3.1, 1.8],
     [1.5, 3.4],
     [3.6, 4.7],
     [2.7, 2.9],
     [7.4, 4.5],
     [5.7, 3.5],
     [9.2, 2.5],
     [7.9, 3.4]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
colors = ['red' if label == 0 else 'blue' for label in y]
y = y.reshape((len(y), 1))

# plt.scatter(X[:,0], X[:,1], c= colors)
# plt.show()

X_b = np.hstack([X, np.ones((9,1))])

W = np.random.rand(3,1)
eta = 0.01
n_iteration = 100

for _ in range(n_iteration):
    h = 1/(1+np.exp(-(X_b.dot(W))))
    grad = X_b.T.dot(h - y)
    W -= eta*grad

pred = 1/(1+np.exp(-(X_b).dot(W)))
pred.reshape((1,len(pred)))

color_ = ['red' if label<0.5 else 'blue' for label in pred]

plt.scatter(X[:,0], X[:,1], c= colors)
plt.show()
plt.scatter(X[:,0], X[:,1], c= color_)
plt.show()