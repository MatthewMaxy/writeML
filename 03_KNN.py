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
plt.scatter(X[:,0], X[:,1], c= colors)
# plt.show()
new_x = np.array([7.1, 3.6])
distance = []
for dot in X:
    distance.append(sum((dot-new_x)**2))
nearest = np.argsort(distance)
k = 3
count = [0, 0]
for i in nearest[:k]:
    count[y[i]]+=1
label = 0 if count[0] > count[1] else 1
plt.scatter(x=new_x[0], y=new_x[1], c='green')
plt.show()
print(label)
