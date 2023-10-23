# import modules
import numpy as np
import matplotlib.pyplot as plt
# dataset
dataset = np.array([
    [2600,550000],
    [3000,565000],
    [3200,610000],
    [3600,680000],
    [4000,725000]
])
sig_x = sum(dataset[:,0])
sig_y = sum(dataset[:,1])
sig_xy = sum(dataset[:,0]*dataset[:,1])
sig_x_2 = sum(dataset[:,0])**2
sig_x2 = sum(dataset[:,0]**2)
n = dataset.shape[0]

a = (n*sig_xy - (sig_x)*(sig_y))/(n*(sig_x2)-sig_x_2)
b = sig_y/n - a*sig_x/n

# plotting the points
for i in range(n):
    x = dataset[i,0]
    y = dataset[i,1]
    plt.plot(x,y, linestyle=':', marker='o', color='blue')
# plotting the line
x = np.linspace(min(dataset[:,0]),max(dataset[:,0]),1000)
y = a*x + b
label_str = "y = " + str(a) + "x + " + str(b)

plt.plot(x, y, '-r', label = label_str)
plt.title("LINEAR REGRESSION")
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()