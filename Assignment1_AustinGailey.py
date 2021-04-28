# HW 1 - Linear Regression and Loss Function - Author: Austin Gailey
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([35., 38., 31., 20., 22., 25., 17., 60., 8., 60.])
y_data = 2*x_data+50+5*np.random.random(10)

bias = np.arange(0,100,1) #bias
weight = np.arange(-5, 5,0.1) #weight
Z = np.zeros((len(bias),len(weight)))
 
for i in range(len(bias)):
    for j in range(len(weight)):
        b = bias[i]
        w = weight[j]
        Z[j][i] = 0        
        for n in range(len(x_data)):
            x = n
            Z[j][i] = Z[j][i] + (w*x_data[n] + b - y_data[n])**2 #loss function SSE
        Z[j][i] = Z[j][i]/len(x_data)

b = 0 # initial b
w = 0 # initial w

lr = 0.0001 # example learning rate
iteration = 10000 # example iteration number

# Store parameters for plotting
b_history = [b]
w_history = [w]

# model by gradient descent
for i in range(iteration):
    b_gradient = 0
    w_gradient = 0
    for n in range (len(x_data)):
        b_gradient = b_gradient + (b + w*x_data[n] - y_data[n])*1.0
        w_gradient = w_gradient + (b + w*x_data[n] - y_data[n]) * x_data[n]
        b = b - b_gradient * lr
        w = w - w_gradient * lr
        b_history.append(b)
        w_history.append(w)
        
plt.plot(b_history,w_history,'o-',ms=3, lw=1.5,color='black')
plt.xlim(0,100)
plt.ylim(-5,5)
plt.contourf(bias,weight,Z,50,alpha=0.5,cmap = plt.get_cmap('jet'))
plt.show()
