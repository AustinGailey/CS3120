import numpy as np
import matplotlib.pyplot as plt

# Create Arrays x and y
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#Create variable with complex function
[X,Y] = np.meshgrid(x,y)

Z = np.array((X-5)**2) + np.array((Y-5)**2)

#Plot Contour of Z
plt.contourf(X,Y,Z, 20, alpha =0.5, cmap = plt.get_cmap('jet'))
#Get min value of Z
minValue = np.where(Z == np.amin(Z))
plt.plot(minValue[0][0]+1,minValue[1][0]+1, 'o', ms=12, markeredgewidth=3, color='orange')
plt.xlim(0,10)
plt.ylim(0,10)
plt.xlabel(r'$X$',fontsize=16)
plt.ylabel(r'$Y$',fontsize=16)
plt.title("Austin_Gailey_Quiz1Contour")
plt.show()
    