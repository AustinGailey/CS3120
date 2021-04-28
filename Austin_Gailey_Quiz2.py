# Quiz 2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

m = 500
np.random.seed(seed=5)
X = 6 * np.random.random(m).reshape(-1, 1)  - 3
y = 0.5 * X**5 -X**3- X**2 + 2 +  5*np.random.randn(m, 1) # y with both positive and negative randomness added

# Split the data into training/testing sets
X_train = X[:-200]
X_test = X[-200:]

# Split the targets into training/testing sets
y_train = y[:-200]
y_test = y[-200:]

# Create linear regression object
regr = LinearRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)
# Make predictions using the testing set
y_pred = regr.predict(X_test)

linear_loss_history = [0,0]
poly_loss_history = [0,0]

def createRegression():
    for i in range(2,25):
        regr2 = LinearRegression()
        degree = i
        poly2_features = PolynomialFeatures(degree)
        X_poly2 = poly2_features.fit_transform(X_train)
        X_poly2_test = poly2_features.fit_transform(X_test)
        regr2.fit(X_poly2, y_train)
        y_pred2 = regr2.predict(X_poly2_test)

        print('Degree: ', degree)
        # The coefficients
        print('Coefficients: \n', regr.coef_)
        print('Itercept: \n', regr.intercept_)
        # The mean squared error
        linear_loss_history.append(mean_squared_error(y_test, y_pred))
        print('Mean squared error of linear model: %.2f'      % mean_squared_error(y_test, y_pred))
        poly_loss_history.append(mean_squared_error(y_test, y_pred2))
        print('Mean squared error of poly2 model: %.2f'      % mean_squared_error(y_test, y_pred2))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.2f'      % r2_score(y_test, y_pred))
        print('Coefficient of determination ploy2: %.2f'      % r2_score(y_test, y_pred2))

        # Plot outputs
        plt.scatter(X_test, y_test,  color='red')
        plt.scatter(X_test, y_pred, color='blue', linewidth=3)
        plt.scatter(X_test, y_pred2, color='green', linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        
createRegression()

plt.plot(linear_loss_history,'o-',ms=3, lw=1.5,color='blue')
plt.plot(poly_loss_history,'o-',ms=3, lw=1.5,color='green')
plt.xlim(2,25)
plt.show()

