import numpy as np
import matplotlib.pyplot as plt

from data_preprocessor import DataPreProcessor
dpp = DataPreProcessor()
wd = dpp()
X = wd.X
y = wd.y

# Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)
# this is the make the plot more curvy
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

y_pred_poly = lin_reg2.predict(poly_reg.fit_transform(X_grid))

# visualizing the linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, y_pred_lin, color='blue')
plt.title('truth or bluf (linear regression')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# visualizing the polynomial regression results
plt.scatter(X, y, color='red')
plt.plot(X_grid, y_pred_poly, color='blue')
plt.title('truth or bluf (linear regression')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# predicting a new result with linear regression
lin_reg.predict(6.5)

# predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
