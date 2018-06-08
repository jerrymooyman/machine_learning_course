# # Multiple Linear Regression
#
# # Importing the libraries
import numpy as np

from data_preprocessor import DataPreProcessor
dpp = DataPreProcessor()
ttd, wd = dpp()

# Fitting Multiple Linear Regression to the  Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(ttd.X_train, ttd.y_train)

# Predicting the Test set results
y_pred = regressor.predict(ttd.X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=wd.X, axis=1)
y = wd.y
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
