#
from machine_learning_utils import \
    DataPreProcessor, \
    Regressor, \
    FeatureScaler, \
    DataShaper, \
    Visualizer

dpp = DataPreProcessor()
wd = dpp()
X = wd.X
y = wd.y

# scale feature
fs_X = FeatureScaler()
fs_y = FeatureScaler()
X = fs_X.scale_feature(X)
y = fs_y.scale_feature(y)

# fitting Support Vector Regression (SVR) to the dataset
regressor = Regressor().DecisionTree()
regressor.fit(X, y)

y_pred = regressor.predict(6.5)

data_shaper = DataShaper()
X_grid = data_shaper.to_high_res(X, 0.01)

vs = Visualizer()
plt = vs(X, y, X_grid, regressor.predict(X_grid),
         'Truth or Blue (SVR)', 'Position level', 'Salary')
plt.show()
