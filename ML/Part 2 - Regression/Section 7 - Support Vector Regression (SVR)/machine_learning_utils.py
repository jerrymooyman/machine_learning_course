# Data Preprocessing Template

# Importing the libraries
import pandas as pd
import numpy as np


def read_data_set(data_set_path):
    # Importing the dataset
    dataset = pd.read_csv(data_set_path)
    return dataset

def split_X_and_y(
        dataset,
        independent_index_start,
        independent_index_stop,
        dependent_index_start,
        dependent_index_stop):
    _X = dataset.iloc[:, independent_index_start:independent_index_stop].values
    _y = dataset.iloc[:, dependent_index_start:dependent_index_stop].values
    return _X, _y

def handle_missing_data(X):
    # Handle missing data
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])
    return X

# NB: don't forget to handle the 'dummy variable trap'
def encode_categorical_data(_X, _y, independent_index, dependent_index):
    # Encode categorical data
    # Encode independant variable
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    label_encoder__x = LabelEncoder()
    _X[:, independent_index] = label_encoder__x.fit_transform(_X[:, independent_index])
    one_hot_encoder = OneHotEncoder(categorical_features=[independent_index])
    _X = one_hot_encoder.fit_transform(_X).toarray()
    # Encode dependant variable
    # label_encoder_y = LabelEncoder()
    # _y = label_encoder_y.fit_transform(_X[:, dependent_index])
    # Handle the 'dummy variable trap'
    _X = _X[:, 1:]
    return _X, _y


def split_data_sets(_X, _y, test_size=0.2):
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    return train_test_split(_X, _y, test_size=test_size, random_state=0)


class FeatureScaler:
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self._sc = StandardScaler()

    def scale_feature(self, value):
        # Feature Scaling
        return self._sc.fit_transform(value)

    def scale_value(self, value):
        return self._sc.transform(np.array([[value]]))

    def inverse_scale(self, value):
        return self._sc.inverse_transform(value)


class TrainingTestData:
    def __init__(self, _X_train, _X_test, _y_train, _y_test):
        self.X_train = _X_train
        self.X_test = _X_test
        self.y_train = _y_train
        self.y_test = _y_test


class WorkingData:
    def __init__(self, _X, _y, _dataset):
        self.X = _X
        self.y = _y
        self.dataset = _dataset


class DataPreProcessor:
    def __call__(self):
        dataset = read_data_set('Position_Salaries.csv')
        _X, _y = split_X_and_y(
            dataset,
            independent_index_start=1,
            independent_index_stop=2,
            dependent_index_start=2,
            dependent_index_stop=3)
        # _X = handle_missing_data(_X)
        # _X, _y = encode_categorical_data(
        #     _X,
        #     _y,
        #     independent_index=3,
        #     dependent_index=0)
        # _X_train, _X_test, _y_train, _y_test = split_data_sets(_X, _y, test_size=0.2)
        # _X_train, _X_test = scale_features(_X_train, _X_test)
        # featureScaler = FeatureScaler()
        # _X, = featureScaler.scale_feature(_X)
        # _y, = featureScaler.scale_feature(_y)
        return WorkingData(_X, _y, dataset)

class Regressor:
    def SVR(self, kernel='rbf'):
        from sklearn.svm import SVR
        return SVR(kernel)


class DataShaper:
    def to_high_res(self, _X):
        _X_grid = np.arange(min(_X), max(_X), 0.1)
        _X_grid = _X_grid.reshape((len(_X_grid), 1))
        return _X_grid

class Visualizer:
    def __call__(self, scatter_X, scatter_y, plot_X, predict_X, _title, _x_label, _y_label):
        import matplotlib.pyplot as plt
        plt.scatter(scatter_X, scatter_y, color='red')
        plt.plot(plot_X, predict_X, color='blue')
        plt.title(_title)
        plt.xlabel(_x_label)
        plt.ylabel(_y_label)
        return plt
