# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_data_set(data_set_path, independant_index, dependant_index):
    # Importing the dataset
    dataset = pd.read_csv(data_set_path)
    _X = dataset.iloc[:, :independant_index].values
    _y = dataset.iloc[:, dependant_index].values
    return _X, _y


def handle_missing_data(X):
    # Handle missing data
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])
    return X


def encode_categorical_data(_X, _y):
    # Encode categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    label_encoder__x = LabelEncoder()
    _X[:, 0] = label_encoder__x.fit_transform(_X[:, 0])
    one_hot_encoder = OneHotEncoder(categorical_features=[0])
    _X = one_hot_encoder.fit_transform(_X).toarray()
    label_encoder_y = LabelEncoder()
    _y = label_encoder_y.fit_transform(_X[:, 0])
    return _X, _y


def split_data_sets(_X, _y, test_size=0.2):
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    return train_test_split(_X, _y, test_size=test_size, random_state=0)


def scale_features(_X_train, _X_test):
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    _X_train = sc_X.fit_transform(_X_train)
    _X_test = sc_X.transform(_X_test)
    return _X_train, _X_test


class DataPreProcessor:
    def __call__(self):
        _X, _y = read_data_set(
            'Salary_Data.csv',
            independant_index=-1,
            dependant_index=1)
        # _X = handle_missing_data(_X)
        # _X, _y = encode_categorical_data(_X, _y)
        _X_train, _X_test, _y_train, _y_test = split_data_sets(_X, _y, test_size=0.2)
        # _X_train, _X_test = scale_features(_X_train, _X_test)
        return _X_train, _X_test, _y_train, _y_test

