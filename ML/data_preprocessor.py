# Data Preprocessing Template

# Importing the libraries
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


def scale_features(_X_train, _X_test):
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    _X_train = sc_X.fit_transform(_X_train)
    _X_test = sc_X.transform(_X_test)
    return _X_train, _X_test


class TrainingTestData:
    def __init__(self, _X_train, _X_test, _y_train, _y_test):
        self.X_train = _X_train
        self.X_test = _X_test
        self.y_train = _y_train
        self.y_test = _y_test


class WorkingData:
    def __init__(self, _X, _y):
        self.X = _X
        self.y = _y


class DataPreProcessor:
    def __call__(self):
        _X, _y = read_data_set(
            '50_Startups.csv',
            independant_index=-1,
            dependant_index=4)
        # _X = handle_missing_data(_X)
        _X, _y = encode_categorical_data(
            _X,
            _y,
            independent_index=3,
            dependent_index=0)
        _X_train, _X_test, _y_train, _y_test = split_data_sets(_X, _y, test_size=0.2)
        # _X_train, _X_test = scale_features(_X_train, _X_test)
        return TrainingTestData(_X_train, _X_test, _y_train, _y_test), WorkingData(_X, _y)

