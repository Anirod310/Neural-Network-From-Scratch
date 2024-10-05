import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_moons, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """
    Loads and preprocesses the data

    Returns:
        X_train (numpy.ndarray): training data, shape(Input size, num training examples).
        X_test (numpy.ndarray): testing data, shape(Input size, num testing examples).
        y_train (numpy.ndarray): training labels, shape(1, num training examples).
        y_test (numpy.ndarray): testing labels, shape(1, num testing examples).
    """    
    
    spambase = fetch_openml(name="spambase", version=1, as_frame=True)
    X, y = spambase.data, spambase.target
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.to_numpy().reshape(-1, 1).T, y_test.to_numpy().reshape(-1, 1).T
    
    return X_train, X_test, y_train, y_test


def size_and_shape_of_data(X_train, X_test, y_train, y_test):
    """
    Displays the different shapes and sizes of the data.

    Args:
        X_train_orig (numpy.ndarray): the training set features (images).
        X_test_orig (numpy.ndarray): the testing set features.
        Y_train (numpy.ndarray): the training set labels.
        Y_test (numpy.ndarray): the testing set labels.

    """   
    m_train = X_train.shape[1]
    num_px = X_train.shape[0]
    m_test = X_test.shape[1]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each doc is of size: (" + str(num_px) + ")")
    print ("train_x_orig shape: " + str(X_train.shape))
    print ("train_y shape: " + str(y_train.shape))
    print ("test_x_orig shape: " + str(X_test.shape))
    print ("test_y shape: " + str(y_test.shape))

