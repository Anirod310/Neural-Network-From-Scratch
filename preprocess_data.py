import numpy as np
import h5py
import matplotlib.pyplot as plt



plt.rcParams['figure.figsize'] = (5.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'


def load_data():
    """
    Loads the training and testing datasets from HDF5 files.

    Returns:
        train_set_x_orig (numpy.ndarray) : the training set features.
        train_set_y_orig (numpy.ndarray) : the training set labels.
        test_set_x_orig (numpy.ndarray) : the testing set features.
        test_set_y_orig (numpy.ndarray) : the testing set labels.            
        classes (numpy.ndarray) : the list of classes.
            
    """    
    train_set = h5py.File('c:/Users/bouse/Desktop/Programmation/Python/GitHub/Projects/Neural-network-from-scratch/archive/train_catvnoncat.h5', 'r')
    train_set_x_orig = np.array(train_set["train_set_x"][:]) 
    train_set_y_orig = np.array(train_set["train_set_y"][:])

    test_set = h5py.File('c:/Users/bouse/Desktop/Programmation/Python/GitHub/Projects/Neural-network-from-scratch/archive/test_catvnoncat.h5', 'r')
    test_set_x_orig = np.array(test_set["test_set_x"][:]) 
    test_set_y_orig = np.array(test_set["test_set_y"][:]) 

    classes = np.array(test_set['list_classes'][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def img_from_training_dataset(index, X_train_orig, Y_train, classes):
    """
    Displays a specific image from the training dataset and it's corresponding label.

    Args:
        index (int): Index of the image to display.
        X_train_orig (numpy.ndarray): the training set features (images).
        Y_train (numpy.ndarray): the training set labels.
        classes (numpy.ndarray): the list of classes.
    """    
    plt.imshow(X_train_orig[index])
    plt.show()
    print ("y = " + str(Y_train[0,index]) + ". It's a " + classes[Y_train[0,index]].decode("utf-8") +  " picture.")


def size_and_shape_of_data(X_train_orig, X_test_orig, Y_train, Y_test):
    """
    Displays the different shapes and sizes of the data.

    Args:
        X_train_orig (numpy.ndarray): the training set features (images).
        X_test_orig (numpy.ndarray): the testing set features.
        Y_train (numpy.ndarray): the training set labels.
        Y_test (numpy.ndarray): the testing set labels.

    """   
    m_train = X_train_orig.shape[0]
    num_px = X_train_orig.shape[1]
    m_test = X_test_orig.shape[0]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(X_train_orig.shape))
    print ("train_y shape: " + str(Y_train.shape))
    print ("test_x_orig shape: " + str(X_test_orig.shape))
    print ("test_y shape: " + str(Y_test.shape))


def reshape_standardize_train_test_examples(X_train_orig, X_test_orig):
    """
    Reshapes the training and testing set features into shape (image_shape, num_examples), 
    then standardizes them.

    Args:
        X_train_orig (numpy.ndarray): the training set features.
        X_test_orig (numpy.ndarray): the testing set features.

    Returns:
        X_train (numpy.ndarray): the training set features, reshaped and standardized.
        X_test (numpy.ndarray): the testing set features, reshaped and standardized.
    """    
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T   
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.

    #print ("train_x's shape: " + str(X_train.shape))
    #print ("test_x's shape: " + str(X_test.shape))
    
    return X_train, X_test

