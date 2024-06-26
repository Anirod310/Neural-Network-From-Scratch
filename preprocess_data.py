import numpy as np
import h5py
import matplotlib.pyplot as plt



plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'


def load_data():
    train_set = h5py.File('c:/Users/bouse/Desktop/Programmation/Python/GitHub/Projects/Neural-network-from-scratch/archive/train_catvnoncat.h5', 'r')
    train_set_x_orig = np.array(train_set["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_set["train_set_y"][:])

    test_set = h5py.File('c:/Users/bouse/Desktop/Programmation/Python/GitHub/Projects/Neural-network-from-scratch/archive/test_catvnoncat.h5', 'r')
    test_set_x_orig = np.array(test_set["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_set["test_set_y"][:]) # your test set labels

    classes = np.array(test_set['list_classes'][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def img_from_dataset(index, X_train_orig, Y_train, classes):

    plt.imshow(X_train_orig[index])
    plt.show()
    print ("y = " + str(Y_train[0,index]) + ". It's a " + classes[Y_train[0,index]].decode("utf-8") +  " picture.")


def size_and_shape_of_data(X_train_orig, X_test_orig, Y_train, Y_test):
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

    return num_px


def reshape_standardize_train_test_examples(X_train_orig, X_test_orig):
        
    train_x_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T   
    test_x_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

    X_train = train_x_flatten/255.
    X_test = test_x_flatten/255.

    #print ("train_x's shape: " + str(X_train.shape))
    #print ("test_x's shape: " + str(X_test.shape))
    
    return X_train, X_test

