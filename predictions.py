import numpy as np
from base_model import forward_prop
import matplotlib.pyplot as plt

def accuracy(X, Y, parameters):
    """
    Compute the accuracy of the predictions of the neural network.

    Args:
        X (numpy.ndarray): Input data, shape(Input size, num examples).
        Y (numpy.ndarray): Input labels, shape(1, num examples).
        parameters (dict): Parameters learned by the model.

    Returns:
        accuracy (float): Accuracy of the model's predictions.
    """    
    m = X.shape[1]

    predictions = predict(X, parameters)

    accuracy = np.sum(predictions == Y) / m

    return accuracy

def predict(X, parameters):
    """
    Predicts the labels for a given input dataset using the trained parameters.

    Args:
        X (numpy.ndarray): Input data, shape(Input size, num examples).
        parameters (dict): Parameters learned by the model.

    Returns:
        prediction (numpy.ndarray): Predicted labels for the input data, shape(1, num_examples).
    """    

    AL, caches = forward_prop(X, parameters)
    prediction = (AL > 0.5)

    return prediction
