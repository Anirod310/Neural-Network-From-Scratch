import numpy as np
from base_model import forward_prop
from PIL import Image
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


def predict_cat_noncat(image_to_predict, num_px, parameters, classes):
    """
    Predicts wether an image is a cat or not.

    Args:
        image_to_predict (str): Name of the image file to predict.
        num_px (int): Size of the image.
        parameters (dict): Parameters of the trained model.
        classes (numpy.ndarray): Array of classe labels ("cat" or "non-cat").
    """
    fname = "images/" + image_to_predict
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    plt.imshow(image)
    plt.show()
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T

    predicted_image = int(predict(image, parameters))

    print ("y = " + str(np.squeeze(predicted_image)) + ", It's a \"" + classes[int(np.squeeze(predicted_image)),].decode("utf-8") +  "\" picture.")