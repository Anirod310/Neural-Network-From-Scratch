import numpy as np
from utils import *


def linear_forward(A, W, b):
    """
    Computes the linear part of the forward propagation.

    Args:
        A (numpy.ndarray): Activations of the previous layer (or input layer), of shape (size of previous layer, num examples).
        W (numpy.ndarray): Weights matrix, of shape (size of current layer, size of previous layer).
        b (numpy.ndarray): biases vector, of shape(size of current layer, 1).

    Returns:
        Z (numpy.ndarray): The pre-activation parameter.
        cache (tuple): A tuple containing 'A', 'W' and 'b', for computing the backprob.
    """    
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Computes the activation part of the forward propagation.

    Args:
        A_prev (numpy.ndarray): Activations of the previous layer (or input layer), of shape (size of previous layer, num examples).
        W (numpy.ndarray): Weights matrix, of shape (size of current layer, size of previous layer).
        b (numpy.ndarray): biases vector, of shape(size of current layer, 1).
        activation (str): The activation used in this layer, stored as a text string: 'sigmoid', 'tanh' or 'relu'.

    Returns:
        A (numpy.ndarray): The post-activation value.
        cache (tuple): A tuple containing the content of the previous cache('A', 'W', 'b') as well as the content of the activation cache('Z'),
            for computing the backprob.
    """    
    if activation == 'sigmoid' : 
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    elif activation == 'tanh':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def forward_prop(X, parameters):
    """
    Computes the forward propagation of a deep neural network.

    Args:
        X (numpy.ndarray): Input data of shape(input size, num examples).
        parameters (dict): A dictionary containing the parameters 'W1', 'b1', ..., 'WL', 'bL'.

    Returns:
        AL (numpy.ndarray): The output of the last layer(post-activation value).
        cache (list): A list of all the caches of linear_activation_forward():
            the caches are indexed from 0 to L-1.
    """    
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
    Computes the cross-entropy cost.

    Args:
        AL (numpy.ndarray): The post-activation value, corresponding to the labels predictions, of shape (1, num examples).
        Y (numpy.ndarray): The true label vector, of shape (1, num examples).

    Returns:
        cost (float): The cross-entropy cost.
    """    
    m = Y.shape[1]

    logprobs = np.multiply(np.log(AL), Y)
    cost = -1/m * np.sum(logprobs + (1-Y) * np.log(1-AL))

    cost = np.squeeze(cost) 

    return cost


def linear_backward(dZ, cache):
    """
    Computes the linear part of the back propagation.

    Args:
        dZ (numpy.ndarray): Gradient of the cost with respect to the linear output of the current layer l.
        cache (tuple): A tuple of values (A_prev, W, b) coming from the forward propagation in the current layer.

    Returns:
        dA_prev (numpy.ndarray): Gradient of the cost with respect to the activation of the previous layer l-1, same shape as A_prev.
        dW (numpy.ndarray): Gradient of the cost with respect to W of the current layer l, same shape as W.
        db (numpy.ndarray): Gradient of the cost with respect to b of the current layer l, same shape as b.
    """    
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Computes the activation part of the back propagation.

    Args:
        dA (numpy.ndarray): Post-activation gradient for the current layer l.
        cache (tuple): A tuple of values (linear_cache, activation_cache) we store for retrieving the parameters we need for back propagation.
        activation (str): The activation used in this layer, stored as a text string: 'sigmoid', 'tanh' or 'relu'.

    Returns:
        dA_prev (numpy.ndarray): Gradient of the cost with respect to the activation of the previous layer l-1, same shape as A_prev.
        dW (numpy.ndarray): Gradient of the cost with respect to W of the current layer l, same shape as W.
        db (numpy.ndarray): Gradient of the cost with respect to b of the current layer l, same shape as b.
    """    
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    elif activation == 'tanh':
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def back_prob(AL, Y, caches):
    """
    Computes the back propagation for a deep neural network.

    Args:
        AL (numpy.ndarray): Probability vector, output of the forward propagation.
        Y (numpy.ndarray): True label vector, of shape (1, num examples).
        caches (list): List of caches containing every cache of linear_activation_forward() with "relu" (there are L-1 of them, indexed from 0 to L-2),
          and the cache of linear_activation_forward() with "sigmoid" (indexed L-1).

    Returns:
        grads (dict): A dictionary with the gradients with respect to different parameters:
            - grads["dA" + str(l)] = dA_l
            - grads["dW" + str(l)] = dW_l
            - grads["db" + str(l)] = db_l
    """    
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, 'sigmoid')
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
    
    return grads



def nn_model_base(X, Y, layer_dims, iterations, alpha):
    """
    Implements a L-layer neural network.

    Args:
        X (numpy.ndarray): Input data, shape(Input size, num examples).
        Y (numpy.ndarray): Input labels, shape(1, num examples).
        layer_dims (list): A list representing the neural network dimensions : 
            it length is the num of layers and each element is the num of nodes in the layer l.
        iterations (int): Number of iterations of the optimisation loop.
        alpha (float): Learning rate of the gradient descent update rule.

    Returns:
        parameters (dict): Parameters learned by the model. They can be used to predict.
        cost (float): The final cost after the last iteration.
    """    
    np.random.seed(1)

    parameters = init_params(layer_dims)

    for i in range(iterations):
        AL, caches = forward_prop(X, parameters)

        cost = compute_cost(AL, Y)

        grads = back_prob(AL, Y, caches)

        parameters = update_params(parameters, grads, alpha)

        if i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters, cost