import numpy as np
import copy


def relu(Z):
    """
    Computes the reLU activation function.

    Args:
        Z (numpy.ndarray): The input to the reLU function(pre-activation parameter).

    Returns:
        A (numpy.ndarray): The ouput of the reLU function(post-activation value).
        cache (tuple): A tuple containing 'Z' for use in backpropagation.
    """    
    A = np.maximum(0, Z)
    cache = (Z)
    return A, cache

def sigmoid(Z):
    """
    Computes the sigmoid activation function.

    Args:
        Z (numpy.ndarray): The input of the sigmoid function(pre-activation parameter).
    Returns:
        A (numpy.ndarray): The output of the sigmoid function(post-activation value).
        cache (tuple): A tuple containing 'Z' for use in backpropagation.
    """    
    A = 1/(1+np.exp(-Z))
    cache = (Z)
    return A, cache

def tanh(Z):
    """
    Computes the tanh activation function.

    Args:
        Z (numpy.ndarray): The input of the tanh function(pre-activation parameter).

    Returns:
        A (numpy.ndarray): The output of the tanh function(post-activation value).
        cache (tuple): A tuple containing 'Z' for use in backpropagation.
    """    
    A = np.tanh(Z)
    cache = (Z)
    return A, cache


def tanh_backward(dA, cache):
    """
    Computes the backpropagation for a single tanh unit.

    Args:
        dA (numpy.ndarray): The gradient of the cost with respect to the activation(post-activation gradient).
        cache (tuple): A tuple containing 'Z', used to compute A and therefore dZ.

    Returns:
        dZ (numpy.ndarray): The gradient of the cost with respect to Z.
    """    
    Z = cache

    A = np.tanh(Z)

    dZ = dA * (1 - A**2)
    
    return dZ

def relu_backward(dA, cache):
    """
    Computes the backpropagation for a single relu unit.

    Args:
        dA (numpy.ndarray): The gradient of the cost with respect to the activation(post-activation gradient).
        cache (tuple): A tuple containing 'Z', used to compute dZ.

    Returns:
        dZ (numpy.ndarray): The gradient of the cost with respect to Z.
    """    
    Z = cache
    
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Computes the backpropagation for a single sigmoid unit.

    Args:
        dA (numpy.ndarray): The gradient of the cost with respect to the activation(post-activation gradient).
        cache (tuple): A tuple containing 'Z', used to compute dZ.

    Returns:
        dZ (numpy.ndarray): The gradient of the cost with respect to Z.
    """    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def init_params(layer_dims):
    """
    Initializes the weight matrices and biase vectors for a deep neural network.

    Args:
        layer_dims (list): A list representing the neural network dimensions : 
            it length is the num of layers and each element is the num of nodes in the layer l.

    Returns:
        parameters (dict):  A dictionary containing the initialized weight matrices and bias vectors for each layer:
            - parameters['W' + str(l)] = W_l, where W_l is the weight matrix for layer l.
            - parameters['b' + str(l)] = b_l, where b_l is the bias vector for layer l.
    """    
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def update_params(params, grads, alpha):
    """
    Updates parameters using gradient descent.

    Args:
        params (dict): A dictionary containing the current parameters of the model.
        grads (dict): A dictionary containing the gradient of the cost with respect to the parameters.
        alpha (float): The learning rate.

    Returns:
        parameters (dict): A dictionary containing the updated parameters of the model.
    """    
    parameters = copy.deepcopy(params)

    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - alpha * grads['dW' + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - alpha * grads["db" + str(l+1)]
    
    return parameters
