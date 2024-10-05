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


def init_v(parameters):
    """
    Initializes the v hyperparameter for the momentum optimizer.

    Args:
        parameters (dict):  A dictionary containing the initialized weight matrices and bias vectors for each layer:
            - parameters['W' + str(l)] = W_l, where W_l is the weight matrix for layer l.
            - parameters['b' + str(l)] = b_l, where b_l is the bias vector for layer l.

    Returns:
        v (dict): A dictionary containing the initialized v values for each dW and db of each layer.
    """   
    v = {}
    L = len(parameters) // 2

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
    
    return v


def init_adam(parameters):
    """
    Initializes the v and s hyperparameters for the adam optimizer.

    Args:
        parameters (dict):  A dictionary containing the initialized weight matrices and bias vectors for each layer:
            - parameters['W' + str(l)] = W_l, where W_l is the weight matrix for layer l.
            - parameters['b' + str(l)] = b_l, where b_l is the bias vector for layer l.

    Returns:
        v (dict): A dictionary containing the initialized v values for each dW and db of each layer
        s (dict): A dictionary containing the initialized s values for each dW and db of each layer
    """  

    L = len(parameters) // 2 
    v = {}
    s = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
    
    return v, s


def update_params(params, grads, alpha):
    """
    Updates parameters using gradient descent.

    Args:
        params (dict): A dictionary containing the current parameters of the model.
        grads (dict): A dictionary containing the gradient of the cost with respect to the parameters.
        alpha (float): The learning rate.

    Returns:
        params (dict): A dictionary containing the updated parameters of the model.
    """    
    params = copy.deepcopy(params)

    L = len(params) // 2

    for l in range(L):
        params['W' + str(l+1)] = params['W' + str(l+1)] - alpha * grads['dW' + str(l+1)]
        params["b" + str(l+1)] = params["b" + str(l+1)] - alpha * grads["db" + str(l+1)]
    
    return params


def update_params_momentum(params, grads, alpha, v, beta=0.9):
    """
    Updates parameters using momentum.

    Args:
        params (dict): A dictionary containing the current parameters of the model.
        grads (dict): A dictionary containing the gradient of the cost with respect to the parameters.
        alpha (float): The learning rate.
        v (dict): A dictionary containing the first moment values (v).
        beta (float): momentum coefficient.

    Returns:
        params (dict): A dictionary containing the updated parameters of the model.
        v (dict): A dictionary containing the updated first moment values (v).
    """    
    L = len(params) // 2

    for l in range(1, L+1):
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]

        params['W' + str(l)] = params['W' + str(l)] - alpha * grads['dW' + str(l)]
        params["b" + str(l)] = params["b" + str(l)] - alpha * grads["db" + str(l)]

    return params, v


def update_params_adam(params, grads, v, s, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Updates parameters using the Adam.

    Args:
        params (dict): A dictionary containing the current parameters of the model, including weights and biases.
        grads (dict): A dictionary containing the gradients of the cost function with respect to the parameters.
        v (dict): A dictionary containing the exponentially weighted averages of the gradients (first moment).
        s (dict): A dictionary containing the exponentially weighted averages of the squared gradients (second moment).
        t (int): The current time step (iteration number).
        alpha (float, optional): The learning rate used for updating the parameters. Default is 0.001.
        beta1 (float, optional): The exponential decay rate for the first moment estimate. Default is 0.9.
        beta2 (float, optional): The exponential decay rate for the second moment estimate. Default is 0.999.
        epsilon (float, optional): A small constant added to prevent division by zero. Default is 1e-8.

    Returns:
        params (dict): A dictionary containing the updated parameters of the model.
        v (dict): A dictionary containing the updated first moment values (v).
        s (dict): A dictionary containing the updated second moment values (s).
    """    
    L = len(params) // 2 
    v_corrected = {}  
    s_corrected = {}          
    
    for l in range(L):
        
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]
        
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * np.square(grads["dW" + str(l+1)])
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * np.square(grads["db" + str(l+1)])
        
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1**t)

        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2**t)
        
        params["W" + str(l+1)] = params["W" + str(l+1)] - alpha * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        params["b" + str(l+1)] = params["b" + str(l+1)] - alpha * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
    
    return params, v, s