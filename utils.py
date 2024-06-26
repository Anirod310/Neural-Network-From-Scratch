import numpy as np
import copy


def relu(Z):
    A = np.maximum(0, Z)
    cache = (Z)
    return A, cache

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = (Z)
    return A, cache

def tanh(Z):
    A = np.tanh(Z)
    cache = (Z)
    return A, cache


def tanh_backward(dA, cache):
    Z = cache

    A = np.tanh(Z)

    dZ = dA * (1 - A**2)
    
    return dZ

def relu_backward(dA, cache):
    Z = cache
    
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def init_params(layer_dims):
    
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def update_params(params, grads, alpha):

    parameters = copy.deepcopy(params)

    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - alpha * grads['dW' + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - alpha * grads["db" + str(l+1)]
    
    return parameters
