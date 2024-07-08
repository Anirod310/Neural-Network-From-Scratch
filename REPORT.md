# Deep Learning Model from scratch : Step-by-step approach

## Introduction
With the increasing capability of computing power and the advent of sophisticated machine learning techniques, deep learning has become a pivotal area in artificial intelligence research. This project aims first to explore the incremental development of a deep learning model, highlighting the impact of each improvement step-by-step, but also to help me understand and learn by myself the basics of the machine learning and deep learning area.

## Objective
The primary objective of this project is to develop a deep learning model for cat image classification, starting with a simple neural network and progressively implementing various enhancements, such as regularization techniques, to improve its performance. By documenting each step, this report serves as both an educational resource and a detailed project log.

## Structure
- **Step 1: Basic Neural Network**: Introduction of a simple neural network model.
- **Step 2: Regularization**: Implementation of regularization techniques.
- **Step 3: Advanced Techniques**: Further enhancements and final model evaluation.
- **Conclusion**: Summary of findings and future work.  

Without delay, let's get starded !

## Step 1 : Basic Neural Network
In this section, we build and evaluate a simple neural network model. This basic model serves as a foundation for further improvments and enhancements that will be discussed in subsequent sections.

### A - Model architecture

#### I - <u>Data preprocessing</u>
To begin with, the data set we're using is composed by two h5 files(representing training and testing set respectively), the traing set consisting of 209 cat or non-cat images and the testing set of 50 images. The output label is either 1(if it's a cat image) or 0(if it's a non-cat image).  
For each image, we reshape it and flatten it so that it can be used as an numpy array of dimensions **(input size, number of examples)**.

#### II - <u>Parameters, hyperparamters, forward/backward propagation, activation functions</u>

For this model, the only parameters are W and b (respectively representing weight and biase), and are initialized in 4 different ways : 
 - Initialization with zeros for both W and b.
 - Random initialization for W(scaled by 0.01) and with zeros for b
 - Xavier initialization for W and with zeros for b
 - He initialization for W and with zeros for b


As you can see in the [main](main.py) file, the hyperparameters we use are the learning rate alpha, the number of iterations of the optimization loop, and the dimensions of the layer.

Also, by looking at the [base_model](base_model.py) file, you can see that the forprop and backprop steps are entirely coded from scratch, using the basic math formulas and numpy. Also, instead of coding directly all the forprop and backprop process in one function, I implemented several steps to split all the process in smaller functions, such as the linear computing step, the activation computing step, etc.

The activation functions we use are sigmoid, reLU and tanh.

We'll see after how the modification of the paramaters, hyperparameters, number and size of layers and activation functions impacts the learning process of the model.

### B - Results analysis

To evaluate the differences among various parameter initializations, hyperparameters, and activation functions, I conducted a series of experiments. I first varied the parameter initialization methods and compared their performances. Next, I explored different activation functions, followed by an examination of various hyperparameters.
To finish with, I varied the size and the number of layers and compared resulting performances. 

The base model used is a 2-layer neural network. The training and testing inputs have shapes of (12288, 209) and (12288, 50), respectively. The activation function for the hidden layer is reLU and for the output layer, it is sigmoid. The model was trained for 2500 iterations with a learning rate (alpha) of 0.0075. The size of the hidden layer is 4.

**Base Model Summary**

| Aspect                 | Description                                   |
|------------------------|-----------------------------------------------|
| Model Type             | 2-layer Neural Network                        |
| Training Input Shape   | (12288, 209)                                  |
| Testing Input Shape    | (12288, 50)                                   |
| Hidden Layer Activation| ReLU                                          |
| Output Layer Activation| Sigmoid                                       |
| Number of Iterations   | 2500                                          |
| Learning Rate (alpha)  | 0.0075                                        |
| Hidden layer size      | 4                                             |

#### I - <u>Parameter initialization</u>

Parameter initialization is a critical step in training neural networks. Proper initialization can speed up the convergence of the training process and improve the overall performance of the model. Conversely, poor initialization can lead to slow convergence, vanishing or exploding gradients, and suboptimal performance. Here, I tried 4 different parameter initialization methods.

1. *Zero Initialization*
    - All weights are initialized to zero.
    - This method is rarely used because it leads to symetry problems where eache neuron in the layer learns the same features.
2. *Random Initialization*
    - Weights are initialized randomly, usually from a Gaussian or uniform distribution.
    - The common method is :
        **Standard normal distribution :** `np.random.randn(shape) * 0.01`
3. *Xavien Initialization*  
    - Weights are initialized from a normal distribution with a mean of 0 and a variance of 1/n, where n is the number of inputs to the neuron.
    - Formula: `np.random.randn(shape) * np.sqrt(1 / n)`
4. *He Initialization*
    - Weights are initialized from a normal distribution with a mean of 0 and a variance of 2/n, where n is the number of inputs to the neuron.
    - Formula: `np.random.randn(shape) * np.sqrt(2 / n)`

Each initialization method was tested under identical conditions, wich are those presented above in the **Base model summary** to ensure a fair comparison.

**Results**

| Initialization Method  | Training Accuracy | Testing Accuracy | Cost before training | Cost after training |
|------------------------|-------------------|------------------|----------------------|----------------------|
| Zero Initialization    | 65.55%            | 34%              | 0.69                 | 0.64                |
| Random Initialization  | 98.09%            | 66%              | 0.69                 | 0.16                |
| Xavier Initialization  | 100%              | 62%              | 0.68                 | 0.11                |
| He Initialization      | 100%              | 72%              | 0.68                 | 0.06                |

**Discussion**

- Zero Init : As expected, zero initialization led to poor performance due to symetry problems.
- Random Init : This method showed decent performance but as outperformed by more sophisicated methods.
- Xavien Init : In this experiment, it provided worse performances that the random initialization, but it's mostly  because the Xavier initialization performs better on deeper networks.
- He Init : Achieved the best performances, particulary effective with ReLU activation functions due to its ability to mitigate vanishing and exploding gradients.

The results demonstrate that proper initialization can significantly affect the learning process and final performance of the model.

**Parameter initialization : conclusion**

Based on the experimentats, He initialization is recommended for networks with ReLu activation functions due to its superior performance. Proper initialization helps in faster convergence and achieving better accuracy.  
By carefully selecting the initialization method, we can ensure a more efficient and effective training process, ultimately leading to better-performing models.

#### II - <u>Activation functions</u>

Activation functions play a crucial role in the performance of neural networks. They introduce non-linearity into the model, allowing it to learn complex patterns. Different activation functions can significantly affect the convergence speed and overall performance of the model. In this project, I focused on 3 different activation functions.

1. Sigmoid activation function
    - Formula : $\sigma(z) = \frac{1}{1 + e^{-z}}$
    - Characteristics : Outputs values between 0 and 1, used primarily in the output layer for binary classification tasks.
    - Potential issues : Can cause vanishing gradients, leading to slow convergence.

2. Tanh activation function
    - Formula : $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
    - Characteristics : Outputs values between -1 and 1, zero-centered, which can help mitigate some issues seen with sigmoid.
    - Potential issues : Can still cause vanishing gradients, though less severe than sigmoid.

3. ReLU activation function 
    - Formula : $\text{ReLU}(z) = \max(0, z)$
    - Characteristics : Outputs values between 0 and infinity, introduces sparsity in the network by zeroing out negative values.
    - Potential issues : Can cause dead neurons if too many values are zeroed out.

To understand the impact of different activation functions, we conducted several experiments using identical conditions, wich are, again, those presented above in the **Base model summary**, but this time by modifying the activation function of the hidden layer, and using the He initialization for every tests. 

**Results**

| Activation function    | Training Accuracy | Testing Accuracy | Cost before training | Cost after training  |
|------------------------|-------------------|------------------|----------------------|----------------------|
| Sigmoid activation     | 97.13%            | 72.00%           | 0.65                 | 0.22                 |
| Tanh activation        | 99.04%            | 72.00%           | 0.71                 | 0.05                 |
| ReLU activation        | 100%              | 72.00%           | 0.68                 | 0.06                 | 

**Discussion**

- Sigmoid : While it provided decent performances, the model converged more slowly compared to other activation functions due to the vanishing gradient probblem.

- Tanh : Showed better performance than sigmoid, beneftining from being zero-centered, which helped in faster convergence. 

- ReLU : Achieved the highest performances, particulary effective in promoting sparsity and speeding up convergence. However, some neurons might became inactive due to the zeroing effect.

**Activation functions : conclusion**

We can thus say that choosing wisely the activation functions used in the hidden and output layers can significantly increase the performances of the model, leading to faster convvergence and better overall accuracy. By using ReLU activation function for the hidden layers combined to sigmoid activation function for the output layer as well as He initialization, we can get much better results on our dataset than using other activation function. 

#### III - <u>Number and Size of Layers</u>

The architecture of a neural network, including the numbber of layers and the number of neurons in each layer, can significanttly impact its performance. Deeper netwworks with more layers can learn more complexe patterns, while wider networks with more neurons per layer can capture more features. 

1. Shallow Network
    - 2-layer neural network
    - Characteristics: Simpler, fewer paramaters, faster training, but may struggle with complex paterns.

2. Deep Network
    - 4-layer neural network
    - Characteristics: More complex, more parameters, longer training time, better at capturing intricate patterns.

3. Wide Network
    - 2 layer neural network with more neurons per layer
    - Characteristics: More neurons per layer, can capture more features, but increases the number of paramaters and therefore the computing cost.

4. Deep and wide Network
    - 4-layer neural network with more neurons per layer
    - Characteristics: Combines depth and width, highest capacity for learning, but requires careful tuning to avoid oeverfitting.

To understand the impact of different activation functions, we conducted several experiments using identical conditions, wich are, again, those presented above in the **Base model summary**, but this time by modifying the Number and Size of Layers of the model, and using the ReLU activation function and the He Initialization for every tests.

**Results**

| Number/Size of the layer | Training Accuracy | Testing Accuracy | Cost before training | Cost after training  |
|--------------------------|-------------------|------------------|----------------------|----------------------|
| Shallow Network          | 100%              | 70.00%           | 0.68                 | 0.05                 |
| Deep Network             | 98.09%            | 56.00%           | 0.68                 | 0.10                 |
| Wide Network             | 100%              | 78.00%           | 0.69                 | 0.01                 |
| Deep and wide Network    | 100%              | 80.00%           | 1.21                 | 0.08                 | 

**Discussion**

 - Shallow Network : Provided descent performance but could be improved a lot.

 - Deep Network : The ReLU activation function appears to have caused the "dying ReLU" problem, significantly decreasing the model's accuracy by zeroing out neurons with negative values. The deeper the network, the more negative values can occur, leading to a substantial number of inactive neurons. This issue is highlighted by the fact that using the Tanh activation function with the same number of layers resulted in an accuracy improvement of over 20%, even with He initialization. In this case, employing the Leaky ReLU activation function could be a potential solution. Aside from this issue, the deep network with the Tanh activation function performs better than the shallow network, demonstrating that with the right approach, a deep network usually outperforms a shallow one. 

 - Wide Network : The wider network imrpoved testing accuracy compared to the shallow and deep networks, indicating that the additional neuroons helped in capturing more features without more overtiffing.

 - Deep and wide Network : This network configuration performed the best on the testing set, showing that a combination of depth and width helped a lot in learning more complex patterns and generalizing better to unseend data.

  
 **Number and Size of Layers : conclusion**

 The results highlight the importance of balancing the depths and width of the network. While deeper networks can capture more complex patterns, they are prone to overfitting and to other specific problems like dying ReLU which can deacreases significantly the accuracy of the model. Wider networks can help in capturing more features, but there is a trade-off between complexity and overfitting. The deep and wide network configuration proovided the best results, sugesting that a balanced approach in increasing both the depth and width of the network can lead to better generalization and performance.  
 By carefully selecting the number and size of layers, we can enhance the learning capacity of the model, leading to imrpoved performance on both training and testing sets.  
 Building an L-layer model, where you can adjust the size of the model by simply modifying the values in a list(as in the model used here), is, in my opinion, an excellent approach to create a neural network that effectively fits the data you are analyzing and predicting. This flexibility allows for easy comparison of accuracies and helps in tailoring the model to achive optimal performance based on the dataset you are using.

 ### C - Conclusion

 In the first part of this project, I focused on developing a simple L-layer neural network with basic paramaters, hyperparamaters, and gradient descent implementation. I explored various parameter initializations, activation functions and network sizes. By analyzing the results for each case, I gained insights into how these factors impact the performance and the accuracy of the model, both in the training task and in the prediction task.  
 Having established a solid foundation with our basic neural network, we now move on to improving the model by incorporating regularization techniques.

## Step 2 : Regularization

In the first part, we significantly improved the performance of the basic neural network by carefully selecting different parameter values. However, despite this progress, the results revealed a major issue: overfitting. All the results showed a difference of over 20% between the training and testing accuracy, indicating that our model is clearly overfitting.  
In this section, we aim to address this problem by incorporating advanced techniques such as regularization.  
The model used for these experiments remains the same as before, utilizing ReLU and Sigmoid activation functions with He Initialization, but with the size layer_dims=[12288, 20, 7, 5, 1] (which has shown the best performance so far).
 
### A - Model improvement

Below are presented the different regularization techniques used in order to prevent the model from overfitting and to improve it performances.

#### I - <u>L2 regularization</u>

This technique helps to prevent overfitting by adding a penalty term to the cost function, which discourages the model from relying too heavily on any one parameter. It helps in keeping the weights small, thereby imrpoving generalization.

#### II - <u>Dropout</u>

Dropout is a regularization method where, during training, a fraction of the neuroons are randomly ignored in each forward and backward pass. This prevents the network from becoming overly relient on specific paths and encourages it to learn more robust features.

By implementing these techniques, we aim to improve generalization by helping the model generalize better to unseen data by preventing overfitting. Through these imrpovements, we will build a more robut neural network capable of delivering superior performances on both training and testing datasets.

### B - Results analysis

Here's a reminder of the model we use to perform our tests and adjustements :

**Base Model Summary**

| Aspect                 | Description                                   |
|------------------------|-----------------------------------------------|
| Model Type             | 4-layer Neural Network                        |
| Training Input Shape   | (12288, 209)                                  |
| Testing Input Shape    | (12288, 50)                                   |
| Hidden Layer Activation| ReLU                                          |
| Output Layer Activation| Sigmoid                                       |
| Number of Iterations   | 3000                                          |
| Learning Rate (alpha)  | 0.0075                                        |
| Hidden layer size      | 20, 7, 5                                      |

#### I - <u>L2 regularization</u>