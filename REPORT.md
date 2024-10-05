# Deep Learning Model from scratch : Step-by-step approach

## Introduction
With the increasing capability of computing power and the advent of sophisticated machine learning techniques, deep learning has become a pivotal area in artificial intelligence research. This project aims first to explore the incremental development of a deep learning model, highlighting the impact of each improvement step-by-step, but also to help me understand and learn by myself the basics of the machine learning and deep learning area.

## Objective
The primary objective of this project is to develop a deep learning model for spam mail detection, starting with a simple neural network and progressively implementing various enhancements to improve its performance. By documenting each step, this report serves as both an educational resource and a detailed project log.

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
To begin with, the dataset we're using was created by George Forman and is publicly available through the UCI Machine Learning Repository and other sources(In in this project we import it from **sklearn.datasets** library). It contains 4,601 emails, and there are 57 attributes in the dataset (that explains the input size of **(57, num_examples)**), which include:
 - 48 continuous features representing the frequency of specific words in the emails.
 - 1 binary class attribute indicating whether the email is spam(1) or not spam(0).
 - 8 additional features that indicates the presence of various types of characters in the email.

 The dataset is preprocessed, including the normalization of the values, and the split of the data into training and testing set. The size of the training set is **(57 features, 3680 examples)** and the size of the testing set is **(57 features , 921 examples)**.

#### II - <u>Parameters, hyperparamters, forward/backward propagation, activation functions</u>

For this model, the only parameters are W and b (respectively representing weight and biase), and are initialized in 4 different ways : 
 - Initialization with zeros for both W and b.
 - Random initialization for W(scaled by 0.01) and with zeros for b.
 - Xavier initialization for W and with zeros for b.
 - He initialization for W and with zeros for b.


As you can see in the [main](main.py) file, the hyperparameters/args we use are the learning rate alpha, the number of iterations of the optimization loop, and the dimensions of the layer.

Also, by looking at the [base_model](base_model.py) file, you can see that the forprop and backprop steps are entirely coded from scratch, using the corresponding math formulas and numpy. Also, instead of coding directly all the forprop and backprop process in one function, I implemented several steps to split all the process in smaller functions, such as the linear computing step, the activation computing step, etc.

The activation functions we use are sigmoid, reLU and tanh.

We'll see after how the modification of the parameters, hyperparameters, number and size of layers and activation functions impacts the learning process of the model.

### B - Results analysis

To evaluate the differences among various parameter initializations, hyperparameters, and activation functions, I conducted a series of experiments. I first varied the parameter initialization methods and compared their performances. Next, I explored different activation functions, followed by an examination of various hyperparameters.
To finish with, I varied the size and the number of layers and compared resulting performances. 

The base model used is a 2-layer neural network. The training and testing inputs have shapes of **(57, 3680)** and **(57, 921)**, respectively. The activation function for the hidden layer is ReLU and for the output layer, it is sigmoid. The model was trained for 5000 iterations with a learning rate (alpha) of 0.01. The size of the hidden layer is 4.

**Base Model Summary**

| Aspect                 | Description                                   |
|------------------------|-----------------------------------------------|
| Model Type             | 2-layer Neural Network                        |
| Training Input Shape   | (57, 3680)                                    |
| Testing Input Shape    | (57, 921)                                     |
| Hidden Layer Activation| ReLU                                          |
| Output Layer Activation| Sigmoid                                       |
| Number of Iterations   | 5000                                          |
| Learning Rate (alpha)  | 0.01                                          |
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
|------------------------|-------------------|------------------|----------------------|---------------------|
| Zero Initialization    | 61.33%            | 58%              | 0.69                 | 0.67                |
| Random Initialization  | 92.91%            | 93%              | 0.69                 | 0.19                |
| Xavier Initialization  | 93.10%            | 93%              | 0.70                 | 0.19                |
| He Initialization      | 93.29%            | 93%              | 0.72                 | 0.19                |

**Discussion**

- Zero Init : As expected, zero initialization led to poor performance due to symetry problems.
- Random Init : This method showed decent performance on this dataset but is often outperformed by more sophisicated methods.
- Xavien Init : In this experiment, it provided same performances that the random initialization, but it's mostly  because the Xavier initialization performs better on deeper networks.
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
| Sigmoid activation     | 91.66%            | 92.62%           | 0.68                 | 0.27                 |
| Tanh activation        | 90.82%            | 90.55%           | 0.71                 | 0.19                 |
| ReLU activation        | 93.10%            | 92.62%           | 0.72                 | 0.19                 | 

**Discussion**

- Sigmoid : While it provided decent performance, the model converged more slowly compared to other activation functions due to the vanishing gradient problem.

- Tanh : Showed worst performance than sigmoid, probably because of the vanishing gradient problem and the fact that we want values between 0 and 1 for binary classification, but the tanh is 0 centered, so the values are between -1 and 1. 

- ReLU : Achieved the highest performance, particulary effective in promoting sparsity and speeding up convergence. However, some neurons might became inactive due to the zeroing effect.

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
| Shallow Network          | 93.10%            | 92.62%           | 0.72                 | 0.19                 |
| Deep Network             | 94.21%            | 93.70%           | 0.72                 | 0.17                 |
| Wide Network             | 94.16%            | 93.70%           | 0.74                 | 0.16                 |
| Deep and wide Network    | 97.91%            | 93.92%           | 0.84                 | 0.07                 | 

**Discussion**

 - Shallow Network : Provided descent performance but could be improved a lot.

 - Deep Network : Performed better that the Shallow Network, but still could be imrpoved. 

 - Wide Network : Performed as well as the deep network.

 - Deep and wide Network : This network configuration performed the best on the testing set, and is far beyond the previous configurations, showing that a combination of depth and width helped a lot in learning more complex patterns and generalizing better to unseend data. However, we begin to see some overfitting problem (difference between the training and testing accuracy is 4%).

  
 **Number and Size of Layers : conclusion**

 The results highlight the importance of balancing the depths and width of the network. While deeper networks can capture more complex patterns, they are prone to overfitting and to other specific problems like dying ReLU which can deacreases significantly the accuracy of the model. Wider networks can help in capturing more features, but there is a trade-off between complexity and overfitting. The deep and wide network configuration proovided the best results, sugesting that a balanced approach in increasing both the depth and width of the network can lead to better generalization and performance, even if the overfitting problem is still present.  
 By carefully selecting the number and size of layers, we can enhance the learning capacity of the model, leading to improved performance on both training and testing sets.  
 Building an L-layer model, where you can adjust the size of the model by simply modifying the values in a list(as in the model used here), is, in my opinion, an excellent approach to create a neural network from scratch that effectively fits the data you are analyzing and predicting. This flexibility allows for easy comparison of accuracies and helps in tailoring the model to achive optimal performance based on the dataset you are using.

 ### C - Conclusion

 In the first part of this project, I focused on developing a simple L-layer neural network with basic paramaters, hyperparamaters, and gradient descent implementation. I explored various parameter initializations, activation functions and network sizes. By analyzing the results for each case, I gained insights into how these factors impact the performance and the accuracy of the model, both in the training task and in the prediction task.  
 Having established a solid foundation with our basic neural network, we now move on to improving the model by incorporating regularization techniques.

## Step 2 : Regularization

In the first part, we significantly improved the performance of the basic neural network by carefully selecting different parameter values. However, despite this progress, the results revealed a major issue: overfitting. Even the best results showed a difference of over 4% between the training and testing accuracy, indicating that our model is clearly overfitting.  
In this section, we aim to address this problem by incorporating advanced techniques such as regularization.  
The model used for these experiments remains the same as before, utilizing ReLU and Sigmoid activation functions with He Initialization, but with the size **layer_dims=[57, 128, 64, 32, 16, 1]** (which has shown the best performance so far).
 
### A - Model improvement

Below are presented the different regularization techniques used in order to prevent the model from overfitting and to improve its performances.

#### I - <u>L2 regularization</u>

This technique helps to prevent overfitting by adding a penalty term to the cost function, which discourages the model from relying too heavily on any one parameter. It helps in keeping the weights small, thereby improving generalization.

#### II - <u>Dropout</u>

Dropout is a regularization method where, during training, a fraction of the neuroons are randomly ignored in each forward and backward pass. This prevents the network from becoming overly relient on specific paths and encourages it to learn more robust features.

By implementing these techniques, we aim to improve generalization by helping the model generalize better to unseen data by preventing overfitting. Through these improvements, we will build a more robut neural network capable of delivering superior performances on both training and testing datasets.

### B - Results analysis

Here's a reminder of the model we use to perform our tests and adjustements :

**Base Model Summary**

| Aspect                 | Description                                   |
|------------------------|-----------------------------------------------|
| Model Type             | 5-layer Neural Network                        |
| Training Input Shape   | (57, 3680)                                    |
| Testing Input Shape    | (57, 921)                                     |
| Hidden Layer Activation| ReLU                                          |
| Output Layer Activation| Sigmoid                                       |
| Number of Iterations   | 5000                                          |
| Learning Rate (alpha)  | 0.01                                          |
| Hidden layer size      | 128, 64, 32, 16                               |

**Results**

| Regularization method | Training Accuracy | Testing Accuracy |
|-----------------------|-------------------|------------------|
| None                  | 97.91%            | 93.92%           |
| L2 regularization     | 97%               | 94%              |
| Dropout               | 94.35%            | 94%              |

**Discussion**

The results of the experiments demonstrated that applying L2 regularization has not a significant impact in the reduction of the model overfitting problem. 

However, the dropout method significantly reduces the model's tendency to learn noise or overly specific details from the training set. This allows the model to generalize better to new data, leading to improved accuracy on the testing dataset. We reduced the difference between training and testing accuracy from 4% to almost 0% with dropout—showcasing the effectiveness of regularization.

### C - Conclusion

In the second part of this project, I delved into more advanced concepts, specifically regularization techniques, to enhance the performance of the base neural network model. The primary goal was to mitigate overfitting to the training data and improve the model's efficiency on unseen data. Regularization proved to be indispensable in this context, particularly given the small dataset used in this project.  

Techniques like dropout and L2 regularization are not only straightforward to implement but also require minimal adjustments to the existing neural network structure. Despite their simplicity, these methods significantly boost the model's generalization capability (especially with dropout in this application), reducing the gap between training and testing accuracy. This demonstrates that even basic regularization techniques can have a profound impact on model performance, making them essential tools in the deep learning toolkit, especially when working with limited data.  

As we move forward, it's crucial to consider these regularization methods as fundamental steps in the development of robust neural networks, particularly when aiming to build models that perform well in real-world scenarios where data is often scarce or noisy.

## Step 3 : Optimization algorithms

Now that we have found the right hyperparameters, layer dims and that we implemented successfully some regularization techniques, one of the final ( and often crucial and most impactful) part in the developpement of a better neural network is the use of a different optimizer than the simple gradient descent. While the basic gradient descent, better known as the Stochastic gradient descent(SGD), achieves quite good performances overall, some optimisation techniques can increase a lot the convergence of our model during training.

In this example, we'll implement two of the most famous optimization techniques, still used today in the state-of-the-art deep learning model : the momentum optimizer and the adam optimizer.

### A - Optimizers presentation

#### I - <u>Momentum</u>

The momentum optimizer is an extension of the Stochastich Gradient Descent (SGD) method that aims to improve the speed and efficency of learning. In regular SGD, the updates are based on the current gradient of the loss function, but momentum incorporates a running average of past gradients to accelerate updates in the relevant direction. 

How it works? It introduces a velocity term, which accumulates the gradients of the past steps. Updates to the parameters are then influenced not only by the current gradient but also by the previous velocity. This optimizer is beneficial when you want to achieve faster convergence on tasks with steep and narrow regions in the loss function.

#### II - <u>Adam</u>

The adam optimizer (Adaptive Moment Estimation) is one of the most popular optimizers due to its adaptative learning rate properties. It combines ideas from both Momentum and RMSProp optimizers, keeping track of the first and second moments (mean and uncentered variance) of the gradients.

How it works? It maintains two running averages for each parameter : 
- first moment (mean) estimate : Similar to momentum, it calculates the exponentially weighted average of the gradients.
- second moment (variance) estimate : It computes the exponentially weighted average of the squared gradients(like RMSProp).  

Adam is widely used for tasks where quick convergence and adaptative learning rate are essential, such as in deep learning tasks involving large and/or complex datasets. It's often a great starting point when you are unsure which optimizer to choose.

### B - Results analysis

Now that we understand better what is momentum and adam, we will implement them one by one and analyze the results. As usual, here's a reminder of the model architecture we'll use for these tests(we're not using any regularization technique in the following tests unless it's specified ) : 

**Base Model Summary**

| Aspect                 | Description                                   |
|------------------------|-----------------------------------------------|
| Model Type             | 5-layer Neural Network                        |
| Training Input Shape   | (57, 3680)                                    |
| Testing Input Shape    | (57, 921)                                     |
| Hidden Layer Activation| ReLU                                          |
| Output Layer Activation| Sigmoid                                       |
| Number of Iterations   | 5000                                          |
| Learning Rate (alpha)  | 0.01                                          |
| Hidden layer size      | 128, 64, 32, 16                               |


**Results**

*Note : For the adam optimizer, we changed the learning rate to 0.001, since this is the value well-known to outperforms the other ones by far in most cases. We also decreased the number of iteration to 2500, since the adam optimizer is also well-known for working well with the early stopping method.*

| Optimizer               | Training Accuracy | Testing Accuracy |
|-------------------------|-------------------|------------------|
| SGD                     | 97.91%            | 93.92%           |
| Momentum                | 98,31%            | 94%              |
| Adam                    | 99.81%            | 94,03%           |
| Adam + Dropout + L2 Reg | 99.02%            | 96.20%           |



**Discussion**

From the table, it's clear that the choice of the optimizer, along wirh regularization techniques, significantly affects both training and testing accuracy.

- SGD achieves respectable results, However, like before, it shows a slight gap between training and testing accuracy, indicating a bit of overfitting. This is expected as SGD does not incorporate momentum or any adaptative learning techniques, leading to slower convergence and less ability to escape local minima.

- Momentum improves upon SGD. By using past gradients to accelerate convergence, it reduces the generalization gap slightly compared to SGD, but overfitting is still present.

- Adam performs way better especially in terms of training accuracy, but the testing accuracy isn't mooving. This suggests that Adam may be prone to overfitting in this case, as it adapts the learning rate individually for each parameter, which can lead to very efficient learning of the training set but less generalization on the test set.

- Adam with Dropout and Regularization achives a way more balanced performance. The addition of Dropout and L2 regularization helps mitigate overfitting, and thus enhancing the model's ability to generalize to unseen data.

In summary, on this dataset, while Adam alone results in—by far—the highest training accuracy, it tends to overfit. But by combining Adam to regularization techniques (here Dropout, L2 regularization and early stopping), the model strikes a better balance between training and testing accuracy, making it the most effective approach for this particular task.

## Final Conclusion

**Throughout this project, we explored various ways to improve the performance of a simple L-layer neural network**. Starting with basic gradient descent and hyperparameter tuning, we systematically tested different parameter initializations, activation functions, network architectures, regularization techniques, and optimizers.

- **Parameter Initialization**: Among the different methods we tried, He initialization consistently provided the best results. It enabled faster convergence and prevented the vanishing gradient problem, particularly when using deeper networks.

- **Activation Functions**: ReLU (Rectified Linear Unit) proved to be the best activation function for this task. It outperformed other functions like sigmoid and tanh by promoting faster learning and avoiding issues like saturation, which are common with other activation functions.

- **Network Architecture**: A wide and deep network—one with more layers and nodes—yielded the best performance. This configuration enabled the model to capture more complex patterns in the data, but it also increased the risk of overfitting, which we addressed with regularization techniques.

- **Regularization Techniques**: Both L2 regularization and Dropout were critical to improving the model’s generalization performance. These methods successfully reduced overfitting by penalizing large weights and randomly deactivating neurons during training, ensuring the network didn't overly rely on specific neurons or weights.

- **Optimizers**: Finally, the implementation of advanced optimization techniques significantly enhanced performance. While Momentum showed some improvement over basic gradient descent, the Adam optimizer combined with regularization techniques (L2 and Dropout) was by far the most effective. Adam’s adaptive learning rates, coupled with regularization, struck the best balance between fast convergence and generalization, yielding the highest testing accuracy.

**In conclusion**, for this particular task, the combination of He initialization, ReLU activation, a deep and wide network, regularization, and the Adam optimizer resulted in the most optimal model performance. These methods, together, provided a well-rounded and efficient neural network capable of accurately learning from the data while avoiding overfitting.

**I would like to extend my thanks** to everyone who have read this far the report of this project. The insights gained here have been invaluable and have laid the groundwork for future improvements.

**Looking ahead**, the next project will focus on Convolutional Neural Networks (CNNs), where I’ll dive into tasks involving image data. I'll explore how CNN architectures, such as pooling layers and convolutions, can extract features from images, and I'll experiment with different CNN configurations to push the boundaries of performance. **Stay tuned! :)**