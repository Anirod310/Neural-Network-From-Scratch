# Deep Learning Model from scratch : Step-by-step approach

## Introduction
With the increasing capability of computing power and the advent of sophisticated machine learning techniques, deep learning has become a pivotal area in artificial intelligence research. This project aims first to explore the incremental development of a deep learning model, highlighting the impact of each improvement step-by-step, but also to help me understand and learn by myself the basics of the machine learning and deep learning area.

## Objective
The primary objective of this project is to develop a deep learning model for cat image classification, starting with a simple neural network and progressively implementing various enhancements, such as regularization techniques, to improve its performance. By documenting each step, this report serves as both an educational resource and a detailed project log.

## Structure
- **Step 1: Basic Neural Network**: Introduction of a simple neural network model.
- **Step 2: Regularization**: Implementation of regularization techniques.
- **Step 3: Advanced Techniques**: Further enhancements and final model evaluation.
- **Conclusion**: Summary of findings and potential future work.  

Without delay, let's get starded !

## Step 1 : Basic Neural Network
In this section, we build and evaluate a simple neural network model. This basic model serves as a foundation for further improvments and enhancements that will be discussed in subsequent sections.

### Model architecture

#### I - <u>Data preprocessing</u>
To begin with, the data set we're using is composed by two h5 files(representing training and testing set respectively), the traing set consisting of 209 cat or non-cat images and the testing set of 50 images. The output label is either 1(if it's a cat image) or 0(if it's a non-cat image).  
For each image, we reshape it and flatten it so that it can be used as an numpy array of dimensions **(input size, number of examples)**.

#### II - <u>Parameters, hyperparamters, forward/backward propagation, activation functions</u>

For this model, the only parameters are W and b (respectively representing weight and biase), and are initialized in 3 different ways : 
 - Initialization with zeros for both W and b.
 - Random initialization for W(scaled by 0.01) and with zeros for b
 - He initialization for W and with zeros for b

As you can see in the **main.py** file, the hyperparameters we use are the learning rate alpha, the number of iterations of the optimization loop, and the dimensions of the layer.

Also, by looking at the **base_model.py** file, you can see that the forprop and backprop steps are entirely coded from scratch, using the basic math formulas and numpy. Also, instead of coding directly all the forprop and backprop process, I implemented several steps to split all the process in smaller functions, such as the linear computing step, the activation computing step etc.

The activation functions we use are sigmoid, reLU and tanh.

We'll see after how the modification of the paramaters, hyperparameters and activation functions impacts the learning process of the model.

#### III - <u>Results analysis</u>