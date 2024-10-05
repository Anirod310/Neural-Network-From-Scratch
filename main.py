from preprocess_data import *
from base_model import *
from predictions import *



X_train, X_test, y_train, y_test = load_data()


parameters, cost= nn_model_train(X_train, y_train, layer_dims=[57, 128, 64, 32, 16, 1], iterations=1500, alpha=0.1, lambd=0, keep_prob=0.8)
print("cost at the end", cost)


train_set_accuracy = accuracy(X_train, y_train, parameters)
print(f'accuracy of the model on the training set : {train_set_accuracy * 100:.2f}%')

test_set_accuracy = accuracy(X_test, y_test, parameters)
print(f'accuracy of the model on the test set : {test_set_accuracy * 100:.2f}%')
