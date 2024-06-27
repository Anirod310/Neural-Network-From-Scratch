from preprocess_data import *
from base_model import *
from predictions import *




X_train_orig, Y_train, X_test_orig, Y_test, classes = load_data()
num_px = X_train_orig.shape[1]
X_train, X_test = reshape_standardize_train_test_examples(X_train_orig, X_test_orig)



layer_dims = [12288, 20, 7, 5, 1]

parameters, cost= nn_model_base(X_train, Y_train, layer_dims, 3000, 0.0075)
print("cost at the end", cost)


train_set_accuracy = accuracy(X_train, Y_train, parameters)
print(f'accuracy of the model on the training set : {train_set_accuracy * 100:.2f}%')

test_set_accuracy = accuracy(X_test, Y_test, parameters)
print(f'accuracy of the model on the test set : {test_set_accuracy * 100:.2f}%')

predict_cat_noncat("db3.jpg", num_px, parameters, classes)