from preprocess_data import *
from model import *
from PIL import Image


X_train_orig, Y_train, X_test_orig, Y_test, classes = load_data()
num_px = X_train_orig.shape[1]
X_train, X_test = reshape_standardize_train_test_examples(X_train_orig, X_test_orig)


def accuracy(X, Y, parameters):
    m = X.shape[1]

    AL, caches = forward_prop(X, parameters)
    predictions = (AL > 0.5)

    accuracy = np.sum(predictions == Y) / m

    return accuracy

def predict(X, parameters):
    m = X.shape[1]

    AL, caches = forward_prop(X, parameters)
    prediction = (AL > 0.5)

    return prediction


layer_dims = [12288, 20, 7, 5, 1]

test_set_accuracy = 0
while test_set_accuracy < 0.8:
    parameters, cost= nn_model(X_train, Y_train, layer_dims, 3000, 0.0075)
    print("cost at the end", cost)

    train_set_accuracy = accuracy(X_train, Y_train, parameters)
    print(f'accuracy of the model on the training set : {train_set_accuracy * 100:.2f}%')

    test_set_accuracy = accuracy(X_test, Y_test, parameters)
    print(f'accuracy of the model on the test set : {test_set_accuracy * 100:.2f}%')


image_to_predict = "db3.jpg"
my_label_y = [1]

fname = "images/" + image_to_predict
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
plt.show()
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T

predicted_image = int(predict(image, parameters))

print ("y = " + str(np.squeeze(predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(predicted_image)),].decode("utf-8") +  "\" picture.")