import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

from NN import NeuralNetwork

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train_flat = X_train.reshape(X_train.shape[0], 784)
X_test_flat = X_test.reshape(X_test.shape[0], 784)

y_for_nn = np.eye(10)[y_train]

model_index = 4

try:
    with open(f'./models/model_{model_index-1}.pickle', 'rb') as f:
        nn = pickle.load(f)
    print("Loaded model from file.")
except FileNotFoundError:
    print("No saved model found. Initializing a new one.")
    nn = NeuralNetwork(layers=[784, 128, 64, 10])

nn.train(X_train_flat, y_for_nn, epochs=20)

with open(f'./models/model_{model_index}.pickle', 'wb') as f:
    pickle.dump(nn, f)
    print(f"Model saved to ./models/model_{model_index}.pickle")

y_pred = nn.predict(X_test_flat)
