import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from tensorflow.keras.losses import CategoricalCrossentropy


def sigmoid_elem(x):
    if x > 10:
        return 1
    if x < -10:
        return 0
    return 1 / (1 + np.exp(-x))


def sigmoid(X):
    return np.vectorize(sigmoid_elem)(X)


def sigmoid_derivative(output):
    return output * (1 - output)


class NeuralNetwork:
    def __init__(self, layers):
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]))
            self.biases.append(np.zeros((1, layers[i + 1])))

    def feed_forward(self, X):
        activations = [X]

        for w, b in zip(self.weights, self.biases):
            X = sigmoid(np.dot(X, w) + b)
            activations.append(X)

        return activations

    def back_propagation(self, X, y, lr=0.1):
        activations = self.feed_forward(X)

        delta = (activations[-1] - y) * sigmoid_derivative(activations[-1])

        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            dw = np.dot(a_prev.T, delta)
            db = np.sum(delta, axis=0, keepdims=True)

            delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(a_prev)

            self.weights[i] -= lr * dw / X.shape[0]
            self.biases[i] -= lr * db / X.shape[0]

    def train(self, X, y, epochs=1):
        for e in range(epochs):
            self.back_propagation(X, y)

            y_pred = self.feed_forward(X)[-1]

            cce = CategoricalCrossentropy()
            loss = cce(y_pred, y)

            print(f"Epoch {e} -- loss:{loss}")

    def predict(self, X):
        return np.argmax(self.feed_forward(X)[-1])
