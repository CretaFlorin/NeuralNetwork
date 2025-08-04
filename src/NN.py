import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score

# Activation functions and derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Xavier Initialization
def xavier_init(in_dim, out_dim):
    limit = np.sqrt(6 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size=(in_dim, out_dim))

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        # Initialize weights and biases with Xavier init
        for i in range(len(layers) - 1):
            self.weights.append(xavier_init(layers[i], layers[i+1]))
            self.biases.append(np.zeros((1, layers[i+1])))

    def feed_forward(self, X):
        activations = [X]
        pre_activations = []

        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            a = relu(z)
            activations.append(a)

        # Output layer
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        pre_activations.append(z)
        a = softmax(z)
        activations.append(a)

        return activations, pre_activations

    def back_propagation(self, X, y, activations, pre_activations, lr):
        m = X.shape[0]
        delta = activations[-1] - y  # output layer error

        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            dw = np.dot(a_prev.T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

            if i > 0:
                dz = np.dot(delta, self.weights[i].T)
                delta = dz * relu_derivative(pre_activations[i-1])

    def train(self, X, y, epochs=10, batch_size=64, lr=0.01):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                activations, pre_activations = self.feed_forward(X_batch)
                self.back_propagation(X_batch, y_batch, activations, pre_activations, lr)

                batch_loss = -np.mean(np.sum(y_batch * np.log(activations[-1] + 1e-9), axis=1))
                epoch_loss += batch_loss * X_batch.shape[0]

            epoch_loss /= n_samples
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    def predict(self, X):
        activations, _ = self.feed_forward(X)
        return np.argmax(activations[-1], axis=1)



