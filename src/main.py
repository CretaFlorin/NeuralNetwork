from tensorflow.keras.datasets import mnist
import cv2
import numpy as np
import pickle

from NN import NeuralNetwork
from canvas import DrawingCanvas

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# nn = NeuralNetwork(layers=[28*28, 16, 10])
nn = None

with open('./models/model_1.pickle', 'rb') as f:
    nn = pickle.load(f)


y_for_nn = []
for y in y_train:
    l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    l[y] = 1
    y_for_nn.append(l)
y_for_nn = np.array(y_for_nn)


nn.train(X_train.reshape(X_train.shape[0], 784), y_for_nn, 20)

with open('./models/model_1.pickle', 'wb') as f:
    pickle.dump(nn, f)


# y_pred = nn.pred(X_test)

# print(accuracy_score(y_pred, y_test))

# if __name__ == "__main__":
#     dc = DrawingCanvas()
#     dc.draw()
