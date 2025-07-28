from tensorflow.keras.datasets import mnist
import cv2
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

nn = NN()

nn.train(X_train, y_train)

y_pred = nn.pred(X_test)

print(accuracy_score(y_pred, y_test))