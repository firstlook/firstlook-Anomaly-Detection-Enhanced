from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#学習データ
x_train_s, x_test_s, x_test_b = [], [], []
x_ref, y_ref = [], []

x_train_shape = x_train.shape


for i in range(len(x_train)):
    if y_train[i] == 7:#スニーカーは7
        temp = x_train[i]
        x_train_s.append(temp.reshape((x_train_shape[1:])))
    else:
        temp = x_train[i]
        x_ref.append