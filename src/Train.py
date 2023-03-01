#!/usr/bin/env python
# coding: utf-8

## This Class contain Trin  functions


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU
from tensorflow.keras import losses, optimizers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

from Processing import ImageProcessor

class CNNModel:
    def __init__(self, input_shape=(100, 70, 3)):
        self.model = Sequential()
        self.model.add(Conv2D(32, (2, 2), activation=ReLU(), padding='same', input_shape=input_shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation=ReLU()))
        self.model.add(Dense(1, activation='sigmoid'))
        adam = optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss=losses.binary_crossentropy, optimizer=adam, metrics=['accuracy'])
    
    def train(self, X_train, y_train, X_val, y_val, epochs=12, batch_size=64, verbose=1):
        self.history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), shuffle=True, epochs=epochs, batch_size=batch_size, verbose=verbose)

def run_training():
    processor = ImageProcessor()

    X_noflip =np.array(processor.process('../images/training/0'))
    X_flip = np.array(processor.process('../images/training/1'))

    y_noflip = np.zeros(X_noflip.shape[0])
    y_flip = np.ones(X_flip.shape[0])

    X = np.concatenate((X_noflip,X_flip ))
    y = np.concatenate((y_noflip, y_flip))

    #########
    X_train, X_val, y_train, y_val = train_test_split(X, y , test_size = 0.2, shuffle=True)

    cnn = CNNModel()
    cnn.train(X_train, y_train, X_val, y_val)
    model.save("../models/model_CNN_02.h5")

if __name__ == '__main__':
    run_training()