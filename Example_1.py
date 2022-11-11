#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:45:03 2022

@author: Jocelyne Horanituze

implement a simple machine learning (ML) model with only input and output layers 
"""
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np

#=================#
# Dataset Loading #
#=================#
x_train = np.array([[0, 0, 0, 0],[0, 1, 1, 0],[0, 1, 1, 0],[0, 1, 1, 0],[1, 0, 0, 0],[1, 0, 1, 0], [1, 0, 0, 0],[1, 1, 0, 0]])
y_train = np.array([[0, 0],[0, 0],[0, 1],[0, 1],[0, 0],[1, 1],[1, 1],[1, 1]])
x_valid = np.array([[0, 1, 0, 1],[1, 1, 0, 1]])
y_valid = np.array([[0, 1],[1, 1]])
x_test = np.array([[0, 1, 0, 0],[1, 0, 1, 1]])
y_test = np.array([[0, 0],[1, 1]])
x_unknown = np.array([[0, 1, 1, 1],[1, 0, 1, 1]])

#================#
# Model Creation #
#================#

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu', name = 'input'))
model.add(Dense(2, activation='sigmoid', name = 'Output'))

model.compile(optimizer = 'adam', loss='mse', metrics=['accuracy'])
model.summary()

#=======================#
# Training & Validation #
#=======================#

# Custom callback to stop training early if the validation loss is below a certain threshold
class LossThresholdCallback(tf.keras.callbacks.Callback):
        def __init__(self, threshold):
                super(LossThresholdCallback, self).__init__()
                self.threshold = threshold
        def on_epoch_end(self, epoch, logs=None):
                if logs['val_loss'] <= self.threshold:
                        self.model.stop_training=True

print("[TRAINING]")

model.fit(x_train, 
          y_train, 
          validation_data=(x_valid, y_valid), 
          verbose=1,
          epochs=300,
          callbacks=[LossThresholdCallback(0.01)])


#=========#
# Testing #
#=========#

print("[TESTING]")
model.evaluate(x_test, y_test)

#============#
# Prediction #
#============#

print("[PREDICTION]")
predictions = model.predict(x_unknown)

for i in range(len(predictions)):
        # Prints the input and output data, as well as the rounded value of the outputs
        print(str(x_unknown[i]) + " => " + str(predictions[i]) + " => " + str(np.round(predictions[i])))
