# -*- coding: utf-8 -*-


"""
Adaptive Traffic Light System based on DQN
Author: collins mtonga
Date: May, 2023

This implementation is based on:
- The original DQN algorithm by DeepMind (2013)
  Source: https://arxiv.org/abs/1312.5602
- The OpenAI Gym library
  Source: https://gym.openai.com/
- The Keras library
  Source: https://keras.io/
- The SUMO toolkit for traffic simulation
  Source: https://www.eclipse.org/sumo/
-RituPandes work adaptive traffic lights
  https://github.com/RituPande/DQL-TSC.git
"""
"""
Created on Mon May  1 11:29:38 2023

@author:Collins
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

class Model:

    def __init__(self, num_states, num_actions):
        # Create a Sequential model
        model = Sequential()
        
        # Input layer with 'num_states' input features
        model.add(Dense(400, input_dim=num_states, activation='relu'))
        
        # Dropout layer for regularization
        model.add(Dropout(0.2))
        
        # First hidden layer
        model.add(Dense(400, activation='relu'))
        
        # Dropout layer for regularization
        model.add(Dropout(0.2))
        
        # Second hidden layer
        model.add(Dense(200, activation='relu'))
        
        # Output layer with 'num_actions' output neurons
        model.add(Dense(num_actions, activation='linear'))
        
        # Compile the model with Mean Squared Error loss and Adam optimizer
        model.compile(loss='mse', optimizer=Adam())
        
        self.model = model

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        self.model.set_weights(w)

    def predict(self, state):
        return self.model.predict(state)

    def fit(self, x_batch, y_batch, batch_size, verbose=0):
        self.model.fit(x_batch, y_batch, batch_size, verbose=0)

    def save(self, filename):
        self.model.save(filename)
