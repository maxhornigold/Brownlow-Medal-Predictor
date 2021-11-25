
#import pandas as pd
#import numpy as np

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error

#import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import CategoricalCrossentropy, MSE, MeanSquaredError

###############################################################################

def create_model(optimizer = 'Adam', loss = 'Categorical Crossentropy'):

    # create the model
    model = models.Sequential()
    model.add(layers.Dense(20, input_dim=55, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(7, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    # define optimizer, learning rate and momentum
    optimizer = Adam()

    # define loss function
    loss = CategoricalCrossentropy(from_logits=True)

    # compile the model
    model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=['accuracy'])
    
    return model

###############################################################################

def train_model(model, x_tr, y_oh_tr, num_epochs, batch_size, validation_split):

    # set class weights
    #class_weights = {0: 1, 1: 5, 2: 5, 3: 5}    

    # train the model on the training data
    history = model.fit(x_tr, y_oh_tr,
                            epochs=num_epochs,
                            batch_size=batch_size,
                            validation_split=validation_split)
    
    return history





