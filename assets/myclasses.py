#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 17:11:59 2020

@author: zhoubo
"""
import numpy as np
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels0, labels1, batch_size=32, dim=(50), n_channels=1,
                 n_classes0=47, n_classes1=9, shuffle=True, file_path='PY/datagen_2d/all/'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.labels0 = labels0
        self.labels1 = labels1
        self.n_channels = n_channels
        self.n_classes0 = n_classes0
        self.n_classes1 = n_classes1
        self.shuffle = shuffle
        self.on_epoch_end()
        self.file_path = file_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # print(indexes)
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # print(list_IDs_temp)
        # Generate data
        X, y0, y1 = self.__data_generation(list_IDs_temp)

        return X, y0, y1, list_IDs_temp

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))  # total data samples
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y0 = np.empty((self.batch_size), dtype=int)
        y1 = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(self.file_path + str(ID) + '.npy')

            # Store class
            y0[i] = self.labels0[ID]
            y1[i] = self.labels1[ID]

        return X, y0, y1