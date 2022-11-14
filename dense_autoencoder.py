from collections.abc import Iterable
import random
from class_basesimulation import BaseSimulation
from class_simulationhelper import SimulationHelpers
from pyod.utils.stat_models import pairwise_distances_no_broadcast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras


class DENSE_Model(tf.keras.Model):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

        self.layers1 = keras.layers.Dense(64, input_shape=(self.n_features,), activation = 'relu')
        self.layers2 = keras.layers.Dense(32, activation='relu')
        self.layers3 = keras.layers.Dense(32, activation='relu')
        self.layers4 = keras.layers.Dense(64, activation='relu')
        self.output_layer = keras.layers.Dense(self.n_features)

    def call(self, inputs):

        x = self.layers1(inputs)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.output_layer(x)
        return x



