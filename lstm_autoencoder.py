from collections.abc import Iterable
import random
from class_basesimulation import BaseSimulation
from class_simulationhelper import SimulationHelpers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras


def random_perturb_1(x: Iterable):
    # on geom brownian
    return 0.0001 * (np.random.rand(len(x)) * 2 - 1) + 0.001 * (
            np.cos(x) + np.sin(x - 0.01)
    )


class DataGeneration:
    def __init__(self, total_time=10000, seq_size=10, n_feature=3):
        self.total_time = total_time  # length of simulation duration
        self.seq_size = seq_size  # length of looking back
        self.n_feature = n_feature
        self.sim = BaseSimulation()


    def to_sequences(self):
        seq_normal = []
        seq_outlier = []
        for i in range(len(self.data_normal) - self.seq_size):
            seq_normal.append(self.data_normal[i:i + self.seq_size])
            seq_outlier.append(self.data_outlier[i:i + self.seq_size])
        return np.array(seq_normal), np.array(seq_outlier)

    def multi_data(self):
        helper = SimulationHelpers()
        sigma = 0.02
        Sig = helper.gen_rand_cov_mat(
            self.n_feature,
            # sigma = sigma
        )

        data = self.sim.correlated_brownian_process(n=self.total_time, mu=0, cov_mat=Sig, S0=100).T

        X = temporalize(X=data, seq_size=self.seq_size)

        X = np.array(X)
        X = X.reshape(X.shape[0], self.seq_size, self.n_feature)

        return X


class MyModel(tf.keras.Model):
    def __init__(self, seq_size, n_features):
        super().__init__()
        self.seq_size = seq_size
        self.lstm1 = tf.keras.layers.LSTM(128, activation=tf.nn.tanh, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(64, activation=tf.nn.tanh, return_sequences=False)
        self.repeat_v = tf.keras.layers.RepeatVector(self.seq_size)
        self.lstm3 = tf.keras.layers.LSTM(64, activation=tf.nn.tanh, return_sequences=True)
        self.lstm4 = tf.keras.layers.LSTM(128, activation=tf.nn.tanh, return_sequences=True)
        self.time_distribute = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.repeat_v(x)
        x = self.lstm3(x)
        x = self.lstm4(x)
        x = self.time_distribute(x)
        return x


def temporalize(X, seq_size):
    output_X = []

    for i in range(len(X) - seq_size + 1):
        output_X.append(X[i:i + seq_size, :])

    return output_X


def reconstruction(seq_data, n_features, seq_size):
    multi = []
    for i in range(n_features):
        uni_seq = seq_data[:, :, i]

        uni = np.array([])
        j = 0

        for j in range(len(uni_seq)):
            uni = np.append(uni, uni_seq[j, 0])

        print(uni)
        uni = np.append(uni, uni_seq[-1, 1:])
        multi.append(uni)

    multi = np.array(multi)
    return multi


if __name__ == "__main__":
    # system setup
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
    random.seed(10)
    helper = SimulationHelpers()

    # parameters
    total_time = 10000
    seq_size = 10
    n_feature = 3

    # data

    d = DataGeneration(total_time=total_time, seq_size=seq_size)
    x_normal = d.multi_data()

    # model training and prediction
    model = MyModel(seq_size, n_feature)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_normal, x_normal, epochs=80, batch_size=512)
    model.save('tmp_model')

    # model prediction/reconstruction
    model = keras.models.load_model('tmp_model')
    pred = model.predict(x_normal)

    x_reconstructed = reconstruction(x_normal, n_feature, seq_size)
    pred_reconstructed = reconstruction(pred, n_feature, seq_size)

    # plotting
    helper.plot(args=x_reconstructed, preds=pred_reconstructed)
    print("done")
