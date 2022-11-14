import tensorflow as tf
import keras
import random
import numpy as np
from data_generation import gen_data
from pyod.utils import pairwise_distances_no_broadcast
from dense_autoencoder import DENSE_Model
from lstm_autoencoder import LSTM_Model
from class_simulationhelper import SimulationHelpers
from lstm_autoencoder import DataGeneration
from lstm_autoencoder import reconstruction


def lstm_run(train_data, test_data):
    # model training and prediction
    model = LSTM_Model(seq_size, n_feature)
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_data, train_data, epochs=80, batch_size=512)
    model.save('lstm_model')

    # model prediction/reconstruction
    model = keras.models.load_model('lstm_model')
    pred = model.predict(test_data)

    test_reconstructed = reconstruction(test_data, n_feature, seq_size)
    pred_reconstructed = reconstruction(pred, n_feature, seq_size)

    distances = pairwise_distances_no_broadcast(test_reconstructed, pred_reconstructed)

    # top 1000 outliers
    ind = np.argpartition(distances, -1000)[-1000:]

    return pred_reconstructed, ind


def dense_run(train_data, test_data):
    # model training and prediction
    model = DENSE_Model(n_feature)
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_data, train_data, epochs=80, batch_size=512)
    model.save('dense_model')

    # model prediction/reconstruction
    model = keras.models.load_model('dense_model')
    pred = model.predict(test_data)
    print(pred.shape, test_data.shape)

    distances = pairwise_distances_no_broadcast(test_data, pred)

    # top 1000 outliers
    ind = np.argpartition(distances, -1000)[-1000:]
    return pred, ind


def temporalize(X, seq_size):
    # break data into seq_size
    output_X = []

    for i in range(len(X) - seq_size + 1):
        output_X.append(X[i:i + seq_size, :])

    return np.array(output_X)


def reconstruction(seq_data, n_features, seq_size):
    multi = []
    for i in range(n_features):
        uni_seq = seq_data[:, :, i]

        uni = np.array([])
        j = 0

        for j in range(len(uni_seq)):
            uni = np.append(uni, uni_seq[j, 0])

        uni = np.append(uni, uni_seq[-1, 1:])
        multi.append(uni)

    multi = np.array(multi)
    return multi.T


if __name__ == "__main__":
    # system setup
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

    total_time = 30000
    seq_size = 50
    n_feature = 3

    data = gen_data(10).to_numpy()

    partition_size = int(len(data) * 3 / 4)

    data_train = data[0: partition_size]
    data_test = data[partition_size:]

    data_train_seq = temporalize(data_train, seq_size)
    data_test_seq = temporalize(data_test, seq_size)

    lstm_pred, lstm_outliers = lstm_run(data_train_seq, data_test_seq)
    dense_pred, dense_outliers = dense_run(data_train, data_test)

    print(f"Data test shape {data_test.shape}")
    print(f"lstm_pred shape {lstm_pred.shape}")
    print(f"dense_pred shape {dense_pred.shape}")

    # plot the curves
    # TODO


