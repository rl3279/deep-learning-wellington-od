import tensorflow as tf
import keras
import random
import numpy as np
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

    #top 1000 outliers
    ind = np.argpartition(distances, -1000)[-1000:]

    return pred, ind


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

    #top 1000 outliers
    ind = np.argpartition(distances, -1000)[-1000:]
    return pred, ind


if __name__ == "__main__":
    # system setup
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
    random.seed(10)

    # parameters
    total_time = 30000
    seq_size = 50
    n_feature = 3

    # data
    helper = SimulationHelpers()
    d = DataGeneration(total_time=total_time, seq_size=seq_size)
    x_normal = d.multi_data()
    partition_size = int(len(x_normal)*3/4)

    x_train_seq = x_normal[0: partition_size]
    x_test_seq = x_normal[partition_size:]

    x_train_whole = reconstruction(x_train_seq, n_feature, seq_size)
    x_test_whole = reconstruction(x_test_seq, n_feature, seq_size)

    # x_reconstructed = reconstruction(x_normal, n_feature, seq_size)
    #
    #lstm_pred, lstm_outliers = lstm_run(x_train_seq, x_test_seq)
    dense_pred, dense_outliers = dense_run(x_train_whole, x_test_whole)

    #lstm_pred_reconstructed = reconstruction(lstm_pred, n_feature, seq_size)
    helper.plot(args=x_test_whole.T, preds=dense_pred.T, markers=dense_outliers)




















    # model training and prediction
    # model = LSTM_Model(seq_size, n_feature)
    # model.compile(optimizer='adam', loss='mse')
    # model.fit(x_normal, x_normal, epochs=80, batch_size=512)
    # model.save('tmp_model')

    # model prediction/reconstruction
    # model = keras.models.load_model('tmp_model')
    # pred = model.predict(x_normal)
    #
    # x_reconstructed = reconstruction(x_normal, n_feature, seq_size)
    # pred_reconstructed = reconstruction(pred, n_feature, seq_size)
    #
    # distances = pairwise_distances_no_broadcast(x_reconstructed, pred_reconstructed)
    # ind = np.argpartition(distances, -1000)[-1000:]
    #
    # print(ind)
    # print(distances[ind])
    # # plotting
    # helper.plot(args=x_reconstructed.T, preds=pred_reconstructed.T, markers=ind)
    # print("done")