# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:43:18 2017

@author: hanar
"""

import time
import numpy as np
import os
# from keras.utils import np_utils
gpu_id = 1
import tensorflow as tf
# print(tf.__version__)
# if tf.__version__ >= "2.1.0":
#     physical_devices = tf.config.list_physical_devices('GPU')
#     tf.config.list_physical_devices('GPU')
#     tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
# from tensorflow.distribute import MirroredStrategy
#from keras.utils.visualize_util import plot
import tensorflow.keras.callbacks
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import glob
import pandas as pd
import pdb
import csv
import shutil

#Global variables for normalizing parameters
max_x = 1.0
min_x = 0.0
max_y = 1.0
min_y = 0.0

class ExcludeValue(Exception):
    pass

def deep_learning_turbidite(resdir,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            lr=0.02,
                            decay=None,
                            validation_split=0.2,
                            batch_size=2,
                            momentum=0.9,
                            nesterov=True,
                            num_layers=4,
                            dropout=0.5,
                            node_num=2000,
                            epochs=4000):
    """
    Creating the inversion model of turbidity currents by deep learning
    """
    #Normalizing dataset
#     X_train = get_normalized_data(X_train_raw, min_x, max_x)
#     X_test = get_normalized_data(X_test_raw, min_x, max_x)
#     y_train = get_normalized_data(y_train_raw, min_y, max_y)
#     y_test = get_normalized_data(y_test_raw, min_y, max_y)

    # Generate the model
    # mirrored_strategy = MirroredStrategy()
    # with mirrored_strategy.scope():
    model = Sequential()
    model.add(
        Dense(node_num,
              input_dim=X_train.shape[1],
              activation='relu',
              kernel_initializer='glorot_uniform'))  #1st layer
    model.add(Dropout(dropout))
    for i in range(num_layers - 2):
        model.add(
            Dense(node_num,
                  activation='relu',
                  kernel_initializer='glorot_uniform'))  #2nd layer
        model.add(Dropout(dropout))
    model.add(
        Dense(y_train.shape[1],
              activation='relu',
              kernel_initializer='glorot_uniform'))  #last layer

    # Compilation of the model
    model.compile(
        loss="mean_squared_error",
        # optimizer=SGD(learning_rate=lr, weight_decay=decay, momentum=momentum,
        #               nesterov=nesterov),
        optimizer=Adagrad(),
        metrics=["mean_squared_error"])

    # Start training
#     t = time.time()
#     check = ModelCheckpoint(filepath=os.path.join(resdir, "model.hdf5"),
#                             monitor='val_loss',
#                             save_freq=1000,
#                             save_weights_only=True,
#                             mode='min',
#                             save_best_only=True)
#     #es_cb = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
#     tb_cb = TensorBoard(log_dir=os.path.join(resdir, 'logs'),
#                         histogram_freq=0,
#                         write_graph=False,
#                         write_images=False)
#     history = model.fit(X_train,
#                         y_train,
#                         epochs=epochs,
#                         validation_split=validation_split,
#                         batch_size=batch_size,
#                         callbacks=[check, tb_cb])
    log_dir = os.path.join(resdir, 'logdir')
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)  # remove previous execution
    os.mkdir(log_dir)
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split,
                        callbacks=[TensorBoard(log_dir=log_dir)],
                        shuffle=True,
                        verbose=1)
    plot_history(history, resdir)
    save_history(resdir, history)
    
    return model, history

def save_history(dirpath, history):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(dirpath,'history.csv'))
  
def apply_model(model, X, min_x, max_x, min_y, max_y):
    """
    Apply the model to data sets
    """
    X_norm = (X - min_x) / (max_x - min_x)
    Y_norm = model.predict(X_norm)
    Y = Y_norm * (max_y - min_y) + min_y
    return Y


def plot_history(history, savedir):
    # plot training history
    plt.plot(history.history['mean_squared_error'], "o-",markerfacecolor='darkgreen',
                label="Training Data",
                markeredgecolor='darkgreen',
                color='darkgreen')
    plt.plot(history.history['val_mean_squared_error'], "o-", color='orange', label="validation data")
    plt.title('Model Training history', fontsize=18)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=14)
    plt.legend(loc="upper right", fontsize=14)
    fig.patch.set_alpha(0)
    plt.savefig(os.path.join(savedir, "history.svg"))

def save_history(history, dirpath):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(dirpath,'history.csv'))

def test_model(model, x_test):
    #test the model

    x_test_norm = get_normalized_data(x_test, min_x, max_x)
    test_result_norm = model.predict(x_test_norm)
    test_result = get_raw_data(test_result_norm, min_y, max_y)

    return test_result


def save_result(savedir, model=None, history=None, test_result=None):

    if test_result is not None:
        np.savetxt(os.path.join(savedir, 'test_result.txt'),
                   test_result,
                   delimiter=',')
    if history is not None:
        np.savetxt(os.path.join(savedir, 'loss.txt'),
                   history.history.get('loss'),
                   delimiter=',')
        np.savetxt(os.path.join(savedir, 'val_loss.txt'),
                   history.history.get('val_loss'),
                   delimiter=',')

    if model is not None:
        print('save the model')
        model.save(os.path.join(savedir + 'model.hdf5'))


def load_data(datadir):
    """
    This function load training and test data sets, and returns variables
    """
    global min_x, max_x, min_y, max_y

    x_train = np.load(os.path.join(datadir, 'H_train.npy'))
    x_test = np.load(os.path.join(datadir, 'H_test.npy'))
    y_train = np.load(os.path.join(datadir, 'icond_train.npy'))
    y_test = np.load(os.path.join(datadir, 'icond_test.npy'))
    min_y = np.load(os.path.join(datadir, 'icond_min.npy'))
    max_y = np.load(os.path.join(datadir, 'icond_max.npy'))
    [min_x, max_x] = np.load(os.path.join(datadir, 'x_minmax.npy'))

    return x_train, y_train, x_test, y_test


def set_minmax_data(_min_x, _max_x, _min_y, _max_y):
    global min_x, max_x, min_y, max_y

    min_x, max_x, min_y, max_y = _min_x, _max_x, _min_y, _max_y
    return


def get_normalized_data(x, min_val, max_val):
    """
    Normalizing the training and test dataset
    """
    x_norm = (x - min_val) / (max_val - min_val)

    return x_norm


def get_raw_data(x_norm, min_val, max_val):
    """
    Get raw data from the normalized dataset
    """
    x = x_norm * (max_val - min_val) + min_val

    return x

def read_data(data_folder, savedir, target_variable_names, data_variable_names, 
                  sed_samp_point_file, measurement_point_file):

        """Load dataset from file list (format is netCDF4), and connect
           multiple files.
           
           Parameters
           ------------------
           data_folder : string
               Name a folder in which all data files are preserved.

           targat_variable_names : list, string
               Name of variables that are target of inversion
            
           data_variable_names : list, string
                Name of variables that are inputs of inversion
           
           cood_file : string
                Filepath of a csv file with coordintes to be extracted
        """

        original_dataset = np.empty(0)
        target_dataset = np.empty(0)

        filelist = glob.glob(os.path.join(data_folder, '*.nc'))
        for f in filelist:
            dfile = netCDF4.Dataset(os.path.join(data_folder, f), 'r')
            num_runs = dfile.dimensions['run_no'].size

            # check Nan
            for j in range(num_runs):
                check_nan = []
                for k in data_variable_names:
                    check_data = (np.max(np.isnan(
                        dfile[k][j])) == False) 
                    check_nan.append(check_data)

                # if there is no Nan, create ndarray of target and data variable
                if all(check_nan):
                    try:
                        # create ndarray of data variable
                        # read sampling point of sediment and measurement point of flow condition
                        sed_samp_point = pd.read_csv(sed_samp_point_file, header=0)
                        measurement_point = pd.read_csv(measurement_point_file, header=0)
                        sed_x = sed_samp_point["0-dim"].to_numpy()
                        sed_y = sed_samp_point["1-dim"].to_numpy()
                        flow_x = measurement_point["0-dim_vel"].to_numpy()
                        flow_y = measurement_point["1-dim_vel"].to_numpy()
                        conc_x = measurement_point["0-dim_conc"].to_numpy()
                        conc_y = measurement_point["1-dim_conc"].to_numpy()
                        original_data_row = np.empty(0)
                        for data_variable in data_variable_names:
                            if "sed_volume_per_unit_area" in data_variable:
                                for i in range(len(sed_x)):
                                        original_data = dfile[data_variable][j, round(sed_x[i]), round(sed_y[i])]
                                        original_data_row = np.append(original_data_row, original_data)
                            else:
                                for l in range(len(flow_x)):
                                        if "layer_ave_vel" in data_variable:
                                            original_data = -dfile[data_variable][j, round(flow_x[l]), round(flow_y[l])]
                                            original_data_row = np.append(original_data_row, original_data)
                                        elif "layer_ave_conc" in data_variable:
                                            original_data = dfile[data_variable][j, round(conc_x[l]), round(conc_y[l])]
                                            original_data_row = np.append(original_data_row, original_data)
                                        else:    
                                            original_data = dfile[data_variable][j, round(flow_x[l]), round(flow_y[l])]
                                            original_data_row = np.append(original_data_row, original_data)
                            original_data_row = original_data_row[np.newaxis, :]
                        if all(original_data_row[0:len(sed_x)]==0):
                            raise ExcludeValue
                        if len(original_dataset)<1:
                            original_dataset = original_data_row
                        else:
                            original_dataset = np.concatenate([original_dataset, original_data_row], axis=0)

                        # create ndarray of target variable
                        target_arr = np.empty(0)
                        for target_name in target_variable_names:
                            target = np.array(dfile[target_name][j])
                            # if len(target_arr) < 1:
                            #     target_arr = target
                            # elif len(target_arr) >= 1:
                            target_arr = np.append(target_arr, target)
                        target_arr = target_arr[np.newaxis,:]
                        if len(target_dataset)<1:
                            target_dataset =target_arr
                        elif len(target_dataset)>=1:
                            target_dataset = np.concatenate([target_dataset,target_arr],axis=0)
                    except ExcludeValue as e:
                        with open(os.path.join(savedir, "zero_data.csv"), "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow('{}\n'.format(j))

                else:
                    with open(os.path.join(savedir, "remove_data.csv"), "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(j)

        return original_dataset, target_dataset

def preprocess(original_dataset, target_dataset, num_test, num_train=None, savedir=""):

    """This is the method that separates training dataset from test dataset.
    """
    num_data = original_dataset.shape[0]
    if num_train is None:
        num_train = num_data - num_test
    elif num_train > num_data - num_test:
        num_train = num_data - num_test
    elif num_test > num_data:
        raise ValueError('num_test is larger than num_data.')

    print('number of data is {}'.format(num_data))
    print('number of training data is {}'.format(num_train))
    print('number of test data is {}'.format(num_test))

    x_train_raw = original_dataset[0:(num_data - num_test)]
    x_test_raw = original_dataset[(num_data - num_test):]
    y_train_raw = target_dataset[0:(num_data - num_test)]
    y_test_raw = target_dataset[(num_data - num_test):]
    
    with open(os.path.join(savedir, "num_data.csv"), "w", newline="") as f:
      writer = csv.writer(f)
      header = ['num_data', 'num_train', 'num_test']
      writer.writerow(header)

    norm_x = np.array([-0.01, 0.01])
    norm_y = np.array(
        [np.min(y_train_raw, axis=0),
            np.max(y_train_raw, axis=0)])
    x_train = np.zeros_like(x_train_raw)
    x_test = np.zeros_like(x_test_raw)

    y_train = np.zeros(y_train_raw.shape)
    y_test = np.zeros(y_test_raw.shape)
    x_train[:, :] = x_train_raw
    x_test[:, :] = x_test_raw

    # normalization
    y_train = (y_train_raw - norm_y[0]) / (norm_y[1] -
                                                norm_y[0])
    y_test = (y_test_raw - norm_y[0]) / (norm_y[1] -
                                                norm_y[0])

    return x_train, y_train, x_test, y_test, norm_y

def reproduce_y(y, norm_y):
        """reproduce y value that was preprocessed
        """

        # y_reproduced = (y + 1.0) / 2 * (
        #     self.norm_y[1] - self.norm_y[0]) + self.norm_y[0]
        y_reproduced = (y) * (norm_y[1] - norm_y[0]) + norm_y[0]

        return y_reproduced

if __name__ == "__main__":
    pdb.set_trace()
    data_folder = '/mnt/c/Users/Seiya/Desktop/test_flowparam/3eq_2'
    resdir = '/mnt/c/Users/Seiya/Desktop/opt_test'
    cood_file = '/mnt/c/Users/Seiya/Desktop/test_flowparam/fcn_test/sed_vol.csv'
    measuremnt_point_file = '/mnt/c/Users/Seiya/Desktop/test_flowparam/fcn_test/mea_point.csv'
    shutil.copy("nninv_1d.py", resdir)
    target_variable_names = ["Cf", "alpha_4eq", "r0", "C_ini", "U_ini", "endtime"]
    data_variable_names = ["sed_volume_per_unit_area_0", "sed_volume_per_unit_area_1", "sed_volume_per_unit_area_2", "sed_volume_per_unit_area_3",
                           "layer_ave_vel", "layer_ave_conc_0", "layer_ave_conc_1", "layer_ave_conc_2", "layer_ave_conc_3", "flow_depth"]
    original_dataset, target_dataset = read_data(data_folder, resdir, target_variable_names, data_variable_names, cood_file, measuremnt_point_file)
    x_train, y_train, x_test, y_test, norm_y = preprocess(original_dataset, target_dataset, 2, num_train=None, savedir=resdir)

    model, history = deep_learning_turbidite(resdir,
                                                 x_train,
                                                 y_train,
                                                 x_test,
                                                 y_test,
                                                 epochs=10,
                                                 num_layers=6)
    save_history(history, resdir)
    save_result(resdir, model=model)
    plot_history(history, resdir)
    test_result_norm = model.predict(x_test)
    test_result = reproduce_y(test_result_norm, norm_y)
    test_original = reproduce_y(y_test, norm_y)
    min_val = np.min(np.min([test_original, test_result], axis=0), axis=0)
    min_val[min_val < 0] = 0
    max_val = np.max(np.max([test_original, test_result], axis=0), axis=0)
    val_name = ['Cf', 'alpha', '$r_{0}$', '$C_{1}$', '$C_{2}$','$C_{3}$','$C_{4}$', 'salinity', 'Initial Flow Velocity', 'Flow Duration']
    units = ['', '', '', '', '', '', '', '' ,'(m/s)', '(s)']
    fontname = 'Segoe UI'

    for i in range(test_result.shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(3.93, 3.93),tight_layout=True)
        ax.plot(test_original[:, i], test_result[:, i], 'o')
        ax.plot([min_val[i], max_val[i] * 1.1], [min_val[i], max_val[i] * 1.1],
                marker=None,
                linewidth=2)
        ax.set_title(val_name[i], fontsize=18, fontname=fontname)
        ax.set_xlabel('Original Value' + units[i],
                        fontsize=14,
                        fontname=fontname)
        ax.set_ylabel('Reconstructed Value' + units[i],
                        fontsize=14,
                        fontname=fontname)
        ax.tick_params(labelsize=14)
        ax.set_aspect('equal')
        fig.patch.set_alpha(0)
        plt.savefig(os.path.join(resdir, 'test_result{}.svg'.format(i)))
        ax.cla()

    np.savetxt(os.path.join(resdir, 'test_result.csv'), test_result, delimiter=',')
    np.savetxt(os.path.join(resdir,'test_original.csv'), test_original, delimiter=',')
    # Load data
    # datadir_training_num = './distance/10/data'
    # resdir_training_num = './result_training_num_10'
    # if not os.path.exists(resdir_training_num):
    #     os.mkdir(resdir_training_num)

    # x_train, y_train, x_test, y_test = load_data(datadir_training_num)

    # # Start training
    # # testcases_train_num = [500, 1000, 1500, 2000, 2500, 3000, 3500]
    # testcases_train_num = []
    # for i in range(len(testcases_train_num)):
    #     resdir_case = os.path.join(resdir_training_num,
    #                                '{}/'.format(testcases_train_num[i]))
    #     if not os.path.exists(resdir_case):
    #         os.mkdir(resdir_case)
    #     x_train_sub = x_train[0:testcases_train_num[i], :]
    #     y_train_sub = y_train[0:testcases_train_num[i], :]
    #     model, history = deep_learning_turbidite(resdir_case,
    #                                              x_train_sub,
    #                                              y_train_sub,
    #                                              x_test,
    #                                              y_test,
    #                                              epochs=20000,
    #                                              num_layers=6)
    #     plot_history(history, resdir)
    #     save_history(history, resdir)
    #     # Verification and test
    #     # model = load_model(os.path.join(resdir_case, 'model.hdf5'))
    #     result = test_model(model, x_test)
    #     save_result(resdir_case,
    #                 model=model,
    #                 history=history,
    #                 test_result=result)
    #     # save_result(resdir_case, test_result=result)

    # # #load data 
    # # datadir_distance = './distance'
    # # resdir_distance = './result_distance_3500_2'

    # # #学習の実行
    # # num_data = 3500
    # # # testcases_distance = [1, 2, 3, 4, 5, 10]
    # # testcases_distance = [15, 20, 25, 30]
    # # for i in range(len(testcases_distance)):
    # #     x_train, y_train, x_test, y_test = load_data(
    # #         os.path.join(datadir_distance, '{}'.format(testcases_distance[i]),
    # #                      'data'))
    # #     resdir_case = os.path.join(resdir_distance,
    # #                                '{}/'.format(testcases_distance[i]))
    # #     if not os.path.exists(resdir_case):
    # #         os.mkdir(resdir_case)
    # #     x_train_sub = x_train[0:num_data, :]
    # #     y_train_sub = y_train[0:num_data, :]
    # #     model, history = deep_learning_turbidite(resdir_case,
    # #                                              x_train_sub,
    # #                                              y_train_sub,
    # #                                              x_test,
    # #                                              y_test,
    # #                                              epochs=20000,
    # #                                              num_layers=6)

    # #     
    # #     # model = load_model(os.path.join(resdir_case, 'model.hdf5'))
    # #     result = test_model(model, x_test)
    # #     save_result(resdir_case, model=model, history=history, test_result=result)
    # #     # save_result(resdir_case, test_result=result)
