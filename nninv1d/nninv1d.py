'-*- coding: utf-8 -*-'
"""
Created on Tue Mar  7 15:43:18 2017

@author: hanar
"""

import time
import numpy as np
import os
# from keras.utils import np_utils
gpu_id = 0
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
# print(tf.__version__)
# if tf.__version__ >= "2.1.0":
#     physical_devices = tf.config.list_physical_devices('GPU')
#     tf.config.list_physical_devices('GPU')
#     tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input
from tensorflow.keras.optimizers import SGD, Adagrad, Adam, Nadam, Adadelta, Adamax, RMSprop, Ftrl
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow.python.keras.backend as K
from tensorflow.python import debug as tf_debug
# from tensorflow.distribute import MirroredStrategy
#from keras.utils.visualize_util import plot
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import glob
import pandas as pd
import pdb
import csv
import shutil

class Exclude_value(Exception):
    pass

def deep_learning_turbidite(resdir,
                            X_train,
                            y_train,
                            lr=0.02,
                            decay=None,
                            validation_split=0.2,
                            batch_size=2,
                            activation_func='relu',
                            activation_output='relu',
                            initializer='he_uniform',
                            loss_func="mean_squared_error",
                            optimizer='Adagrad',
                            momentum=0.9,
                            nesterov=True,
                            metrics="mean_squared_error",
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
    # input shape 
    model.add(Input(shape=(X_train.shape[1],)))
    # input layer
    model.add(
        Dense(node_num,
              activation=activation_func,
              kernel_initializer=initializer))
    model.add(Dropout(dropout))
    # hidden layer
    for i in range(num_layers - 2):
        model.add(
            Dense(node_num,
                  activation=activation_func,
                  kernel_initializer=initializer))
        model.add(Dropout(dropout))
    # output layer
    model.add(
        Dense(y_train.shape[1],
              activation=activation_output,
              kernel_initializer=initializer))
    
    if optimizer == 'Adagrad':
        optimizer = Adagrad(learning_rate=lr)
    elif optimizer == 'Adadelta':
        optimizer = Adadelta(learning_rate=lr)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=lr,  momentum=momentum, nesterov=nesterov)
    elif optimizer == 'Adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer == "RMSprop":
        optimizer = RMSprop(learinig_rate=lr)
    elif optimizer == 'Adamax':
        optimizer = Adamax(learning_rate=lr)
    elif optimizer == 'Nadam':
        optimizer = Nadam(learning_rate=lr)
    elif optimizer == 'Ftrl':
        optimizer = Ftrl(learning_rate=lr)
    else:
        raise ValueError('Select the appropriate optimizer')
    
    # Compilation of the model
    model.compile(
        loss=loss_func,
        # optimizer=SGD(learning_rate=lr,  momentum=momentum,
        #              nesterov=nesterov),
        optimizer=optimizer,
        metrics=[metrics])

    # Start training
#     t = time.time()
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(resdir, "model.keras"),
                            monitor='val_loss',
                            verbose=0,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='min',
                            save_freq='epoch',
                            initial_value_threshold=None
                            )
#     #es_cb = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
#     tb_cb = TensorBoard(log_dir=os.path.join(resdir, 'logs'),
#                         histogram_freq=0,
#                         write_graph=False,
#                         write_images=False)
# #     history = model.fit(X_train,
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
    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split,
                        callbacks=[model_checkpoint, TensorBoard(log_dir=log_dir)],
                        shuffle=True,
                        verbose=1)
    
    return model, history
 
# def apply_model(model, X, min_x, max_x, min_y, max_y):
#     """
#     Apply the model to data sets
#     """
#     X_norm = (X - min_x) / (max_x - min_x)
#     Y_norm = model.predict(X_norm)
#     Y = Y_norm * (max_y - min_y) + min_y
#     return Y

def plot_history(history, savedir):
    # plot training history                                                                                                                                  
    plt.plot(history.history['mean_squared_error'], "o-", label="Training")
    plt.plot(history.history['val_mean_squared_error'], "o-", label="Validation")
    plt.title('Training History', fontsize=18)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=14)
    plt.legend(loc="upper right", fontsize=14)
    plt.savefig(os.path.join(savedir, "history.svg"))
    plt.close()

def save_history(history, dirpath):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(dirpath,'history.csv'))

# def test_model(model, x_test):
#     #test the model

#     x_test_norm = get_normalized_data(x_test, min_x, max_x)
#     test_result_norm = model.predict(x_test_norm)
#     test_result = get_raw_data(test_result_norm, min_y, max_y)

    # return test_result


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
        model.save(os.path.join(savedir,'model.keras'))


# def load_data(datadir):
#     """
#     This function load training and test data sets, and returns variables
#     """
#     global min_x, max_x, min_y, max_y

#     x_train = np.load(os.path.join(datadir, 'H_train.npy'))
#     x_test = np.load(os.path.join(datadir, 'H_test.npy'))
#     y_train = np.load(os.path.join(datadir, 'icond_train.npy'))
#     y_test = np.load(os.path.join(datadir, 'icond_test.npy'))
#     min_y = np.load(os.path.join(datadir, 'icond_min.npy'))
#     max_y = np.load(os.path.join(datadir, 'icond_max.npy'))
#     [min_x, max_x] = np.load(os.path.join(datadir, 'x_minmax.npy'))

#     return x_train, y_train, x_test, y_test


# def set_minmax_data(_min_x, _max_x, _min_y, _max_y):
#     global min_x, max_x, min_y, max_y

#     min_x, max_x, min_y, max_y = _min_x, _max_x, _min_y, _max_y
#     return


# def get_normalized_data(x, min_val, max_val):
#     """
#     Normalizing the training and test dataset
#     """
#     x_norm = (x - min_val) / (max_val - min_val)

#     return x_norm


# def get_raw_data(x_norm, min_val, max_val):
#     """
#     Get raw data from the normalized dataset
#     """
#     x = x_norm * (max_val - min_val) + min_val

#     return x

def read_data(data_folder, result_dir, target_variable_names, data_variable_names, 
                  sed_samp_point_file):

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
        filepath = os.path.join(data_folder, '**')
        filelist = glob.glob(os.path.join(filepath, '*.nc'), recursive=True)
        for f in filelist:
            dfile = netCDF4.Dataset(os.path.join(data_folder, f), 'r')
            num_runs = dfile.dimensions['run_no'].size

            # check Nan
            for j in range(num_runs):
                check_nan = []
                for k in data_variable_names:
                    check_data = (
                                 (np.max(np.isnan(dfile[k][j])) == False)
                                  and (np.max(dfile[k][j]) < 0.5)
                                  and (np.min(dfile[k][j] > -0.003))
                                 )
                    check_nan.append(check_data)

                # if there is no Nan, create ndarray of target and data variable
                if all(check_nan):
                    try:
                    # create ndarray of data variable
                    # read sampling point of sediment and measurement point of flow condition
                        sed_samp_point = pd.read_csv(sed_samp_point_file, header=0)
                        sed_x = sed_samp_point["0-dim"].to_numpy()
                        sed_y = sed_samp_point["1-dim"].to_numpy()
                        original_data_row = np.empty(0)
                        for data_variable in data_variable_names:
                            if "sed_volume_per_unit_area" in data_variable:
                                for i in range(len(sed_x)):
                                    original_data = dfile[data_variable][j, sed_x[i], sed_y[i]]
                                    original_data_row = np.append(original_data_row, original_data)

                        if all(original_data_row<0.0001):
                            raise Exclude_value
                        original_data_row = original_data_row[np.newaxis, :]
                        if len(original_dataset)<1:
                            original_dataset = original_data_row
                        else:
                            original_dataset = np.concatenate([original_dataset, original_data_row], axis=0)

                        # create ndarray of target variable                                                                                                                                               
                        target_arr = np.empty(0)
                        for target_name in target_variable_names:
                            target = np.array(dfile[target_name][j])                      
                            target_arr = np.append(target_arr, target)
                        target_arr = target_arr[np.newaxis,:]
                        if len(target_dataset)<1:
                            target_dataset =target_arr
                        elif len(target_dataset)>=1:
                            target_dataset = np.concatenate([target_dataset,target_arr],axis=0)
                    except Exclude_value as e:
                        with open(os.path.join(result_dir,'remove_num.txt'), mode='a') as f:
                            f.write('{}\n'.format(j))
                        #print('Run no.{} was removed'.format(j))
        
        return original_dataset, target_dataset

def preprocess(original_dataset, target_dataset, num_test=100, num_train=None, savedir=""):

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
    x_train_raw = original_dataset[0:(num_train), :]
    x_test_raw = original_dataset[num_train:(num_train+num_test):, :]
    y_train_raw = target_dataset[0:(num_train), :]
    y_test_raw = target_dataset[(num_train):(num_train+num_test), :]
    np.save(os.path.join(savedir,"y_train_raw"), y_train_raw)
    np.save(os.path.join(savedir,"y_test_raw"), y_test_raw)
    
    with open(os.path.join(savedir, "num_data.csv"), "w", newline="") as f:
      writer = csv.writer(f)
      header = ['num_data', 'num_train', 'num_test']
      writer.writerow(header)
      writer.writerow([num_data, num_train, num_test])

    norm_y = np.array(
        [np.min(y_train_raw, axis=0),
            np.max(y_train_raw, axis=0)])
    np.save(os.path.join(savedir, 'norm_y'), norm_y)

    norm_x = np.array([np.amin(x_train_raw), np.amax(x_train_raw)])
    np.save(os.path.join(savedir, 'norm_x'), norm_x)

    x_train = np.zeros_like(x_train_raw)
    x_test = np.zeros_like(x_test_raw)

    y_train = np.zeros(y_train_raw.shape)
    y_test = np.zeros(y_test_raw.shape)
    x_train[:, :] = x_train_raw
    x_test[:, :] = x_test_raw
    # x_train[:, :] = (x_train_raw - norm_x[0]) / (
    #       norm_x[1] - norm_x[0])
    # x_test[:, :] = (x_test_raw - norm_x[0]) / (
    #        norm_x[1] - norm_x[0])
    x_train.dump(os.path.join(savedir, "x_train"))
    x_test.dump(os.path.join(savedir, "x_test"))
    # normalization
    # max value is transformed to 1, min value is transformed to 0
    y_train = (y_train_raw - norm_y[0]) / (norm_y[1] -
                                                norm_y[0])
    y_test = (y_test_raw - norm_y[0]) / (norm_y[1] -
                                                norm_y[0])
    
    np.save(os.path.join(savedir, "y_train"), y_train)
    np.save(os.path.join(savedir, "y_test"), y_test)
    # np.save(os.path.join(savedir, "x_train"), x_train)
    # np.save(os.path.join(savedir, "x_test"), x_test)

    return x_train, y_train, x_test, y_test, norm_y

def reproduce_y(y, norm_y):
        """reproduce y value that was preprocessed
        """
        y_reproduced = (y) * (norm_y[1] - norm_y[0]) + norm_y[0]

        return y_reproduced

def save_test_results(test_original, test_predicted, val_name, savedir):
    test_original = pd.DataFrame(test_original, columns=val_name)
    test_original.to_csv(os.path.join(savedir, 'test_original.csv'))
    test_predicted = pd.DataFrame(test_predicted, columns=val_name)
    test_predicted.to_csv(os.path.join(savedir, 'test_result.csv'))

def plot_test_results(model, X_test, y_test, norm_y, val_name, units, savedir):
    test_pred_norm = model.predict(X_test)
    test_pred = reproduce_y(test_pred_norm, norm_y)
    test_original = reproduce_y(y_test, norm_y)
    save_test_results(test_original, test_pred, val_name, savedir)
    plt.plot(test_original, test_pred, "o")
    min_val = np.min(np.min([test_pred, test_original], axis=0), axis=0)
    min_val[min_val < 0] = 0
    max_val = np.max(np.max([test_pred, test_original], axis=0), axis=0)

    for i in range(test_pred.shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(3.93, 3.93), tight_layout=True)
        ax.plot(test_original[:, i], test_pred[:, i], 'o')
        ax.plot([min_val[i], max_val[i] * 1.1], [min_val[i], max_val[i] * 1.1],
                marker=None,
                linewidth=2)
        ax.set_title(val_name[i], fontsize=18)
        ax.set_xlabel('Original Value' + units[i],
                        fontsize=14)
        ax.set_ylabel('Reconstructed Value' + units[i],
                        fontsize=14)
        ax.tick_params(labelsize=14)
        ax.set_aspect('equal')
        fig.patch.set_alpha(0)
        plt.savefig(os.path.join(savedir, 'test_result{}.svg'.format(i)))
        ax.cla()
    plt.close()

if __name__ == "__main__":
    # pdb.set_trace()
    data_folder = '/home/biosphere/sed_share2/fujishima/phd_research/data/exp2/run2/no_detrainment/no_erosion/after_cfrev/cf0.002_ro2_c0.0001-0.01/data'
    resdir = '/home/biosphere/sed_share2/fujishima/phd_research/data/exp2/run2/no_detrainment/no_erosion/after_cfrev/cf0.002_ro2_c0.0001-0.01/inv_result/remove_abn_12000'
    if not os.path.exists(resdir):
        os.mkdir(resdir)
    cood_file = '/mnt/sed_share2/fujishima/3d_model/exp2/diff_bed_run_1_2/sed_volume_run2/sed_dx0.05.csv'
    # cood_file = '/home/biosphere/sed_share2/fujishima/phd_research/data/exp1/detrainment/cf0.002_detrate1.0/samp_point2.csv'
    # cood_file = '/mnt/sed_share2/fujishima/phd_research/data/exp3/exp3_cf0.002_ro2_noerosion/after_revcf/inv_result/nonormx_8000/samp_idx_exp3.csv'
    src = "nninv1d.py"
    dst = os.path.join(resdir, "nninv1d.py")
    shutil.copyfile(src, dst)
    # target_variable_names = ["Cf", "alpha_4eq", "r0", "C_ini",  "U_ini", "h_ini", "endtime"]
    target_variable_names = ["C_ini",  "U_ini", "h_ini", "endtime"]
    # data_variable_names = ["sed_volume_per_unit_area_0", "sed_volume_per_unit_area_1", "sed_volume_per_unit_area_2", "sed_volume_per_unit_area_3",
    #                       "layer_ave_vel", "layer_ave_conc_0", "layer_ave_conc_1", "layer_ave_conc_2", "layer_ave_conc_3", "flow_depth"]
    data_variable_names = ["sed_volume_per_unit_area_0", "sed_volume_per_unit_area_1", "sed_volume_per_unit_area_2", "sed_volume_per_unit_area_3"]
    original_dataset, target_dataset = read_data(data_folder, resdir, target_variable_names, data_variable_names, cood_file)
    # original_dataset = np.load("/home/biosphere/sed_share2/fujishima/phd_research/data/exp1/detrainment/cf0.002_detrate1.0/data/c0.00001_0.0001/extract_data_rot90/original_dataset.npy")
    # target_dataset = np.load("/home/biosphere/sed_share2/fujishima/phd_research/data/exp1/detrainment/cf0.002_detrate1.0/data/c0.00001_0.0001/extract_data_rot90/target_dataset.npy")
    x_train, y_train, x_test, y_test, norm_y = preprocess(original_dataset, target_dataset, num_test=100, num_train=None, savedir=resdir)
    y_train = np.load(os.path.join(resdir,'y_train.npy'))
    y_test = np.load(os.path.join(resdir,'y_test.npy'))
    norm_y = np.load(os.path.join(resdir,'norm_y.npy'))
    model, history = deep_learning_turbidite(resdir,
                                             X_train=x_train,
                                             y_train=y_train,
                                             lr=0.001,
                                             decay=None,
                                             validation_split=0.2,
                                             batch_size=2,
                                            activation_func='relu',
                                            activation_output='relu',
                                            initializer='he_uniform',
                                            loss_func="mean_squared_error",
                                            optimizer='SGD',
                                            momentum=0.9,
                                            nesterov=True,
                                            metrics="mean_squared_error",
                                            num_layers=4,
                                            dropout=0.1,
                                            node_num=1000,
                                            epochs=1
                                            )

    save_history(history, resdir)
    save_result(resdir, model=model)
    plot_history(history, resdir)
    test_result_norm = model.predict(x_test)
    test_result = reproduce_y(test_result_norm, norm_y)
    test_original = reproduce_y(y_test, norm_y)
    np.savetxt(os.path.join(resdir, 'test_result.csv'), test_result, delimiter=',')
    np.savetxt(os.path.join(resdir,'test_original.csv'), test_original, delimiter=',')
    min_val = np.min(np.min([test_original, test_result], axis=0), axis=0)
    min_val[min_val < 0] = 0
    max_val = np.max(np.max([test_original, test_result], axis=0), axis=0)
    val_name = ['$C_{1}$ at inlet', '$C_{2}$ at inlet','$C_{3}$ at inlet','$C_{4}$ at inlet', '$S_{0}$ at inlet', 'Flow Velocity at inlet', 'Flow Depth at inlet', 'Flow Duration']
    # val_name = ['$C_{1}$', '$C_{2}$','$C_{3}$','$C_{4}$', 'Initial Flow Velocity', 'Initial Flow Height', 'Flow Duration']
    #val_name = ['Cf', '$r_{0}$', '$C_{1}$', '$C_{2}$','$C_{3}$','$C_{4}$', 'Salinity', 'Initial Flow Velocity', 'Flow Duration']
    # units = ['', '', '', '' , '', ' (m/s)', ' (m)', ' (s)']
    units = ['', '', '', '' , '', ' (m/s)', ' (m)', ' (s)']

    for i in range(test_result.shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(3.93, 3.93),tight_layout=True)
        ax.plot(test_original[:, i], test_result[:, i], 'o')
        ax.plot([min_val[i], max_val[i] * 1.1], [min_val[i], max_val[i] * 1.1],
                marker=None,
                linewidth=2)
        ax.set_title(val_name[i], fontsize=18)
        ax.set_xlabel('Original Value' + units[i],
                        fontsize=14)
        ax.set_ylabel('Reconstructed Value' + units[i],
                        fontsize=14)
        ax.tick_params(labelsize=14)
        ax.set_aspect('equal')
        fig.patch.set_alpha(0)
        plt.savefig(os.path.join(resdir, 'test_result{}.svg'.format(i)))
        ax.cla()
    plt.close()
    check_result_norm = model.predict(x_train)
    check_result = reproduce_y(check_result_norm, norm_y)
    check_original = reproduce_y(y_train, norm_y)
    np.savetxt(os.path.join(resdir, 'check_result.csv'), check_result, delimiter=',')
    np.savetxt(os.path.join(resdir,'check_original.csv'), check_original, delimiter=',')
    min_val = np.min(np.min([check_original, check_result], axis=0), axis=0)
    min_val[min_val < 0] = 0
    max_val = np.max(np.max([check_original, check_result], axis=0), axis=0)
    # val_name = ['$C_{1}$', '$C_{2}$','$C_{3}$','$C_{4}$', 'Flow Velocity', 'Flow Height', 'Flow Duration']
    val_name = ['$C_{1}$', '$C_{2}$','$C_{3}$','$C_{4}$', '$S_{0}$', 'Initial Flow Velocity', 'Initial Flow Height', 'Flow Duration']
    # val_name = ['Cf', '$r_{0}$', '$C_{1}$', '$C_{2}$','$C_{3}$','$C_{4}$', 'Salinity', 'Initial Flow Velocity', 'Flow Duration']
    units = ['', '', '' , '', '', ' (m/s)', ' (m)', ' (s)']
    fontname = 'Segoe UI'

    for i in range(check_result.shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(3.93, 3.93),tight_layout=True)
        ax.plot(check_original[:, i], check_result[:, i], 'o')
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
        plt.savefig(os.path.join(resdir, 'check_result{}.svg'.format(i)))
        ax.cla()

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
