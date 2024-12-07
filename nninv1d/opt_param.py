import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adagrad, Adam, Adadelta, Nadam, RMSprop, Adamax
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
from nninv1d import read_data, save_history, save_result, plot_history, reproduce_y, preprocess
import os
import numpy as np
import pdb
# pdb.set_trace() 

data_folder = '/home/biosphere/sed_share2/fujishima/phd_research/data/exp2/run1/4eq/fix_r0_cf_alpha/gp1991'
savedir_path = '/home/biosphere/sed_share2/fujishima/phd_research/data/exp2/run1/4eq/fix_r0_cf_alpha/gp1991/inv_result_test_optuna'
if not os.path.exists(savedir_path):
    os.mkdir(savedir_path)
# cood_file = '/home/biosphere/sed_share2/fujishima/param_optim/cood_data/exp2/cood_file.csv'
cood_file = '/home/biosphere/sed_share2/fujishima/3d_model/exp2/run1/sed3.csv'
measuremnt_point_file = '/home/biosphere/sed_share2/fujishima/param_optim/cood_data/exp2/run1/mea_point.csv'
shutil.copy("nninv1d.py", savedir_path)
# target_variable_names = ["Cf", "alpha_4eq", "r0", "C_ini",  "U_ini", "h_ini", "endtime"]
target_variable_names = ["C_ini",  "U_ini", "h_ini", "endtime"]
# data_variable_names = ["sed_volume_per_unit_area_0", "sed_volume_per_unit_area_1", "sed_volume_per_unit_area_2", "sed_volume_per_unit_area_3",
#                       "layer_ave_vel", "layer_ave_conc_0", "layer_ave_conc_1", "layer_ave_conc_2", "layer_ave_conc_3", "flow_depth"]
data_variable_names = ["sed_volume_per_unit_area_0", "sed_volume_per_unit_area_1", "sed_volume_per_unit_area_2", "sed_volume_per_unit_area_3"]
original_dataset, target_dataset = read_data(data_folder, savedir_path, target_variable_names, data_variable_names, cood_file, measuremnt_point_file)
x_train, y_train, x_test, y_test, norm_y = preprocess(original_dataset, target_dataset, 100, num_train=3000, savedir=savedir_path)

def create_model(trial, x_train):
    # set target of optimization
    n_hiddenlayers = trial.suggest_int('n_layers', 1, 5)
    node_num = trial.suggest_int('node_num', 1000, 5000)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

    # create model
    model = Sequential()
    # input layer
    model.add(
        Dense(node_num,
            input_dim=x_train.shape[1],
            activation='relu',
            kernel_initializer='glorot_uniform'))
    model.add(Dropout(dropout_rate))
    # hidden layer
    for i in range(n_hiddenlayers):
        model.add(
            Dense(node_num,
                activation='relu',
                kernel_initializer='glorot_uniform'))
        model.add(Dropout(dropout_rate))
    # output layer
    model.add(
        Dense(y_train.shape[1],
            activation='relu',
            kernel_initializer='glorot_uniform'))
    
    return model

# def create_optimizer(trial):
#     opt = trial.suggest_categorical('optimizer', ["SGD", "Adagrad", "Adam", "Adadelta", "RMSprop", "Nadam", "Adamax"])
#     learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)

#     if opt=="SGD":
#         optimizer = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
#     elif optimizer=="Adagrad":
#         opt = Adagrad(learning_rate=learning_rate)
#     elif optimizer == "Adam":
#         optimizer = Adam(learning_rate=learning_rate)
#     elif opt == "Adadelta":
#         optimizer = Adadelta(learning_rate=learning_rate)
#     elif opt == "RMSprop":
#         optimizer = RMSprop(learning_rate=learning_rate)
#     elif opt == "Nadam":
#         optimizer = Nadam(learning_rate=learning_rate)
#     elif opt == "Adamax":
#         optimizer = Adamax(learning_rate=learning_rate)

#     return optimizer

def trainer(trial, x_train, y_train, x_test, y_test, epochs=10000, validation_split=0.2, dirpath=""):


    # create model
    model = create_model(trial, x_train)
    # optimizer = create_optimizer(trial)
    # pdb.set_trace()
    opt = trial.suggest_categorical('optimizer', ["SGD", "Adagrad", "Adam", "Adadelta", "RMSprop", "Nadam", "Adamax"])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)

    if opt=="SGD":
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    elif opt=="Adagrad":
        optimizer = Adagrad(learning_rate=learning_rate)
    elif opt == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif opt == "Adadelta":
        optimizer = Adadelta(learning_rate=learning_rate)
    elif opt == "RMSprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    elif opt == "Nadam":
        optimizer = Nadam(learning_rate=learning_rate)
    elif opt == "Adamax":
        optimizer = Adamax(learning_rate=learning_rate)

    # set target of optimization
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])

    # pdb.set_trace()
    dirname = opt+"_"+str(learning_rate)+"_"+str(batch_size)
    savedir = os.path.join(dirpath, dirname)
    os.mkdir(savedir)
    
    model.compile(
    loss="mean_squared_error",
    optimizer=optimizer,
    metrics=["mean_squared_error"])

    # create logdir for tensorboard
    log_dir = os.path.join(savedir, 'logdir')
    # if there is logdir, it is removed
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)

    # training
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split,
                        callbacks=[TensorBoard(log_dir=log_dir)],
                        shuffle=True,
                        verbose=1)

    # plot result of training
    save_history(history, savedir) 
    save_result(savedir, model=model)
    plot_history(history, savedir)

    # test model
    test_result_norm = model.predict(x_test)
    test_result = reproduce_y(test_result_norm, norm_y)
    test_original = reproduce_y(y_test, norm_y)
    np.savetxt(os.path.join(savedir, 'test_result.csv'), test_result, delimiter=',')
    np.savetxt(os.path.join(savedir,'test_original.csv'), test_original, delimiter=',')

    # plot result of test
    min_val = np.min(np.min([test_original, test_result], axis=0), axis=0)
    min_val[min_val < 0] = 0
    max_val = np.max(np.max([test_original, test_result], axis=0), axis=0)
    val_name = ['$C_{1}$', '$C_{2}$','$C_{3}$','$C_{4}$', 'Salinity', 'Flow Velocity', 'Flow Height', 'Flow Duration']
    # val_name = ['Cf', 'Alpha', '$r_{0}$', '$C_{1}$', '$C_{2}$','$C_{3}$','$C_{4}$', 'Salinity', 'Initial Flow Velocity', 'Initial Flow Height', 'Flow Duration']
    #val_name = ['Cf', '$r_{0}$', '$C_{1}$', '$C_{2}$','$C_{3}$','$C_{4}$', 'Salinity', 'Initial Flow Velocity', 'Flow Duration']
    units = ['', '', '', '' ,'', '(m/s)', '(m)', '(s)']

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
        plt.savefig(os.path.join(savedir, 'test_result{}.svg'.format(i)))
        ax.cla()
    plt.close()

    # get minimum value of loss function for validation datasets
    loss_min = np.min(history.history["val_loss"])

    return loss_min

def objective(trial):
    loss = trainer(trial, x_train, y_train, x_test, y_test, epochs=100, validation_split=0.2, dirpath=savedir_path)

    return loss

# pdb.set_trace()
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
best_params = study.best_params
print(f"Best unit: {best_params}")

hist_df = study.trials_dataframe(multi_index=True)
hist_df.to_csv(os.path.join(savedir_path, "result.csv"))