import numpy  as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
import os
import shutil
import pandas as pd
from nninv1d import reproduce_y, read_data, preprocess, deep_learning_turbidite, plot_history, save_history, save_result, plot_test_results
from validation import Validation

if __name__ == "__main__":
    data_folder =  '/mnt/sed_share2/fujishima/inv_confined/tr_data'
    resdir = '/mnt/sed_share2/fujishima/inv_confined/inv_result/num_data/num2600'
    if not os.path.exists(resdir):
        os.mkdir(resdir)
    cood_file = '/mnt/sed_share2/fujishima/inv_confined/inv_result/samp_point_50.csv'
    src = "do_inv.py"
    dst = os.path.join(resdir, "do_inv.py")
    shutil.copyfile(src, dst)
    target_variable_names = ["C_ini",  "U_ini", "h_ini", "endtime"]
    data_variable_names = ["sed_volume_per_unit_area_0", 
                           "sed_volume_per_unit_area_1"]
    original_dataset, target_dataset = read_data(data_folder, resdir, target_variable_names, data_variable_names, cood_file, shuffle_data=False)
    x_train, y_train, x_test, y_test, norm_y = preprocess(original_dataset, target_dataset, num_test=100, num_train=2600, savedir=resdir)

    model, history = deep_learning_turbidite(resdir,
                                             x_train,
                                             y_train,
                                             lr=0.01,
                                             decay=None,
                                             validation_split=0.2,
                                             batch_size=64,
                                             activation_func='relu',
                                             activation_output='relu',
                                             initializer='he_uniform',
                                             loss_func="mean_squared_error",
                                             optimizer='Adagrad',
                                             momentum=0.9,
                                             nesterov=True,
                                             metrics="mean_squared_error",
                                             num_layers=6,
                                             dropout=0.5,
                                             node_num=4000,
                                             epochs=10000)

    save_history(history, resdir)
    plot_history(history, resdir)

    # Plot test results
    plot_test_results(
        model=model,
        X_test=x_test,
        y_test=y_test,
        norm_y=norm_y,
        val_name=['$C_{0, 1}$', 
                '$C_{0, 2}$',
                '$U_{0}$', 
                '$h_{0}$', 
                '$T_{\mathrm{d}}$'],
        units=['', 
                '',
            ' (m/s)', 
            ' (m)', 
            ' (s)'],
        savedir=resdir
    )

    # validation of inverse model using flume experiment
    # validation = Validation()
    # validation.predict(model = model, 
    #                    cood_file = cood_file, 
    #                    savedir = resdir, 
    #                    normx_file = None, 
    #                    normy_file = None, 
    #                    norm_y = norm_y,
    #                    data_variable_names=["bed__sediment_volume_per_unit_area_0", 
    #                                         "bed__sediment_volume_per_unit_area_1", 
    #                                         "bed__sediment_volume_per_unit_area_2", 
    #                                         "bed__sediment_volume_per_unit_area_3"], 
    #                    val_list=['C0,1', 'C0,2', 'C0,3', 'C0,4', 'Salinity', 'U0', 'h0', 'Fd']
    #                    )
