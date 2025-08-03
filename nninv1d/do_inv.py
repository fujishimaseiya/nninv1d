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
    data_folder = '/mnt/sed_share2/fujishima/phd_research/data/exp2/run2/no_detrainment/no_erosion/after_cfrev/cf0.002_ro2/data'
    resdir = '/mnt/sed_share2/fujishima/phd_research/data/exp2/run2/no_detrainment/no_erosion/after_cfrev/cf0.002_ro2/inv_result/test'
    if not os.path.exists(resdir):
        os.mkdir(resdir)
    cood_file = '/mnt/sed_share2/fujishima/3d_model/exp2/diff_bed_run_1_2/sed_volume_run2/sed_dx0.05.csv'
    src = "nninv1d.py"
    dst = os.path.join(resdir, "nninv1d.py")
    shutil.copyfile(src, dst)
    target_variable_names = ["C_ini",  "U_ini", "h_ini", "endtime"]
    data_variable_names = ["sed_volume_per_unit_area_0", "sed_volume_per_unit_area_1", "sed_volume_per_unit_area_2", "sed_volume_per_unit_area_3"]
    original_dataset, target_dataset = read_data(data_folder, resdir, target_variable_names, data_variable_names, cood_file)
    x_train, y_train, x_test, y_test, norm_y = preprocess(original_dataset, target_dataset, num_test=100, num_train=None, savedir=resdir)

    model, history = deep_learning_turbidite(resdir,
                                             x_train,
                                             y_train,
                                             epochs=1,
                                             batch_size=2,
                                             lr=0.001,
                                             dropout=0.1,
                                             node_num=4000,
                                             num_layers=6)

    save_history(history, resdir)
    save_result(resdir, model=model)
    plot_history(history, resdir)

    # Plot test results
    plot_test_results(
        model=model,
        X_test=x_test,
        y_test=y_test,
        norm_y=norm_y,
        val_name=['$C_{1}$ at inlet', 
                  '$C_{2}$ at inlet',
                  '$C_{3}$ at inlet',
                  '$C_{4}$ at inlet', 
                  '$S_{0}$ at inlet', 
                  'Flow Velocity at inlet', 
                  'Flow Depth at inlet', 
                  'Flow Duration'],
        units=['', 
               '', 
               '', 
               '' , 
               '', 
               ' (m/s)', 
               ' (m)', 
               ' (s)'],
        savedir=resdir
    )

    # validation of inverse model using flume experiment
    validation = Validation()
    validation.predict(model = model, 
                       cood_file = cood_file, 
                       savedir = resdir, 
                       normx_file = None, 
                       normy_file = None, 
                       norm_y = norm_y,
                       data_variable_names=["bed__sediment_volume_per_unit_area_0", 
                                            "bed__sediment_volume_per_unit_area_1", 
                                            "bed__sediment_volume_per_unit_area_2", 
                                            "bed__sediment_volume_per_unit_area_3"], 
                       val_list=['C0,1', 'C0,2', 'C0,3', 'C0,4', 'Salinity', 'U0', 'h0', 'Fd']
                       )
