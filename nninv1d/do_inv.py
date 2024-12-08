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
    # pdb.set_trace()
    data_folder = '/mnt/sed_share2/fujishima/phd_research/data/exp2/run2/no_detrainment/no_erosion/after_cfrev/cf0.002_ro2/data'
    resdir = '/mnt/sed_share2/fujishima/phd_research/data/exp2/run2/no_detrainment/no_erosion/after_cfrev/cf0.002_ro2/inv_result/test'
    if not os.path.exists(resdir):
        os.mkdir(resdir)
    cood_file = '/mnt/sed_share2/fujishima/3d_model/exp2/diff_bed_run_1_2/sed_volume_run2/sed_dx0.05.csv'
    # cood_file = '/home/biosphere/sed_share2/fujishima/phd_research/data/exp1/detrainment/cf0.002_detrate1.0/samp_point2.csv'
    # cood_file = '/mnt/sed_share2/fujishima/phd_research/data/exp3/exp3_cf0.002_ro2_noerosion/after_revcf/inv_result/nonormx_8000/samp_idx_exp3.csv'
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
    
    for i in range(check_result.shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(3.93, 3.93),tight_layout=True)
        ax.plot(check_original[:, i], check_result[:, i], 'o')
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
        plt.savefig(os.path.join(resdir, 'check_result{}.svg'.format(i)))
        ax.cla()