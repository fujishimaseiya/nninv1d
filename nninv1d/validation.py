from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from nninv1d import reproduce_y, read_data, preprocess
import os
import pdb
import shutil

class Validation():

    def __init__(self):
        pass

    def load_model(self, model_path):
        model = load_model(model_path)

        return model

    def predict(self, model, cood_file, savedir, normx_file, normy_file, norm_y, data_variable_names, val_list):
        """This method is to read measeured data and predict                                                                                                                                                                                               \
                                                                                                                                                                                                                                                            
        """
        if normy_file is None and norm_y is not None:
            norm_y = norm_y
        elif normy_file is not None:
            norm_y = np.load(normy_file)
        if normx_file is not None:
            norm_x = np.load(normx_file)

        df = pd.read_csv(cood_file, header=0)
        num_sed_variable = len(data_variable_names)
        input_data_raw = np.empty(0)
        for i in range(num_sed_variable):
            sed_i = df[data_variable_names[i]].to_numpy()
            input_data_raw = np.append(input_data_raw, sed_i)

        if normx_file is None:
            input_data = input_data_raw[np.newaxis, :]
        elif normx_file is not None:
            input_data = (input_data_raw - norm_x[0]) / (norm_x[1] - norm_x[0])
            input_data = input_data_raw[np.newaxis, :]

        model = model
        predict_result_raw = model.predict(input_data)
        predict_result = reproduce_y(predict_result_raw, norm_y)
        data = pd.DataFrame(predict_result, columns=val_list)
        data.to_csv(os.path.join(savedir, "predict_result.csv"))

if __name__ == "__main__":
    # pdb.set_trace()
    model_path = "/mnt/sed_share2/fujishima/phd_research/data/exp2/run2/no_detrainment/no_erosion/after_cfrev/cf0.002_ro2_c0.0001-0.01/inv_result/nonormx_8000/model.keras"
    cood_file = '/mnt/sed_share2/fujishima/3d_model/exp2/diff_bed_run_1_2/sed_volume_run2/sed_dx0.05.csv'
    measuremnt_point_file = '/mnt/sed_share2/fujishima/param_optim/cood_data/exp2/run1/mea_point.csv'
    savedir = '/mnt/sed_share2/fujishima/phd_research/data/exp2/run2/no_detrainment/no_erosion/after_cfrev/cf0.002_ro2_c0.0001-0.01/inv_result/nonormx_8000/pred_result'
    normx_file = '/mnt/sed_share2/fujishima/phd_research/data/exp2/run2/no_detrainment/no_erosion/after_cfrev/cf0.002_ro2_c0.0001-0.01/inv_result/nonormx_8000/norm_x.npy'
    normy_file = '/mnt/sed_share2/fujishima/phd_research/data/exp2/run2/no_detrainment/no_erosion/after_cfrev/cf0.002_ro2_c0.0001-0.01/inv_result/nonormx_8000/norm_y.npy'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    shutil.copy("validation.py", savedir)
    shutil.copy(cood_file, savedir)
    target_variable_names = ["C_ini", "U_ini", "h_ini", "endtime"]
    #target_variable_names = ["Cf", "alpha_4eq", "r0", "C_ini", "U_ini", "h_ini", "endtime"]
    #target_variable_names = ["Cf", "r0", "C_ini", "U_ini", "h_ini", "endtime"]
    #data_variable_names = ["sed_volume_per_unit_area_0", "sed_volume_per_unit_area_1", "sed_volume_per_unit_area_2", "sed_volume_per_unit_area_3",
    #                        "layer_ave_vel", "layer_ave_conc_0", "layer_ave_conc_1", "layer_ave_conc_2", "layer_ave_conc_3", "flow_depth"]
    data_variable_names = ["bed__sediment_volume_per_unit_area_0", "bed__sediment_volume_per_unit_area_1", "bed__sediment_volume_per_unit_area_2", "bed__sediment_volume_per_unit_area_3"]
    validation = Validation()
    #pdb.set_trace()
    validation.predict(model_path, savedir, normx_file, normy_file, data_variable_names, target_variable_names, cood_file, measuremnt_point_file)
