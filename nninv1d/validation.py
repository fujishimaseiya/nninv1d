from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from nninv1d import reproduce_y, read_data, preprocess
import os
import pdb
import shutil

class Validation():

    def __init__(self, trdata_dir):
        self.trdata_dir =trdata_dir

    def load_model(self, model_path):
        model = load_model(model_path)

        return model

    def predict(self, model_path, savedir, normx_file, normy_file, data_variable_name, target_variable_names, cood_file, measuremnt_point_file):
        """This method is to read measeured data and predict                                                                                                                                                                                               \
                                                                                                                                                                                                                                                            
        """
        # original_dataset, target_dataset = read_data(self.trdata_dir, savedir,target_variable_names, data_variable_names, cood_file, measuremnt_point_file)
        # x_train, y_train, x_test, y_test, norm_y = preprocess(original_dataset, target_dataset, 100, savedir=savedir)
        norm_y = np.load(normy_file)
        # norm_y = norm_y[:, 1:]
        norm_x = np.load(normx_file)
        df = pd.read_csv(cood_file, header=0)
        df_hc = pd.read_csv(measuremnt_point_file, header=0)
        num_sed_variable = 4
        num_sed_datapoints = len(df.index)
        input_data_raw = np.empty(0)
        for i in range(num_sed_variable):
            sed_i = df[data_variable_name[i]].to_numpy()
            input_data_raw = np.append(input_data_raw, sed_i)
        num_cond_variable = 6
        #for j in range(num_cond_variable):
        #    cond_variable = df_hc[data_variable_name[num_sed_variable+j]].to_numpy()
        #    input_data = np.append(input_data, cond_variable)
        input_data = input_data_raw[np.newaxis, :]
        # input_data = (input_data_raw - norm_x[0]) / (norm_x[1] - norm_x[0])
        model = load_model(model_path)
        predict_result_raw = model.predict(input_data)
        predict_result = reproduce_y(predict_result_raw, norm_y)
        # data = pd.DataFrame(predict_result, columns=["Cf", "alpha_4eq", "r0", "C_0", "C_1", "C_2", "C_3", "salinity", "U_ini", "h_ini", "endtime"])
        data = pd.DataFrame(predict_result, columns=["C_0", "C_1", "C_2", "C_3", "Salinity", "U_ini", "h_ini", "endtime"])
        data.to_csv(os.path.join(savedir, "predict_result.csv"))

if __name__ == "__main__":
    # pdb.set_trace()
    model_path = "/mnt/sed_share2/fujishima/phd_research/data/exp2/run2/no_detrainment/no_erosion/after_cfrev/cf0.002_ro2_c0.0001-0.01/inv_result/nonormx_8000/model.keras"
    cood_file = '/mnt/sed_share2/fujishima/3d_model/exp2/diff_bed_run_1_2/sed_volume_run2/sed_dx0.05.csv'
    measuremnt_point_file = '/mnt/sed_share2/fujishima/param_optim/cood_data/exp2/run1/mea_point.csv'
    savedir = '/mnt/sed_share2/fujishima/phd_research/data/exp2/run2/no_detrainment/no_erosion/after_cfrev/cf0.002_ro2_c0.0001-0.01/inv_result/nonormx_8000/pred_result'
    trdata_path = '/mnt/sed_share2/fujishima/phd_research/data/exp2/run2/no_detrainment/no_erosion/after_cfrev/cf0.002_ro2_c0.0001-0.01/data'
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
    validation = Validation(trdata_path)
    #pdb.set_trace()
    validation.predict(model_path, savedir, normx_file, normy_file, data_variable_names, target_variable_names, cood_file, measuremnt_point_file)
