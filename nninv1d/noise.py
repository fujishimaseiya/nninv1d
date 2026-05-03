"""This script is designed to test the robustness of a DNN against noise.
"""
import numpy as np
import matplotlib.pyplot as plt
from nninv1d import reproduce_y
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
import pdb
import glob
from natsort import natsorted
from pathlib import Path
import shutil
import re

def read_data(dirname):
    """Reads data from csv files.
    Args:
        dirname: Directory name where the data files are located.

    Returns:
        x_test: Test input data.
        y_test: Test target data.
        norm_y: Normalization parameters for the target data.
    """

    x_test = np.load(os.path.join(dirname, 'x_test'), allow_pickle=True)
    y_test = np.load(os.path.join(dirname, 'y_test.npy'))
    norm_y = np.load(os.path.join(dirname, 'norm_y.npy'))

    return x_test, y_test, norm_y    

def add_noise(x_test, noise_rate):
    """Adds noise to input data.

    Args:
        x_test: Original test input data.
        noise_rate: Rate of noise to be added to the input data.    
    """

    noise = np.random.normal(size=x_test.shape)
    x_test_with_noise = x_test + noise * noise_rate * x_test

    return x_test_with_noise

def predict(model, x_test, y_test, x_test_with_noise, norm_y):
    """Predicts the target values using the original and noisy input data.
    Args:
        model: Trained DNN model for prediction.
        x_test: Original test input data.
        y_test: Test target data.
        x_test_with_noise: Noisy test input data.
        norm_y: Normalization parameters for the target data.
    
    Returns:
        original_result: Predictions from the original input data.
        noise_result: Predictions from the noisy input data.
        original_truevalue: True values corresponding to the test data.
    """

    # prediction using original and noisy data
    original_result_norm = model.predict(x_test)
    original_result = reproduce_y(original_result_norm, norm_y)
    noise_result_norm = model.predict(x_test_with_noise)
    noise_result = reproduce_y(noise_result_norm, norm_y)
    # reproduction of true values
    original_truevalue = reproduce_y(y_test, norm_y)

    return original_result, noise_result, original_truevalue

def save_result(data, filename, save_dir):
    """Saving result to csv file

    Args:
        data: Data to be saved.
        filename: Name of the file to save the data.
        save_dir: Directory where the file will be saved.
    """
    np.savetxt(os.path.join(save_dir, filename), data)

def plot_result(original_result, noise_result, original_truevalue, val_name, units, save_dir):
    """Plots the original and noise results against the true values.

    Args:
        original_result: Predictions from the original input data.
        noise_result: Predictions from the noisy input data.
        original_truevalue: True values corresponding to the test data.
        val_name: List of parameter names for labeling the plots.
        units: List of units for labeling the axes.
        save_dir: Directory where the plots will be saved.
    """
    # set minimum value and maximum value of plot
    min_val = np.min(np.min(
                            [original_result, noise_result, original_truevalue], axis=0), axis=0
                    )
    min_val[min_val < 0] = 0
    max_val = np.max(np.max(
                            [original_result, noise_result, original_truevalue], axis=0), axis=0
                    )

    # plot
    for i in range(original_result.shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(3.93, 3.93))
        ax.plot(original_result[:, i],
                original_truevalue[:, i],
                'o',
                label='Without Noise')
        ax.plot(noise_result[:, i],
                original_truevalue[:, i],
                'o',
                label='With Noise')
        ax.plot([min_val[i], max_val[i] * 1.1],
                [min_val[i], max_val[i] * 1.1],
                marker=None,
                linewidth=2)
        ax.set_title(val_name[i], fontsize=18)
        ax.set_xlabel('Original Value' + units[i],
                        fontsize=14
                        )
        ax.set_ylabel('Reconstructed Value' + units[i],
                        fontsize=14
                        )
        ax.tick_params(labelsize=14)
        ax.set_aspect('equal')
        ax.legend()
        fig.patch.set_alpha(0)
        plt.savefig(os.path.join(save_dir, 'noise_test{}.pdf'.format(i)))
        plt.close(fig)

def calc_error(original_truevalue, noise_result, param_list, save_dir):
    """Calculates error metrics such as normalized RMSE, normalized bias, 
    and SMAPE between the original and noise results compared to the true values.

    Args:
        original_truevalue: True values corresponding to the test data.
        noise_result: Predictions from the noisy input data.
        param_list: List of parameter names for labeling the error metrics.
        save_dir: Directory where the error metrics will be saved.
    """
     
    num_dataset = original_truevalue.shape[0]
    num_param = original_truevalue.shape[1]
    stats_data = pd.DataFrame(index=['normalized_rmse', 'normalized_bias', 'smape'])

    param_name = []
    for j in range(len(param_list)):
        s = param_list[j]
        s = s.strip("'")
        s = s.replace('$', '')
        s = re.sub(r'\\mathrm\{(.*?)\}', r'\1', s)
        s = s.replace('{', '').replace('}', '')
        s = s.replace('\\', '')
        s = s.replace(', ', ',')
        param_name.append(s)
        
    # calculate normalized rmse and bias
    for i in range(num_param):
        normalized_rmse = np.sqrt(1/num_dataset * np.sum(((noise_result[:,i]-original_truevalue[:,i])/original_truevalue[:,i])**2))
        normalized_bias = 1/num_dataset * np.sum((noise_result[:,i]-original_truevalue[:,i])/original_truevalue[:,i])
        smape = 100/num_dataset * np.sum(2*np.abs(noise_result[:, i] - original_truevalue[:, i]) / (np.abs(noise_result[:, i]) + np.abs(original_truevalue[:, i])))
        new_stats = pd.Series([normalized_rmse, normalized_bias, smape], index=['normalized_rmse', 'normalized_bias', 'smape'])
        stats_data[param_name[i]] = new_stats
    mean_stats = pd.Series(stats_data.mean(axis=1), index=['normalized_rmse', 'normalized_bias', 'smape'])
    stats_data['mean'] = mean_stats
    stats_data.to_csv(os.path.join(save_dir, 'prediction_error_vs_input_noise.csv'))

def check_noise_robustness(model, x_test, y_test, norm_y, noise_rate, val_name, units, save_dir):
    """Checks the robustness of the model against noise by adding noise to the input data,
    making predictions, saving results, plotting the results, and calculating error metrics.
    
    Args:
        model: Trained DNN model for prediction.
        x_test: Original test input data.
        y_test: Test target data.
        norm_y: Normalization parameters for the target data.
        noise_rate: Rate of noise to be added to the input data.
        val_name: List of parameter names for labeling the plots and error metrics.
        units: List of units for labeling the axes in the plots.
        save_dir: Directory where the results, plots, and error metrics will be saved.
    """
    x_test_with_noise = add_noise(x_test=x_test, noise_rate=noise_rate)
    original_result, noise_result, original_truevalue = predict(model, x_test, y_test, x_test_with_noise, norm_y)
    save_result(data=original_result, filename='original_result.csv', save_dir=save_dir)
    save_result(data=noise_result, filename='noise_result.csv', save_dir=save_dir)
    save_result(data=original_truevalue, filename='original_truevalue.csv', save_dir=save_dir)
    plot_result(original_result=original_result, 
                noise_result=noise_result, 
                original_truevalue=original_truevalue, 
                val_name=val_name, 
                units=units,
                save_dir=save_dir)
    calc_error(original_truevalue=original_truevalue, noise_result=noise_result, param_list=val_name, save_dir=save_dir)

if __name__ == "__main__":
    pdb.set_trace()
    working_dir = '/mnt/sed_share2/fujishima/inv_confined/inv_result/num_data/num5000'
    x_test, y_test, norm_y = read_data(working_dir)
    model = load_model(os.path.join(working_dir, 'model.keras'))
    val_name = ['$C_{0, 1}$', '$C_{0, 2}$', '$U_{0}$', '$h_{0}$', '$T_{\mathrm{d}}$']
    units = ['', '', ' (m/s)', ' (m)', ' (s)']
    noise_rate = np.arange(0, 0.21, 0.01)
    num_trials = 2

    for trial in range(num_trials):
        save_parent_dir = os.path.join(working_dir, f'noise_test{trial}')
        if not os.path.exists(save_parent_dir):
            os.mkdir(save_parent_dir)
        shutil.copy(__file__, save_parent_dir)

        for i in range(len(noise_rate)):
            save_dir = os.path.join(save_parent_dir, 'noise_rate_{}'.format(noise_rate[i]))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            check_noise_robustness(model, 
                                x_test, 
                                y_test, 
                                norm_y,
                                noise_rate[i], 
                                val_name=val_name, 
                                units=units,
                                save_dir=save_dir)

    dirlist = glob.glob(os.path.join(working_dir, 'noise_test*'))
    dirlist = natsorted(dirlist)
    noise_rate_dir = [d.name for d in Path(dirlist[0]).glob('noise_rate_*') if d.is_dir()]
    noise_rate_dir = natsorted(noise_rate_dir)
    error_data = np.zeros(len(noise_rate_dir))
    for i in range(len(noise_rate_dir)):
        error_params = []
        for j in range(len(dirlist)):
            dir = os.path.join(dirlist[j], noise_rate_dir[i])
            filename = os.path.join(dir, 'prediction_error_vs_input_noise.csv')
            data = pd.read_csv(filename, header=0, index_col=0)
            error = data.loc["smape", "mean"]
            error_params.append(error)
        mean_error_param = np.mean(error_params)
        error_data[i] = mean_error_param
    print(error_data)

    error_rate = np.arange(0, 0.21, 0.01)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    ax.plot(error_rate, error_data, marker='o')
    ax.set_xlabel('r', fontsize=12)
    ax.set_ylabel('Mean SMAPE (%)', fontsize=12)
    fig.savefig(os.path.join(working_dir, 'error_noise_smape.pdf'))
