"""This script is designed to test the robustness of a DNN against noise.
"""
import numpy as np
import matplotlib.pyplot as plt
from nninv1d import reproduce_y

def add_noise(x_test, noise_rate):
    """Adds noise to input data.
    """

    noise = np.random.normal(size=x_test.shape)
    x_test_with_noise = x_test + noise * noise_rate * x_test

    return x_test_with_noise

def predict(model, x_test, y_test, x_test_with_noise):

    # prediction using original and noisy data
    original_result_norm = model.predict(x_test)
    original_result = reproduce_y(original_result_norm)
    noise_result_norm = model.predict(x_test_with_noise)
    noise_result = reproduce_y(noise_result_norm)
    # reproduction of true values
    original_truevalue = reproduce_y(y_test)

    return original_result, noise_result, original_truevalue

def save_result(data, filename):
    """Saving result to csv file
    """
    np.savetxt(filename, data)

def plot_result(original_result, noise_result, original_truevalue, val_name, units):
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
                plt.savefig('noise_test{}.svg'.format(i))
                ax.cla()

def check_noise_robustness(model, x_test, y_test, noise_rate, val_name, units):
        x_test_with_noise = add_noise(x_test=x_test, noise_rate=noise_rate)
        original_result, noise_result, original_truevalue = predict(model, x_test, y_test, x_test_with_noise)
        save_result(data=original_result, filename='original_result.csv')
        save_result(data=noise_result, filename='noise_result.csv')
        save_result(data=original_truevalue, filename='original_truevalue.csv')
        plot_result(original_result=original_result, 
                    noise_result=noise_result, 
                    original_truevalue=original_truevalue, 
                    val_name=val_name, 
                    units=units)