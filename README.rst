nninv1d
========================

Overview
---------------
This is a code for inverse analysis of sedimentary processes by using a deep learning neural network. 


Installation
---------------
You can install it by the following command::

    git clone https://github.com/fujishimaseiya/.nninv1d.git

Alternatively, you can install using the setup script::

    python setup.py install


Usage
---------------
To use this package, run the script `do_inv.py`.
The script internally defines and uses the following parameters:

- ``data_folder``: Path to the folder that contains training dataset.
- ``cood_file``: CSV file specifying coordinates of sampling points of deposits.
- ``resdir``: Output directory for results. Created automatically if not existing.
- ``target_variable_names``: Names of the target physical parameters to be estimated.
- ``data_variable_names``: Names of input data variables such as sediment volume per unit area.


Processing steps:
---------------

1. Load the input and target data from files.
2. Normalize and split into training and test sets.
3. Train the inverse model using deep learning.
4. Save and plot the training history.
5. Evaluate and visualize model performance on test data.
6. Predict using real (e.g., flume experiment) data.
7. Compare predicted and original values, then save them.


Outputs:
---------------

- The following files will be generated inside `<resdir>/`:

  - `test_result.csv`: Reconstructed values from the model
  - `test_original.csv`: Original target values
  - `test_resultX.svg`: Scatter plots for each parameter (reconstructed vs. original)
  - `predict_result.csv`: Predicted values from the inverse model using real data`
  - Training history plots and logs