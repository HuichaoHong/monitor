import numpy as np
import pandas as pd
from pandas import datetime
from donut import complete_timestamp, standardize_kpi
from tfsnippet.utils import get_variables_as_dict, VariableSaver
import matplotlib.pyplot as plt
import tensorflow as tf
from donut import Donut
from tensorflow import keras as K
from tfsnippet.modules import Sequential

from donut import DonutTrainer, DonutPredictor

# read data WS_data
data = pd.read_csv('data_arti_sin2.csv', skiprows=[0, 1], header=None)
data = data.dropna()
values = data[2].values

labels = data[3].values
labels = np.zeros_like(values, dtype=np.int32)
date_str = data[0].values
date = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in date_str]
dateDelta = [x-date[0] for x in date]
timestamp = [x.days*1440+x.seconds/60 for x in dateDelta]

# Complete the timestamp, and obtain the missing point indicators.
timestamp, missing, (values, labels) = complete_timestamp(timestamp, (values, labels))

# Split the training and testing data.
test_portion = 0.3
test_n = int(len(values) * test_portion)
train_values, test_values = values[:-test_n], values[-test_n:]
train_labels, test_labels = labels[:-test_n], labels[-test_n:]
train_missing, test_missing = missing[:-test_n], missing[-test_n:]

# Standardize the training and testing data.
train_values, mean, std = standardize_kpi(train_values, excludes=np.logical_or(train_labels, train_missing))
test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

# We build the entire model within the scope of `model_vs`,
# it should hold exactly all the variables of `model`, including
# the variables created by Keras layers.
#Using keras to create layer
#As is shown in follow:
'''
Argc:
        h_for_p_x (Module or (tf.Tensor) -> tf.Tensor):
            The hidden network for :math:`p(x|z)`.
        h_for_q_z (Module or (tf.Tensor) -> tf.Tensor):
            The hidden network for :math:`q(z|x)`.
        x_dims (int): The number of `x` dimensions.
        z_dims (int): The number of `z` dimensions.
'''
with tf.variable_scope('model') as model_vs:
    model = Donut(
        h_for_p_x=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        h_for_q_z=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        x_dims=120,
        z_dims=5,
    )

# To train the Donut model, and use a trained model for prediction
trainer = DonutTrainer(model=model, model_vs=model_vs)
predictor = DonutPredictor(model)

with tf.Session().as_default():
    #trainer.fit(train_values, train_labels, train_missing, mean, std)
    #var_dict = get_variables_as_dict(model_vs)
    #saver = VariableSaver(var_dict, "donut_without_label_2.ckpt")
    #saver.save()

    # Restore variables from `save_dir`.
    saver = VariableSaver(get_variables_as_dict(model_vs), "donut_without_label_2.ckpt")
    saver.restore()
    test_score = predictor.get_score(test_values, test_missing)
    result = np.array([test_labels[119:], test_score])
    np.savetxt('result_arti_sin2.csv', result.transpose(), delimiter=',', fmt='%.3f')

