import os
import sys
import numpy as np
from keras.models import load_model
import time
from keras.layers import Bidirectional, Conv2D, DepthwiseConv2D, MaxPooling2D, Input
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import keras
import tensorflow as tf
from IPython import embed

keras.backend.set_image_data_format('channels_first')

def get_model(params, network_params):
    x=Input(shape=(2*params['nb_channels'], params['sequence_len'], params['freq_bins']))
    ten_cnn = x
    #CNN
    for i, convCnt in enumerate(network_params['pool_size']):
        ten_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'], kernel_size=(3, 3), padding='same')(ten_cnn)
        ten_cnn = BatchNormalization()(ten_cnn)
        ten_cnn = Activation('relu')(ten_cnn)
        ten_cnn = MaxPooling2D(pool_size=(1, network_params['pool_size'][i]))(ten_cnn)
        ten_cnn = Dropout(network_params['dropout_rate'])(ten_cnn)
    #RNN
    ten_cnn = Permute((2,1,3))(ten_cnn)
    ten_rnn = Reshape((params['sequence_len'], -1))(ten_cnn)
    for nb_rnn in network_params['rnn_size']:
        ten_rnn = Bidirectional(
            GRU(nb_rnn, activation='tanh', dropout=network_params['dropout_rate'], recurrent_dropout=network_params['dropout_rate'],
                return_sequences=True),
            merge_mode='mul'
        )(ten_rnn)
    #FCN - DOA
    doa = ten_rnn
    for nb_fcn in network_params['fcn_size']:
        doa = TimeDistributed(Dense(nb_fcn))(doa)
        doa = Dropout(network_params['dropout_rate'])(doa)
    doa = TimeDistributed(Dense(params['nb_classes']))(doa)
    doa = Activation('linear', name='doa_out')(doa)
    # FCN - SED
    sed = ten_rnn
    for nb_fcn in network_params['fcn_size']:
        sed = TimeDistributed(Dense(nb_fcn))(sed)
        sed = Dropout(network_params['dropout_rate'])(sed)
    sed = TimeDistributed(Dense(params['nb_classes']))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    model = Model(inputs=x, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=params['loss_weights'])
    model.summary()
    return model