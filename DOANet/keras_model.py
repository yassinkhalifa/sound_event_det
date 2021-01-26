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

keras.backend.set_image_data_format('channels_last')


def get_model(params, network_params):
    x=Input(shape=(params['sequence_len'], params['nb_freq_bins'], 2*params['nb_channels']))

    spec_cnn = x

    #1st part of DOAnet: 4 layers of 2D CNN, 64 3*3 filters each:
    for filt_idx, filt_count in enumerate(network_params['specCNN_filt']):
        spec_cnn = Conv2D(filters=filt_count, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization(axis=-2)(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, network_params['pool_size_cnn'][filt_idx]))(spec_cnn)
        spec_cnn = Dropout(network_params['dropout_rate'])(spec_cnn)

    #2nd part of DOAnet: 2 layers of bidirectional RNN with GRU and tanh (64 units):
    spec_rnn = Reshape((params['sequence_len'], -1))(spec_cnn)
    for gru_idx, gru_units in enumerate(network_params['specRNN_size']):
        spec_rnn = Bidirectional(
            GRU(gru_units, activation='tanh', dropout=network_params['dropout_rate'], recurrent_dropout=network_params['dropout_rate'],
                return_sequences=True),
            merge_mode='concat'
        )(spec_rnn)

    spec_fc = Reshape((params['sequence_len'], -1))(spec_rnn)
    spec_fc = TimeDistributed(Dense(network_params['specfcn_size']))(spec_fc)
    spec_fc = Activation('linear')(spec_fc)

    spspec_cnn = Reshape((params['sequence_len'], network_params['specfcn_size'], 1))(spec_fc)
    spspec_cnn = Conv2D(filters=network_params['spspecCNN_filt'][0], kernel_size=(3, 3), padding='same')(spspec_cnn)
    spspec_cnn = BatchNormalization(axis=-2)(spspec_cnn)
    spspec_cnn = Activation('relu')(spspec_cnn)
    spspec_cnn = MaxPooling2D(pool_size=(1, network_params['pool_size_spcnn']))(spspec_cnn)
    spspec_cnn = Dropout(network_params['dropout_rate'])(spspec_cnn)

    spspec_cnn = Conv2D(filters=network_params['spspecCNN_filt'][1], kernel_size=(3, 3), padding='same')(spspec_cnn)
    spspec_cnn = BatchNormalization(axis=-2)(spspec_cnn)
    spspec_cnn = Activation('relu')(spspec_cnn)
    spspec_cnn = Dropout(network_params['dropout_rate'])(spspec_cnn)


    spspec_fc = Reshape((params['sequence_len'], -1))(spspec_cnn)
    spspec_fc = TimeDistributed(Dense(network_params['spspecfcn_size'][0]))(spspec_fc)
    spspec_fc = Activation('linear')(spspec_fc)

    spspec_rnn = Reshape((params['sequence_len'], -1))(spspec_fc)
    for gru_idx, gru_units in enumerate(network_params['spspecRNN_size']):
        spspec_rnn = Bidirectional(
            GRU(gru_units, activation='tanh', dropout=network_params['dropout_rate'], recurrent_dropout=network_params['dropout_rate'],
                return_sequences=True),
            merge_mode='concat'
        )(spspec_rnn)

    spspec_fc_2 = Reshape((params['sequence_len'], -1))(spspec_rnn)
    #spspec_fc_2 = TimeDistributed(Dense(network_params['spspecfcn_size'][1]))(spspec_fc_2)
    #doa = Activation('sigmoid', name='doa_out')(spspec_fc_2)
    spspec_fc_2 = TimeDistributed(Dense(2*params['nb_classes']))(spspec_fc_2)
    doa = Activation('linear', name='doa_out')(spspec_fc_2)

    model = Model(inputs=x, outputs=[doa])
    model.compile(optimizer=Adam(), loss=['mse'])

    model.summary()
    return model