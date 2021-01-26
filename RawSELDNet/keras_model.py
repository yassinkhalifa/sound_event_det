import os
import sys
import numpy as np
from keras.models import load_model
import time
from keras.layers import Bidirectional, Conv1D, MaxPooling1D, Input, Flatten
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import keras
import tensorflow as tf
from IPython import embed

def get_model(params, network_params):
    x=Input(shape=(int(params['win_len']*params['fs']), params['nb_channels']))
    raw_cnn = x
    
    #VGG19-like architecture

    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][0], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][0], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = BatchNormalization()(raw_cnn)
    raw_cnn = MaxPooling1D(pool_size=network_params['pool_size'][0])(raw_cnn)

    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][1], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][1], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = BatchNormalization()(raw_cnn)
    raw_cnn = MaxPooling1D(pool_size=network_params['pool_size'][1])(raw_cnn)

    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][2], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][2], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][2], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][2], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = BatchNormalization()(raw_cnn)
    raw_cnn = MaxPooling1D(pool_size=network_params['pool_size'][2])(raw_cnn)

    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][3], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][3], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][3], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][3], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = BatchNormalization()(raw_cnn)
    raw_cnn = MaxPooling1D(pool_size=network_params['pool_size'][3])(raw_cnn)

    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][4], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][4], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][4], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = Conv1D(filters=network_params['nb_cnn_filt'][4], kernel_size=3, padding='same', activation='relu', data_format='channels_last')(raw_cnn)
    raw_cnn = BatchNormalization()(raw_cnn)
    raw_cnn = MaxPooling1D(pool_size=network_params['pool_size'][4])(raw_cnn)

    # RNN
    raw_rnn = raw_cnn
    for nb_rnn_filt in network_params['rnn_size']:
        raw_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=network_params['dropout_rate'], recurrent_dropout=network_params['dropout_rate'],
                return_sequences=True),
            merge_mode='mul'
        )(raw_rnn)

    # FC - DOA
    doa = Flatten()(raw_rnn)
    for nb_fnn in network_params['fnn_size']:
        doa = Dense(nb_fnn, activation='relu')(doa)
        doa = Dropout(network_params['dropout_rate'])(doa)

    doa = Dense(2*params['nb_classes'])(doa)
    doa = Activation('linear', name='doa_out')(doa)

    # FC - SED
    sed = Flatten()(raw_rnn)
    for nb_fnn in network_params['fnn_size']:
        sed = Dense(nb_fnn, activation='relu')(sed)
        sed = Dropout(network_params['dropout_rate'])(sed)

    sed = Dense(params['nb_classes'])(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    model = Model(inputs=x, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=params['loss_weights'])

    model.summary()
    return model