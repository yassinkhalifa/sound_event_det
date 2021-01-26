import os
import sys
import numpy as np
from keras.models import load_model
import time
from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, Flatten
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

def get_model(doa_params):
    x = Input(shape=(2*doa_params['nb_channels'], doa_params['chunklen'], doa_params['nb_freq_bins']))
    ten_cnn = x
    #CNN
    for cnn_cnt, cnn_pool in enumerate(doa_params['pool_size']):
        ten_cnn = Conv2D(filters=doa_params['nb_cnn2d_filt'], kernel_size=(3, 3), padding='same')(ten_cnn)
        ten_cnn = BatchNormalization()(ten_cnn)
        ten_cnn = Activation('relu')(ten_cnn)
        ten_cnn = MaxPooling2D(pool_size=(1, cnn_pool))(ten_cnn)
        ten_cnn = Dropout(doa_params['dropout_rate'])(ten_cnn)
    ten_cnn = Permute((2, 1, 3))(ten_cnn)
    #RNN
    ten_rnn = Reshape((doa_params['chunklen'], -1))(ten_cnn)
    for nb_rnn in doa_params['rnn_size']:
        ten_rnn = Bidirectional(
            GRU(nb_rnn, activation='tanh', dropout=doa_params['dropout_rate'], recurrent_dropout=doa_params['dropout_rate'],
                return_sequences=True),
            merge_mode='mul'
        )(ten_rnn)
    
    doa = ten_rnn
    for nb_fcn in doa_params['fcn_size']:
        doa = TimeDistributed(Dense(nb_fcn))(doa)
        doa = Dropout(doa_params['dropout_rate'])(doa)
    doa = TimeDistributed(Dense(4*doa_params['nb_classes']))(doa)
    doa = Reshape((doa_params['chunklen'], 2*doa_params['nb_classes'], 2))(doa)
    doa = BatchNormalization()(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    model = Model(inputs=x, outputs=[doa])
    model.compile(optimizer=Adam(), loss=[biternion_loss])
    model.summary()
    return model