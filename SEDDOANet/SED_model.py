import os
import sys
import numpy as np
from keras.layers import Bidirectional, Conv2D, DepthwiseConv2D, MaxPooling2D, Input
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import keras

keras.backend.set_image_data_format('channels_first')

def get_model(params, sed_params):
    x = Input(shape=(params['nb_channels'], params['sequence_len'], params['nb_freq_bins']))
    ten_cnn = x
    #CNN
    for cnn_cnt, cnn_pool in enumerate(sed_params['pool_size']):
        ten_cnn = Conv2D(filters=sed_params['nb_cnn2d_filt'], kernel_size=(3, 3), padding='same')(ten_cnn)
        ten_cnn = BatchNormalization()(ten_cnn)
        ten_cnn = Activation('relu')(ten_cnn)
        ten_cnn = MaxPooling2D(pool_size=(1, cnn_pool))(ten_cnn)
        ten_cnn = Dropout(sed_params['dropout_rate'])(ten_cnn)
    ten_cnn = Permute((2, 1, 3))(ten_cnn)
    #RNN
    ten_rnn = Reshape((params['sequence_len'], -1))(ten_cnn)
    for nb_rnn in sed_params['rnn_size']:
        ten_rnn = Bidirectional(
            GRU(nb_rnn, activation='tanh', dropout=sed_params['dropout_rate'], recurrent_dropout=sed_params['dropout_rate'],
                return_sequences=True),
            merge_mode='mul'
        )(ten_rnn)
    # FCN - SED
    sed = ten_rnn
    for nb_fcn in sed_params['fcn_size']:
        sed = TimeDistributed(Dense(nb_fcn))(sed)
        sed = Dropout(sed_params['dropout_rate'])(sed)
    sed = TimeDistributed(Dense(params['nb_classes']))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    model = Model(inputs=x, outputs=[sed])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy'])
    model.summary()
    return model