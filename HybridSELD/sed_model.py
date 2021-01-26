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

def get_model(sed_params):
    x = Input(shape=(sed_params['nb_channels'], sed_params['chunklen'], sed_params['mel_bins']))
    sed_cnn = x
    #CNN
    for cnn_cnt, cnn_pool in enumerate(sed_params['pool_size']):
        sed_cnn = Conv2D(filters=sed_params['nb_cnn2d_filt'][cnn_cnt], kernel_size=(3, 3), padding='same')(sed_cnn)
        sed_cnn = BatchNormalization()(sed_cnn)
        sed_cnn = Activation('relu')(sed_cnn)
        sed_cnn = Conv2D(filters=sed_params['nb_cnn2d_filt'][cnn_cnt], kernel_size=(3, 3), padding='same')(sed_cnn)
        sed_cnn = BatchNormalization()(sed_cnn)
        sed_cnn = Activation('relu')(sed_cnn)
        sed_cnn = MaxPooling2D(pool_size=(1, cnn_pool))(sed_cnn)
        sed_cnn = Dropout(sed_params['dropout_rate'])(sed_cnn)
    
    sed_cnn = Permute((2, 1, 3))(sed_cnn)
    sed_rnn = Reshape((sed_params['chunklen'], -1))(sed_cnn)
    for nb_rnn in sed_params['rnn_size']:
        sed_rnn = Bidirectional(
            GRU(nb_rnn, activation='tanh', dropout=sed_params['dropout_rate'], recurrent_dropout=sed_params['dropout_rate'],
                return_sequences=True),
            merge_mode='concat'
        )(sed_rnn)
    
    sed = sed_rnn
    sed = TimeDistributed(Dense(sed_params['nb_classes']))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)
    
    model = Model(inputs=x, outputs=[sed])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy'])
    model.summary()
    return model
