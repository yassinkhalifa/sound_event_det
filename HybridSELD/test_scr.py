import argparse
import os
import pdb
from timeit import default_timer as timer
import helper_classes
import h5py
import numpy as np
import utilities
from keras.models import load_model
from keras.layers import Bidirectional, Conv2D, DepthwiseConv2D, MaxPooling2D, Input
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import keras
import evaluation_metrics

dataset_dir = '/home/yassin/Desktop/DCASE19/dataset/' 
feature_dir = '/home/yassin/Desktop/DCASE19/dataset/logmel_features/'
log_dir = './logs'
feature_type = 'logmel' 
data_type = 'dev'
audio_type = 'foa'
splits = 4
fs = 48000
nfft = 1024
nb_channels = 4
nb_classes = 11
hopsize = 320
mel_bins = 128
frames_per_1s = fs // hopsize
chunklen = int(2 * frames_per_1s)
hopframes = int(0.5 * frames_per_1s)
batch_size = 10
nb_epochs = 100
patience = 10

hdf5_folder_name = '/home/yassin/Desktop/DCASE19/dataset/logmel_features/logmel/features/foa'

params = {
        'fs':fs,
        'nfft':nfft,
        'hopsize':hopsize,
        'mel_bins':mel_bins,
        'frames_per_1s':frames_per_1s,
        'chunklen':chunklen,
        'hopframes':hopframes,
        'batch_size': batch_size,
        'nb_epochs':nb_epochs,
        'patience':patience,
        'splits':splits,
        'nb_channels':nb_channels,
        'nb_classes':nb_classes,
        'nb_mel_bins':mel_bins,
        'pool_size':[2, 2, 2],
        'nb_cnn2d_filt': [128, 256, 512],
        'dropout_rate':0,
        'rnn_size':[256, 256],
        'log_dir': log_dir
    }

cls_data_gen = helper_classes.DataGenerator_SED(params, hdf5_folder_name, 1)

full_features, full_target_events, full_target_doas, full_target_dists = utilities.reshape_features(cls_data_gen.validation_features_list, cls_data_gen.validation_target_events_list, cls_data_gen.validation_target_doas_list, cls_data_gen.validation_target_dists_list, params['chunklen'])

model = load_model('/home/yassin/Desktop/DCASE19/summer_2019/HybridSELD/SED_best_model_split_0.h5')

opt = model.predict(full_features)

x = utilities.er_overall_framewise(evaluation_metrics.reshape_3Dto2D(opt), evaluation_metrics.reshape_3Dto2D(full_target_events))