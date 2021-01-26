import datetime
import itertools
import logging
import os
import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from librosa.display import specshow
from tqdm import tqdm


event_labels = ['knock', 'drawer', 'clearthroat', 'phone', 'keysDrop',\
    'speech', 'keyboard', 'pageturn', 'cough', 'doorslam', 'laughter']
lb_to_ix = {lb: i for i, lb in enumerate(event_labels)}
ix_to_lb = {i: lb for i, lb in enumerate(event_labels)}

azimuths = range(-180, 171, 10)
elevations = range(-40, 41, 10)
doa = [azimuths, elevations]
doa_labels = list(itertools.product(*doa))
doa_to_ix = {doa: i for i, doa in enumerate(doa_labels)}
ix_to_doa = {i: doa for i, doa in enumerate(doa_labels)}

train_splits_dict = {1: [2,3,4], 2: [1,3,4], 3: [1,2,4], 4: [1,2,3], -1: [1,2,3,4]}
validation_split_dict = {1: [1], 2: [2], 3: [3], 4: [4], -1: []}
test_split_dict = {1: [1], 2: [2], 3: [3], 4: [4], -1: []}



def get_doas(indexes):
    '''
    Get multiple doas from indexes
    '''
    doas = []
    for idx in indexes:
        doas.append(ix_to_doa[idx])
    return doas


def calculate_scalar(features):

    mean = []
    std = []

    channels = features.shape[0]
    for channel in range(channels):
        feat = features[channel, :, :]
        mean.append(np.mean(feat, axis=0))
        std.append(np.std(feat, axis=0))

    mean = np.array(mean)
    std = np.array(std)
    mean = np.expand_dims(mean, axis=0)
    std = np.expand_dims(std, axis=0)
    mean = np.expand_dims(mean, axis=2)
    std = np.expand_dims(std, axis=2)

    return mean, std

def reshape_features(features, target_events, target_doas, target_dists, chunklen):
    full_features = []
    full_target_events = []
    full_target_doas = []
    full_target_dists = []
    for feat_cnt, file_features in enumerate(features):
        frame_num = file_features.shape[1]
        chunk_cnt = int(frame_num/chunklen)
        new_features = np.zeros((chunk_cnt, file_features.shape[0], chunklen, file_features.shape[2]))
        new_events = np.zeros((chunk_cnt, chunklen, target_events[feat_cnt].shape[1]))
        new_doas = np.zeros((chunk_cnt, chunklen, target_doas[feat_cnt].shape[1]))
        new_dists = np.zeros((chunk_cnt, chunklen, target_dists[feat_cnt].shape[1]))
        for chunk_idx in range(chunk_cnt):
            new_features[chunk_idx, :, :, :] = file_features[:, chunk_idx*chunklen:(chunk_idx+1)*chunklen, :]
            new_events[chunk_idx, :, :] = target_events[feat_cnt][chunk_idx*chunklen:(chunk_idx+1)*chunklen, :]
            new_doas[chunk_idx, :, :] = target_doas[feat_cnt][chunk_idx*chunklen:(chunk_idx+1)*chunklen, :]
            new_dists[chunk_idx, :, :] = target_dists[feat_cnt][chunk_idx*chunklen:(chunk_idx+1)*chunklen, :]
        full_features.append(new_features)
        full_target_events.append(new_events)
        full_target_doas.append(new_doas)
        full_target_dists.append(new_dists)
    full_features = np.concatenate(full_features, axis=0)
    full_target_events = np.concatenate(full_target_events, axis=0)
    full_target_doas = np.concatenate(full_target_doas, axis=0)
    full_target_dists = np.concatenate(full_target_dists, axis=0)

    return full_features, full_target_events, full_target_doas, full_target_dists