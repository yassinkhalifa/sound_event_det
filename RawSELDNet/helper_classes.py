import os
import numpy as np
from IPython import embed
from collections import deque
import random
from keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, h5_file, frame_label, frame_file, frame_fileorder, shuffle=False, perfile=False, train=True, default_azi=180, default_ele=50, splits = [1], max_frames = 1, fs=48000, win_len_s = 0.25, hop_len_s = 0.125, nb_channels = 4, nb_classes = 11, batch_size = 128):
        self._h5_file = h5_file
        self._frame_label = frame_label
        self._frame_file = frame_file
        self._frame_fileorder = frame_fileorder
        self._shuffle = shuffle
        self._train = train
        self._max_frames = max_frames
        self._fs = fs
        self._win_len = int(win_len_s*fs)
        self._hop_len = int(hop_len_s*fs)
        self._nb_channels = nb_channels
        self._nb_classes = nb_classes
        self._splits = splits
        self._batch_size = batch_size
        self._perfile = perfile
        self._default_azi = default_azi
        self._default_ele = default_ele
        self._custom_idxs = []
        for file_idx, file in enumerate(self._frame_file):
            if int(file[5]) in self._splits:
                self._custom_idxs.append(file_idx)
        self._custom_idxs = np.array(self._custom_idxs)
        if self._shuffle:
            random.shuffle(self._custom_idxs)
        if self._perfile:
            self._batch_size = self._max_frames
        self._nb_total_batches = int(np.floor(len(self._custom_idxs)/self._batch_size))
        self.on_epoch_end()

    def on_epoch_end(self):
        if self._train:
            if self._shuffle:
                random.shuffle(self._custom_idxs)
    
    def __data_generation(self, index):
        #batch_info = self._files_splits_buffer.popleft()
        batch_x = np.zeros((self._batch_size, self._win_len, self._nb_channels))
        batch_y = np.zeros((self._batch_size, 3*self._nb_classes))
        batch_idxs = np.sort(self._custom_idxs[(index*self._batch_size):((index+1)*self._batch_size)])
        batch_x = self._h5_file.data[batch_idxs, :]
        batch_y = self._frame_label[batch_idxs, :]
        azi_rad = batch_y[:, self._nb_classes:2 * self._nb_classes] * np.pi / 180
        ele_rad = batch_y[:, 2 * self._nb_classes:] * np.pi / self._default_ele
        batch_y = [
            batch_y[:, :self._nb_classes],  # SED labels
            np.concatenate((azi_rad, ele_rad), -1)  # DOA labels in radians
            ]
        return batch_x, batch_y

    def __len__(self):
        return self._nb_total_batches

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        return X, y
    
    def get_val_labels(self):
        batch_y = self._frame_label[self._custom_idxs[:self._nb_total_batches*self._batch_size], :]
        azi_rad = batch_y[:, self._nb_classes:2 * self._nb_classes] * np.pi / 180
        ele_rad = batch_y[:, 2 * self._nb_classes:] * np.pi / self._default_ele
        batch_y = [
            batch_y[:, :self._nb_classes],  # SED labels
            np.concatenate((azi_rad, ele_rad), -1)  # DOA labels in radians
            ]
        return batch_y[0], batch_y[1]

    def get_nb_total_batches(self):
        return self._nb_total_batches