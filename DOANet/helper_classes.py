import os
import numpy as np
from IPython import embed
from collections import deque
import random
from keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, datadir = '', labelsdir = '', shuffle=False, perfile=False, train=True, default_azi=180, default_ele=50, splits = [0], win_len = 480, hop_len = 240, seq_len=100, nfft = 2048, nb_channels = 4, nb_classes = 11, batch_size = 32):
        self._datadir = datadir
        self._labelsdir = labelsdir
        self._shuffle = shuffle
        self._train = train
        self._win_len = win_len
        self._hop_len = hop_len
        self._nfft = nfft
        self._nb_channels = nb_channels
        self._nb_classes = nb_classes
        self._splits = splits
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._perfile = perfile
        self._default_azi = default_azi
        self._default_ele = default_ele
        self._files_list = os.listdir(self._datadir)
        self._splits_list = []
        self._current_files_list = []
        for file in self._files_list:
            if int(file[5]) in self._splits:
                self._current_files_list.append(file)
                temp_labels = np.load(os.path.join(self._labelsdir, file))
                temp_max_frames = temp_labels.shape[0]
                for frame_cnt in range(0, int(np.floor(temp_max_frames/self._seq_len))):
                    start_idx = frame_cnt*self._seq_len
                    end_idx = (frame_cnt+1)*self._seq_len
                    split_info = {'file_name':file, 'start_idx':start_idx, 'end_idx':end_idx}
                    self._splits_list.append(split_info)
        if self._shuffle:
            random.shuffle(self._splits_list)
        self._files_splits_buffer = deque(self._splits_list)
        if self._perfile:
            self._nb_total_batches = len(self._current_files_list)
        else:
            self._nb_total_batches = int(np.floor(len(self._splits_list)/self._batch_size))
        self.on_epoch_end()

    def on_epoch_end(self):
        if self._train:
            if self._shuffle:
                random.shuffle(self._splits_list)
            self._files_splits_buffer = deque(self._splits_list)
    
    def __data_generation(self, index):
        #batch_info = self._files_splits_buffer.popleft()
        batch_x = np.zeros((self._batch_size, self._seq_len, int(self._nfft/2), 2*self._nb_channels))
        batch_y = np.zeros((self._batch_size, self._seq_len, 3*self._nb_classes))
        seq_count = 0
        for seq_cnt in range(index*self._batch_size, (index+1)*self._batch_size):
            batch_info = self._splits_list[seq_cnt]
            data_x = np.load(os.path.join(self._datadir, batch_info['file_name']))
            data_y = np.load(os.path.join(self._labelsdir, batch_info['file_name']))
            batch_x[seq_count,:,:,:] = data_x[batch_info['start_idx']:batch_info['end_idx'], :].reshape((self._seq_len, int(self._nfft/2), 2*self._nb_channels))
            batch_y[seq_count,:,:] = data_y[batch_info['start_idx']:batch_info['end_idx'], :]
            seq_count += 1
        azi_rad = batch_y[:, :, self._nb_classes:2 * self._nb_classes] * np.pi / 180
        ele_rad = batch_y[:, :, 2 * self._nb_classes:] * np.pi / self._default_ele
        batch_y = np.concatenate((azi_rad, ele_rad), -1)  # DOA labels in radians
        return batch_x, batch_y

    def __len__(self):
        return self._nb_total_batches

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        return X, y
    
    def get_val_labels(self):
        batch_y = np.zeros((self._nb_total_batches*self._batch_size, self._seq_len, 3*self._nb_classes))
        seq_count = 0
        for seq_cnt in range(0, self._nb_total_batches*self._batch_size):
            batch_info = self._splits_list[seq_cnt]
            data_y = np.load(os.path.join(self._labelsdir, batch_info['file_name']))
            batch_y[seq_count,:,:] = data_y[batch_info['start_idx']:batch_info['end_idx'], :]
            seq_count += 1
        azi_rad = batch_y[:, :, self._nb_classes:2 * self._nb_classes] * np.pi / 180
        ele_rad = batch_y[:, :, 2 * self._nb_classes:] * np.pi / self._default_ele
        batch_y = [
            batch_y[:, :, :self._nb_classes],  # SED labels
            np.concatenate((azi_rad, ele_rad), -1)  # DOA labels in radians
            ]
        return batch_y[0], batch_y[1]

    def get_nb_total_batches(self):
        return self._nb_total_batches
