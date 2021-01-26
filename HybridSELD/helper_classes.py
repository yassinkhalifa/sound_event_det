import logging
import os
import pdb
from timeit import default_timer as timer
import random
import h5py
import numpy as np
from keras.utils import Sequence

from utilities import (doa_labels, doa_to_ix, event_labels, lb_to_ix,
                             test_split_dict, train_splits_dict,
                             validation_split_dict)


class DataGenerator_SED(Sequence):
    def __init__(self, args, hdf5_dir, fold):
        """

        Inputs:
            args: all parameters
            hdf5_fn: str, path of hdf5 data
            logging: logging file
        """

        # Parameters
        self.fs = args['fs']
        self.nb_channels = args['nb_channels']
        self.nfft = args['nfft']
        self.hopsize = args['hopsize']
        self.mel_bins = args['mel_bins']
        self.chunklen = args['chunklen']
        self.hopframes = args['hopframes']
        self.batch_size = args['batch_size']
        self._training_batches = 0
        self.class_num = len(event_labels)

        train_splits = train_splits_dict[fold]
        validation_splits = validation_split_dict[fold]

        hdf5_dev_dir = os.path.join(hdf5_dir + '_dev/')
        hdf5_fns = sorted(os.listdir(hdf5_dev_dir))
        
        self.train_hdf5_fn = [fn for fn in hdf5_fns if int(fn[5]) in train_splits]
        self.validation_hdf5_fn = [fn for fn in hdf5_fns if int(fn[5]) in validation_splits]

        # Load the segmented data
        load_begin_time = timer()

        # Train data
        pointer = 0
        self.train_features_list = []
        self.train_fn_list = []
        self.train_target_events_list = []
        self.train_target_doas_list = []
        self.train_target_dists_list = []
        self.train_segmented_indexes = []
        
        for hdf5_fn in self.train_hdf5_fn:
            
            fn = hdf5_fn.split('.')[0]
            hdf5_path = os.path.join(hdf5_dev_dir, hdf5_fn)
            feature, target_event, target_doa, target_dist = \
                self.load_hdf5(hdf5_path)

            train_index = []
            # segment, keep only indexes
            frame_num = feature.shape[1]
            if frame_num > self.chunklen:
                train_index = np.arange(pointer, pointer+frame_num-self.chunklen+1, self.hopframes).tolist()
                if (frame_num - self.chunklen) % self.hopframes != 0:
                    train_index.append(pointer+frame_num-self.chunklen)
            elif frame_num < self.chunklen:
                feature = np.concatenate(
                    (feature, \
                        -100*np.ones((feature.shape[0],self.chunklen-frame_num,feature.shape[-1]))), axis=1)
                target_event = np.concatenate(
                    (target_event, \
                        -100*np.ones((self.chunklen-frame_num,target_event.shape[-1]))), axis=0)
                target_doa = np.concatenate(
                    (target_doa, \
                        -100*np.ones((self.chunklen-frame_num,target_doa.shape[-1]))), axis=0) 
                target_dist = np.concatenate(
                    (target_dist, \
                        -100*np.ones((self.chunklen-frame_num,target_dist.shape[-1]))), axis=0)
                train_index.append(pointer)
            elif frame_num == self.chunklen:
                train_index.append(pointer)
            pointer += frame_num

            self.train_features_list.append(feature)
            self.train_fn_list.append(fn)
            self.train_target_events_list.append(target_event)
            self.train_target_doas_list.append(target_doa)
            self.train_target_dists_list.append(target_dist)
            self.train_segmented_indexes.append(train_index)

        self.train_features = np.concatenate(self.train_features_list, axis=1)
        self.train_target_events = np.concatenate(self.train_target_events_list, axis=0)
        self.train_target_doas = np.concatenate(self.train_target_doas_list, axis=0)
        self.train_target_dists = np.concatenate(self.train_target_dists_list, axis=0)
        self.train_segmented_indexes = np.concatenate(self.train_segmented_indexes, axis=0)

        # Validation data
        self.validation_features_list = []
        self.validation_fn_list = []
        self.validation_target_events_list = []
        self.validation_target_doas_list = []
        self.validation_target_dists_list = []
        for hdf5_fn in self.validation_hdf5_fn:
            
            fn = hdf5_fn.split('.')[0]
            hdf5_path = os.path.join(hdf5_dev_dir, hdf5_fn)
            feature, target_event, target_doa, target_dist = \
                self.load_hdf5(hdf5_path)
            
            self.validation_features_list.append(feature)
            self.validation_fn_list.append(fn)
            self.validation_target_events_list.append(target_event)
            self.validation_target_doas_list.append(target_doa)
            self.validation_target_dists_list.append(target_dist)

        # Scalar
        scalar_path = os.path.join(hdf5_dir + '_scalar.h5')
        with h5py.File(scalar_path, 'r') as hf_scalar:
            self.mean = hf_scalar['mean'][:]
            self.std = hf_scalar['std'][:]

        load_time = timer() - load_begin_time
        print('Loading training data time: {:.3f} s.\n'.format(load_time))
        print('Training audios number: {}\n'.format(len(self.train_segmented_indexes)))
        print('Cross-Validation audios number: {}\n'.format(len(self.validation_fn_list)))

        self._training_batches = int(np.ceil(len(self.train_segmented_indexes)/self.batch_size))
        
    def load_hdf5(self, hdf5_path):
        '''
        Load hdf5. 
        
        Args:
          hdf5_path: string
          
        Returns:
          feature: (channel_num, frame_num, freq_bins)
          target_event: (frame_num, class_num)
          target_doa: (frame_num, 2*class_num) for 'regr' | (frame_num, class_num, ele_num*azi_num=324) for 'clas'
          target_dist: (frame_num, class_num) for 'regr' | (frame_num, class_num, 2) for 'clas'
        '''

        with h5py.File(hdf5_path, 'r') as hf:
            feature = hf['feature'][:]
            event = [e.decode() for e in hf['target']['event'][:]]
            start_time = hf['target']['start_time'][:]
            end_time = hf['target']['end_time'][:]
            elevation = hf['target']['elevation'][:]
            azimuth = hf['target']['azimuth'][:]   
            distance = hf['target']['distance'][:]            
        
        frame_num = feature.shape[1]
        target_event = np.zeros((frame_num, self.class_num))
        target_ele = np.zeros((frame_num, self.class_num))
        target_azi = np.zeros((frame_num, self.class_num))
        target_dist = np.zeros((frame_num, self.class_num))
        
        for n in range(len(event)):
            start_idx = np.int(np.round(start_time[n] * self.fs//self.hopsize)) ##### consider it further about this round!!!
            end_idx = np.int(np.round(end_time[n] * self.fs//self.hopsize))
            class_idx = lb_to_ix[event[n]]
            target_event[start_idx:end_idx, class_idx] = 1.0
            target_ele[start_idx:end_idx, class_idx] = elevation[n] * np.pi / 180.0
            target_azi[start_idx:end_idx, class_idx] = azimuth[n] * np.pi / 180.0
            target_dist[start_idx:end_idx, class_idx] = distance[n] * 1.0

        target_doa = np.concatenate((target_azi, target_ele), axis=-1)

        return feature, target_event, target_doa, target_dist

    def transform(self, x):
        """
        Use the calculated scalar to transform data.
        """

        return (x - self.mean) / self.std


    def on_epoch_end(self):
        random.shuffle(self.train_segmented_indexes)
        
    
    def __data_generation(self, index):
        batch_x = np.zeros((self.batch_size, self.nb_channels, self.chunklen, self.mel_bins))
        batch_y = np.zeros((self.batch_size, self.chunklen, self.class_num))
        batch_start = index * self.batch_size
        for batch_cnt in range(self.batch_size):
            batch_x[batch_cnt, :, :,:] = self.transform(self.train_features[:, self.train_segmented_indexes[batch_start+batch_cnt]:self.train_segmented_indexes[batch_start+batch_cnt]+self.chunklen, :])
            batch_y[batch_cnt, :, :] = self.train_target_events[self.train_segmented_indexes[batch_start+batch_cnt]:self.train_segmented_indexes[batch_start+batch_cnt]+self.chunklen, :]
        return batch_x, batch_y
    
    def __len__(self):
        return self._training_batches

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        return X, y

class DataGenerator_VSED(Sequence):
    def __init__(self, args, valid_features, valid_targets, hdf5_dir):
        # Parameters
        self.valid_features = valid_features
        self.valid_targets = valid_targets
        self.fs = args['fs']
        self.nb_channels = args['nb_channels']
        self.nfft = args['nfft']
        self.hopsize = args['hopsize']
        self.mel_bins = args['mel_bins']
        self.chunklen = args['chunklen']
        self.hopframes = args['hopframes']
        self.batch_size = args['batch_size']
        self._valid_batches = int(len(valid_features)/self.batch_size)
        self.class_num = len(event_labels)
        # Scalar
        scalar_path = os.path.join(hdf5_dir + '_scalar.h5')
        with h5py.File(scalar_path, 'r') as hf_scalar:
            self.mean = hf_scalar['mean'][:]
            self.std = hf_scalar['std'][:]

    def transform(self, x):
        return (x - self.mean) / self.std



    def __data_generation(self, index):
        return self.transform(self.valid_features[index*self.batch_size:(index+1)*self.batch_size, :, :, :]), self.valid_targets[index*self.batch_size:(index+1)*self.batch_size, :, :]
    
    def __len__(self):
        return self._valid_batches

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        return X, y

