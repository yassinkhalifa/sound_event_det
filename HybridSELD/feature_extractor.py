import argparse
import os
import pdb
import sys
from timeit import default_timer as timer

import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
from tqdm import tqdm

from utilities import (calculate_scalar, doa_labels, doa_to_ix, event_labels, lb_to_ix,
                             test_split_dict, train_splits_dict,
                             validation_split_dict)


fs = 48000
nfft = 1024
hopsize = 320
mel_bins = 128
window = 'hann'
fmin = 50
hdf5_folder_name = 'features'


class LogMelExtractor():
    def __init__(self, fs, nfft, hopsize, mel_bins, window, fmin):

        self.nfft = nfft
        self.hopsize = hopsize
        self.window = window
        self.melW = librosa.filters.mel(sr=fs,
                                        n_fft=nfft,
                                        n_mels=mel_bins,
                                        fmin=fmin)

    def transform(self, audio):

        channel_num = audio.shape[0]
        feature_logmel = []

        for n in range(channel_num):
            S = np.abs(librosa.stft(y=audio[n],
                                    n_fft=self.nfft,
                                    hop_length=self.hopsize,
                                    center=True,
                                    window=self.window,
                                    pad_mode='reflect'))**2
            S_mel = np.dot(self.melW, S).T
            S_logmel = librosa.power_to_db(S_mel, ref=1.0, amin=1e-10, top_db=None)
            S_logmel = np.expand_dims(S_logmel, axis=0)
            feature_logmel.append(S_logmel)

        feature_logmel = np.concatenate(feature_logmel, axis=0)

        return feature_logmel

def RT_preprocessing(audio, feature_type):
    if feature_type == 'logmel':
        extractor = LogMelExtractor(fs=fs, 
                                    nfft=nfft,
                                    hopsize=hopsize,
                                    mel_bins=mel_bins,
                                    window=window,
                                    fmin=fmin)
    feature = extractor.transform(audio)
    '''(channels, seq_len, mel_bins)'''
    return feature

def extract_features(args):
    """
    Write features and infos of audios to hdf5.

    Args:
        dataset_dir: dataset path
        feature_dir: feature path
        data_type: 'dev' | 'eval'
        audio_type: 'foa' | 'mic'
    """

    # Path
    audio_dir = os.path.join(args.dataset_dir, args.audio_type + '_' + args.data_type)
    meta_dir = os.path.join(args.dataset_dir, 'metadata_' + args.data_type)

    if args.data_type == 'dev':
        hdf5_dir = os.path.join(args.feature_dir, args.feature_type,
                                hdf5_folder_name, args.audio_type + '_dev/')
    else:
        raise Exception('Wrong data type input.')

    os.makedirs(os.path.dirname(hdf5_dir), exist_ok=True)

    begin_time = timer()
    audio_count = 0

    print('\n============> Start Extracting Features\n')
    
    iterator = tqdm(sorted(os.listdir(audio_dir)), total=len(os.listdir(audio_dir)), unit='it')

    for audio_fn in iterator:

        if audio_fn.endswith('.wav') and not audio_fn.startswith('.'):

            fn = audio_fn.split('.')[0]
            audio_path = os.path.join(audio_dir, audio_fn)

            audio, _ = librosa.load(audio_path, sr=fs, mono=False, dtype=np.float32)
            '''(channel_nums, samples)'''
            audio_count += 1

            if np.sum(np.abs(audio)) < len(audio)*1e-4:
                with open("feature_removed.txt", "a+") as text_file:
                    # print("Purchase Amount: {}".format(TotalAmount), file=text_file)
                    print(f"Silent file removed in feature extractor: {audio_fn}", 
                        file=text_file)
                    tqdm.write("Silent file removed in feature extractor: {}".format(audio_fn))
                continue

            # features
            feature = RT_preprocessing(audio, args.feature_type)
            '''(channels, time, frequency)'''               

            if args.data_type == 'dev':

                meta_fn = fn + '.csv'
                df = pd.read_csv(os.path.join(meta_dir, meta_fn))

                target_event = df['sound_event_recording'].values
                target_start_time = df['start_time'].values
                target_end_time = df['end_time'].values
                target_ele = df['ele'].values
                target_azi = df['azi'].values
                target_dist = df['dist'].values

            elif args.data_type == 'eval':
                raise Exception('Leave for further editing')

            hdf5_path = os.path.join(hdf5_dir, fn + '.h5')
            with h5py.File(hdf5_path, 'w') as hf:

                hf.create_dataset('feature', data=feature, dtype=np.float32)
                # hf.create_dataset('filename', data=[na.encode() for na in [fn]], dtype='S20')

                if args.data_type == 'dev':
                    
                    hf.create_group('target')
                    hf['target'].create_dataset('event', data=[e.encode() for e in target_event], dtype='S20')
                    hf['target'].create_dataset('start_time', data=target_start_time, dtype=np.float32)
                    hf['target'].create_dataset('end_time', data=target_end_time, dtype=np.float32)
                    hf['target'].create_dataset('elevation', data=target_ele, dtype=np.float32)
                    hf['target'].create_dataset('azimuth', data=target_azi, dtype=np.float32)
                    hf['target'].create_dataset('distance', data=target_dist, dtype=np.float32)    

            tqdm.write('{}, {}, {}'.format(audio_count, hdf5_path, feature.shape))
    
    iterator.close()
    print("Extacting feature finished! Time spent: {:.3f} s".format(timer() - begin_time))


def fit(args):
    """
    Calculate scalar.

    Args:
        feature_dir: feature path
        data_type: 'dev' | 'eval'
        audio_type: 'foa' | 'mic'
    """
    
    if args.data_type == 'dev':
        hdf5_dir = os.path.join(args.feature_dir, args.feature_type,
                                hdf5_folder_name, args.audio_type + '_dev/')
    scalar_path = os.path.join(args.feature_dir, args.feature_type,
                            hdf5_folder_name, args.audio_type + '_scalar.h5')

    os.makedirs(os.path.dirname(scalar_path), exist_ok=True)

    print('\n============> Start Calculating Scalar.\n')

    load_time = timer()
    features = []
    for hdf5_fn in os.listdir(hdf5_dir):
        hdf5_path = os.path.join(hdf5_dir, hdf5_fn)
        with h5py.File(hdf5_path, 'r') as hf:
            features.append(hf['feature'][:])
    print('Load feature time: {:.3f} s'.format(timer() - load_time))

    features = np.concatenate(features, axis=1)
    (mean, std) = calculate_scalar(features)

    with h5py.File(scalar_path, 'w') as hf_scalar:
        hf_scalar.create_dataset('mean', data=mean, dtype=np.float32)
        hf_scalar.create_dataset('std', data=std, dtype=np.float32)

    print('Features shape: {}'.format(features.shape))
    print('mean {}:\n{}'.format(mean.shape, mean))
    print('std {}:\n{}'.format(std.shape, std))
    print('Write out scalar to {}'.format(scalar_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from audio files')

    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--feature_type', type=str, required=True,
                                choices=['logmel', 'spec'])   
    parser.add_argument('--data_type', type=str, required=True, 
                                choices=['dev'])
    parser.add_argument('--audio_type', type=str, required=True,
                                choices=['foa', 'mic'])
                                

    args = parser.parse_args()

    extract_features(args)
    fit(args)
    