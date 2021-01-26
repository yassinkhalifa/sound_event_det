#Contains a class with methods for splitting the audio files and generating corresponding labels

import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
from sklearn.externals import joblib
from IPython import embed
import matplotlib.pyplot as plot
import librosa
from sklearn import preprocessing


class DataPreparation:
    def __init__(self, datadir = '', labelsdir = '', outputdir = '', fs = 48000, winlen_s = 0.04, hoplen_s = 0.02, parsing_len_s = 2):
        self._datadir = datadir
        self._labelsdir = labelsdir
        self._outdatadir = outputdir + '/' + 'data'
        self._outlabelsdir = outputdir + '/' + 'labels'
        self._fs = fs
        self._nfft = 2048
        self._winlen_s = winlen_s
        self._winlen = int(winlen_s*fs)
        self._hoplen_s = hoplen_s
        self._hop_len = int(hoplen_s*fs)
        self._parsing_len_s = parsing_len_s
        self._parsing_len = parsing_len_s*fs
        self._eps = np.spacing(np.float(1e-16))
        self._nb_channels = 4
        self._frame_res = self._fs / float(self._hop_len)
        self._nb_frames_1s = int(self._frame_res)
        self._unique_classes = dict()
        self._unique_classes = \
            {
                'cough': 0,
                'knock': 1,
                'laughter': 2,
                'phone': 3,
                'speech': 4
            }

        self._doa_resolution = 10
        self._azi_list = np.array([-144, -72, 0, 72, 144])*np.pi/180
        self._length = len(self._azi_list)

        # For regression task only
        self._default_azi = 2*np.pi

    def _load_audio(self, audio_path):
        fs, audio = wav.read(audio_path)
        audio = audio[:, :self._nb_channels] + self._eps
        if np.remainder(audio.shape[0], self._parsing_len) != 0:
            zero_pad = np.zeros((np.int(np.fix(audio.shape[0]/self._parsing_len)+1)*self._parsing_len - audio.shape[0], audio.shape[1]))
            audio = np.vstack((audio, zero_pad))
        return audio, fs

    def read_desc_file(self, desc_filename, in_sec=False):
        desc_file = {
            'class': list(), 'start': list(), 'end': list(), 'ele': list(), 'azi': list()
        }
        fid = open(desc_filename, 'r')
        next(fid)
        for line in fid:
            split_line = line.strip().split(',')
            desc_file['class'].append(split_line[5])
            # desc_file['class'].append(split_line[0].split('.')[0][:-3])
            if in_sec:
                # return onset-offset time in seconds
                desc_file['start'].append(float(split_line[3]))
                desc_file['end'].append(float(split_line[4]))
            else:
                # return onset-offset time in frames
                desc_file['start'].append(int(np.floor(float(split_line[3])*self._frame_res)))
                desc_file['end'].append(int(np.ceil(float(split_line[4])*self._frame_res)))
            desc_file['ele'].append(float(split_line[1]))
            desc_file['azi'].append(float(split_line[0]))
        fid.close()
        return desc_file

    def get_list_index(self, azi):
        azi = (azi - self._azi_list[0]) // (72*np.pi/180)
        return azi

    def get_matrix_index(self, ind):
        azi = (ind * 72*np.pi/180 + self._azi_list[0])
        return azi

    def _get_doa_labels_regr(self, _desc_file, audio_len):
        max_frames = int(np.ceil(audio_len/ float(self._hop_len)))
        azi_label = self._default_azi*np.ones((max_frames, len(self._unique_classes)))
        for i, ele_ang in enumerate(_desc_file['ele']):
            start_frame = _desc_file['start'][i]
            end_frame = max_frames if _desc_file['end'][i] > max_frames else _desc_file['end'][i]
            azi_ang = _desc_file['azi'][i]
            class_ind = self._unique_classes[_desc_file['class'][i]]
            if (azi_ang >= self._azi_list[0]) & (azi_ang <= self._azi_list[-1]):
                azi_label[start_frame:end_frame + 1, class_ind] = azi_ang
            else:
                print('bad_angle {} {}'.format(azi_ang, ele_ang))
        doa_label_regr = azi_label
        return doa_label_regr

    def _get_se_labels(self, _desc_file, audio_len):
        max_frames = int(np.ceil(audio_len/ float(self._hop_len)))
        se_label = np.zeros((max_frames, len(self._unique_classes)))
        for i, se_class in enumerate(_desc_file['class']):
            start_frame = _desc_file['start'][i]
            end_frame = max_frames if _desc_file['end'][i] > max_frames else _desc_file['end'][i]
            se_label[start_frame:end_frame + 1, self._unique_classes[se_class]] = 1
        return se_label

    def get_labels_for_file(self, _desc_file, audio_len):
        se_label = self._get_se_labels(_desc_file, audio_len)
        doa_label = self._get_doa_labels_regr(_desc_file, audio_len)
        label_mat = np.concatenate((se_label, doa_label), axis=1)
        # print(label_mat.shape)
        return label_mat
    
    def get_clas_labels_for_file(self, _desc_file, audio_len):
        max_frames = int(np.ceil(audio_len/ float(self._hop_len)))
        _labels = np.zeros((max_frames, len(self._unique_classes), len(self._azi_list)))
        for _ind, _start_frame in enumerate(_desc_file['start']):
            _tmp_class = self._unique_classes[_desc_file['class'][_ind]]
            _tmp_azi = _desc_file['azi'][_ind]
            _tmp_end = max_frames if _desc_file['end'][_ind] > max_frames else _desc_file['end'][_ind]
            _tmp_ind = self.get_list_index(_tmp_azi)
            _labels[_start_frame:_tmp_end + 1, _tmp_class, _tmp_ind] = 1
        return _labels

    def _spectrogram(self, audio_input):
        _nb_ch = audio_input.shape[1]
        nb_bins = int(self._nfft/2)
        max_frames = int(np.ceil(audio_input.shape[0]/ float(self._hop_len)))
        spectra = np.zeros((max_frames, nb_bins, _nb_ch), dtype=complex)
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(audio_input[:, ch_cnt], n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._winlen, window='hann')
            spectra[:, :, ch_cnt] = stft_ch[1:, :max_frames].T
        return np.concatenate((np.abs(spectra), np.angle(spectra)), axis=1).reshape(max_frames, -1)

    def extract(self):
        print('Extracting audio and labels files:')
        print('audio_out_dir: {}\nlabels_out_dir: {}'.format(self._outdatadir, self._outlabelsdir))
        create_folder(self._outdatadir)
        create_folder(self._outlabelsdir)
        for file_cnt, file_name in enumerate(os.listdir(self._datadir)):
            print('{}: {}'.format(file_cnt, file_name))
            spec_scalar = preprocessing.StandardScaler()
            csv_filename = '{}.csv'.format(file_name.split('.')[0])
            audio, fs = self._load_audio(os.path.join(self._datadir, file_name))
            spectra = self._spectrogram(audio)
            spectra = spec_scalar.fit_transform(spectra).reshape(spectra.shape[0], int(self._nfft/2), 2*self._nb_channels)
            desc_file = self.read_desc_file(os.path.join(self._labelsdir, csv_filename))
            label_mat = self.get_labels_for_file(desc_file, audio.shape[0])
            np.save(os.path.join(self._outdatadir, '{}.npy'.format(file_name.split('.')[0])), spectra)
            np.save(os.path.join(self._outlabelsdir, '{}.npy'.format(file_name.split('.')[0])), label_mat)

    def get_classes(self):
        return self._unique_classes

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)