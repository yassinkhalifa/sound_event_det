#Contains a class with methods for splitting the audio files and generating corresponding labels

import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
from sklearn.externals import joblib
from IPython import embed
import matplotlib.pyplot as plot
import librosa
import h5py



class DataPreparation:
    def __init__(self, datadir = '', labelsdir = '', outputdir = '', fs = 48000, winlen_s = 0.25, hoplen_s = 0.125):
        self._datadir = datadir
        self._labelsdir = labelsdir
        self._outdatadir = outputdir + '/' + 'data'
        self._outlabelsdir = outputdir + '/' + 'labels'
        self._fs = fs
        self._winlen_s = winlen_s
        self._hoplen_s = hoplen_s
        self._winlen = int(winlen_s*fs)
        self._hoplen = int(hoplen_s*fs)
        self._eps = np.spacing(np.float(1e-16))
        self._nb_channels = 4
        self._frame_res = int(self._fs / float(self._hoplen))
        self._unique_classes = dict()
        self._unique_classes = \
            {
                'clearthroat': 2,
                'cough': 8,
                'doorslam': 9,
                'drawer': 1,
                'keyboard': 6,
                'keysDrop': 4,
                'knock': 0,
                'laughter': 10,
                'pageturn': 7,
                'phone': 3,
                'speech': 5
            }

        self._doa_resolution = 10
        self._azi_list = range(-180, 180, self._doa_resolution)
        self._length = len(self._azi_list)
        self._ele_list = range(-40, 50, self._doa_resolution)
        self._height = len(self._ele_list)

        self._audio_max_len_samples = 60 * self._fs 
        # For regression task only
        self._default_azi = 180
        self._default_ele = 50

        if self._default_azi in self._azi_list:
            print('ERROR: chosen default_azi value {} should not exist in azi_list'.format(self._default_azi))
            exit()
        if self._default_ele in self._ele_list:
            print('ERROR: chosen default_ele value {} should not exist in ele_list'.format(self._default_ele))
            exit()

        self._max_frames = int((self._audio_max_len_samples) / self._hoplen)

    def _load_audio(self, audio_path):
        fs, audio = wav.read(audio_path)
        audio = audio[:, :self._nb_channels] / 32768.0 + self._eps
        if audio.shape[0] < self._audio_max_len_samples:
            zero_pad = np.zeros((self._audio_max_len_samples - audio.shape[0], audio.shape[1]))
            audio = np.vstack((audio, zero_pad))
        elif audio.shape[0] > self._audio_max_len_samples:
            audio = audio[:self._audio_max_len_samples, :]
        return audio, fs
    
    def _extract_frames(self, audio_filename):
        audio_in, fs = self._load_audio(os.path.join(self._datadir, audio_filename))
        audio_frames = np.zeros((self._max_frames, self._winlen, self._nb_channels))
        frame_order = np.zeros((self._max_frames, 1))
        frame_file = []
        for fcount in range(0, self._max_frames):
            start_idx = fcount*self._hoplen
            end_idx = start_idx + self._winlen
            if end_idx > self._audio_max_len_samples:
                end_idx = self._audio_max_len_samples
                audio_frames[fcount, :(end_idx-start_idx), :] = audio_in[start_idx:end_idx, :]
            else:
                audio_frames[fcount, :, :] = audio_in[start_idx:end_idx, :]
            frame_order[fcount,0] = fcount
            frame_file.append(audio_filename)
        return audio_frames, frame_order, frame_file
        #np.save(os.path.join(self._outdatadir, '{}.npy'.format(audio_filename.split('.')[0])), audio_frames)

    
    # OUTPUT LABELS
    def read_desc_file(self, desc_filename, in_sec=False):
        desc_file = {
            'class': list(), 'start': list(), 'end': list(), 'ele': list(), 'azi': list()
        }
        fid = open(desc_filename, 'r')
        next(fid)
        for line in fid:
            split_line = line.strip().split(',')
            desc_file['class'].append(split_line[0])
            # desc_file['class'].append(split_line[0].split('.')[0][:-3])
            if in_sec:
                # return onset-offset time in seconds
                desc_file['start'].append(float(split_line[1]))
                desc_file['end'].append(float(split_line[2]))
            else:
                # return onset-offset time in frames
                desc_file['start'].append(int(np.floor(float(split_line[1])*self._frame_res)))
                desc_file['end'].append(int(np.ceil(float(split_line[2])*self._frame_res)))
            desc_file['ele'].append(int(split_line[3]))
            desc_file['azi'].append(int(split_line[4]))
        fid.close()
        return desc_file

    def _get_doa_labels_regr(self, _desc_file):
        azi_label = self._default_azi*np.ones((self._max_frames, len(self._unique_classes)))
        ele_label = self._default_ele*np.ones((self._max_frames, len(self._unique_classes)))
        for i, ele_ang in enumerate(_desc_file['ele']):
            start_frame = _desc_file['start'][i]
            end_frame = self._max_frames if _desc_file['end'][i] > self._max_frames else _desc_file['end'][i]
            azi_ang = _desc_file['azi'][i]
            class_ind = self._unique_classes[_desc_file['class'][i]]
            if (azi_ang >= self._azi_list[0]) & (azi_ang <= self._azi_list[-1]) & \
                    (ele_ang >= self._ele_list[0]) & (ele_ang <= self._ele_list[-1]):
                azi_label[start_frame:end_frame + 1, class_ind] = azi_ang
                ele_label[start_frame:end_frame + 1, class_ind] = ele_ang
            else:
                print('bad_angle {} {}'.format(azi_ang, ele_ang))
        doa_label_regr = np.concatenate((azi_label, ele_label), axis=1)
        return doa_label_regr

    def _get_se_labels(self, _desc_file):
        se_label = np.zeros((self._max_frames, len(self._unique_classes)))
        for i, se_class in enumerate(_desc_file['class']):
            start_frame = _desc_file['start'][i]
            end_frame = self._max_frames if _desc_file['end'][i] > self._max_frames else _desc_file['end'][i]
            se_label[start_frame:end_frame + 1, self._unique_classes[se_class]] = 1
        return se_label

    def get_labels_for_file(self, _desc_file):
        """
        Reads description csv file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: csv file
        :return: label_mat: labels of the format [sed_label, doa_label],
        where sed_label is of dimension [nb_frames, nb_classes] which is 1 for active sound event else zero
        where doa_labels is of dimension [nb_frames, 2*nb_classes], nb_classes each for azimuth and elevation angles,
        if active, the DOA values will be in degrees, else, it will contain default doa values given by
        self._default_ele and self._default_azi
        """

        se_label = self._get_se_labels(_desc_file)
        doa_label = self._get_doa_labels_regr(_desc_file)
        label_mat = np.concatenate((se_label, doa_label), axis=1)
        # print(label_mat.shape)
        return label_mat

        
    # ------------------------------- PERFORM THE SPLITTING AND GENERATE LABELS -------------------------------
    def split_data(self):
        # setting up folders
        create_folder(self._outdatadir)

        # extraction starts
        print('Splitting Audio:')
        print('\t\tfrom data_dir: {}\n\t\tinto out_dir: {}'.format(
            self._datadir, self._outdatadir))
        frame_data = np.zeros((len(os.listdir(self._datadir))*self._max_frames, self._winlen, self._nb_channels))
        frame_label = np.zeros((len(os.listdir(self._datadir))*self._max_frames, 3*len(self._unique_classes)))
        frame_fileorder = np.zeros((len(os.listdir(self._datadir))*self._max_frames, 1))
        frame_file = []
        for file_cnt, file_name in enumerate(os.listdir(self._datadir)):
            print('{}: {}'.format(file_cnt, file_name))
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            start_idx = file_cnt*self._max_frames
            end_idx = (file_cnt+1)*self._max_frames
            frame_data[start_idx:end_idx, :, :], frame_fileorder[start_idx:end_idx, :], tmp_files = self._extract_frames(wav_filename)
            desc_filename = '{}.csv'.format(file_name.split('.')[0])
            desc_file = self.read_desc_file(os.path.join(self._labelsdir, desc_filename))
            frame_label[start_idx:end_idx, :] = self.get_labels_for_file(desc_file)
            frame_file = np.concatenate((frame_file, tmp_files))
        print('Writing the h5 file containing the dataset (dcase19_dataset.h5) into {}'.format(self._outdatadir))
        h5file = h5py.File(os.path.join(self._outdatadir, 'dcase19_dataset.h5'), 'w')
        h5file.create_dataset('frame_data', data=frame_data)
        h5file.create_dataset('frame_label', data=frame_label)
        h5file.create_dataset('frame_fileorder', data=frame_fileorder)
        h5file.close()
        np.save(os.path.join(self._outdatadir, 'frame_files.npy'), frame_file)

    # ------------------------------- Misc public functions -------------------------------
    def get_classes(self):
        return self._unique_classes

    def get_default_azi_ele_regr(self):
        return self._default_azi, self._default_ele
    
    def get_frame_res(self):
        return self._frame_res
    
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)