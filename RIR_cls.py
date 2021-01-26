import os
import sys
import time
import itertools
import random
import numpy as np
import pyroomacoustics as pra
import librosa
import scipy.io.wavfile as wav
import csv


class roomIR:
    def __init__(self, datadir='', outdir='', room3Ds=[10, 7], Nmics=4, Nsources=5, record_len=60, nb_records=400):
        self._datadir = datadir
        self._outdir = outdir
        self._unique_classess = ['cough', 'knock', 'laughter', 'phone', 'speech']
        self._records_per_class = 20
        """
        self._unique_classes = dict()
        self._unique_classes = \
            {
                'cough': 0,
                'knock': 1,
                'laughter': 2,
                'phone': 3,
                'speech': 4
            }
        """
        self._fs = 44100
        self._rfs = 48000
        self._nb_channels = Nmics
        self._nb_sources = Nsources
        self._record_len = record_len
        self._max_wav_len = 5
        self._max_nb_records = nb_records
        self._max_nb_samples = self._record_len*self._fs
        self._hop_len = 0.02*self._fs
        self._win_len = 2*self._hop_len
        self._room = pra.ShoeBox(room3Ds, fs=self._rfs, max_order=0, sigma2_awgn=10**(-0 / 10) / (4. * np.pi * 3)**2)
        #self._MicArr = np.c_[[4.75, 3.5, 2], [5.25, 3.5, 2], [5, 3.25, 2], [5, 3.75, 2]]
        self._R = pra.circular_2D_array(np.r_[room3Ds] / 2, self._nb_channels, 0., 0.42)
        #self._room.add_microphone_array(pra.MicrophoneArray(self._MicArr, self._room.fs))
        self._room.add_microphone_array(pra.MicrophoneArray(self._R, fs=self._room.fs))
        self._theta = np.array([0, 72, -72, 144, -144])*np.pi/180
        for src_cnt in range(self._nb_sources):
            self._room.add_source([room3Ds[0]/2+3*np.cos(self._theta[src_cnt]), room3Ds[1]/2+3*np.sin(self._theta[src_cnt])])

    def createCombinations(self):
        class_files = []
        for class_cnt in range(self._nb_sources):
            class_files.append(os.listdir(os.path.join(self._datadir, self._unique_classess[class_cnt])))
        full_file_combs = list(itertools.product(*class_files))
        #comb_idxs = np.reshape(np.array(random.sample(range(len(full_file_combs)), self._max_nb_records), dtype=np.int), (self._max_nb_records, 1))
        return random.sample(full_file_combs, self._max_nb_records)


    def prepare_sources(self, file_combs):
        sources = []
        for record_cnt in range(self._max_nb_records):
            per_record_files = file_combs[record_cnt]
            wav_files = []
            wav_classes = []
            delays = []
            x_temp = random.randint(1, 41)/10
            y_temp = np.concatenate(([x_temp], self._max_wav_len+np.array(random.sample(range(10, 51), self._nb_sources-1))/10))
            x_temp = 0
            for wav_cnt in range(self._nb_sources):
                wav_files.append(os.path.join(self._datadir, self._unique_classess[wav_cnt], per_record_files[wav_cnt]))
                wav_classes.append(wav_cnt)
                x_temp = x_temp+y_temp[wav_cnt]
                delays.append(x_temp)
            sources.append({'wav_files': wav_files,
            'wav_classes':wav_classes,
            'delays':delays})
        return sources

    def simulate_room_IR(self):
        file_combs = self.createCombinations()
        sources_list = self.prepare_sources(file_combs)
        for var_cnt, src_s in enumerate(sources_list):
            print('Generating file#{} out of {}'.format(var_cnt+1, len(sources_list)))
            wav_files = src_s['wav_files']
            wav_classes = src_s['wav_classes']
            delays = src_s['delays']
            #idxs = np.array(range(0, self._nb_sources))
            random.shuffle(wav_classes)
            #wav_classes = wav_classes[idxs]
            #wav_files = wav_files[idxs]
            tags_file = []
            for wav_cnt, wav_class in enumerate(wav_classes):
                fs, audio = wav.read(wav_files[wav_class])
                self._room.sources[wav_cnt].add_signal(audio)
                self._room.sources[wav_cnt].delay = delays[wav_cnt]
                tags_file.append({'sound_event_recording':self._unique_classess[wav_class], 'start_time':delays[wav_cnt], 'end_time':delays[wav_cnt]+np.round(audio.shape[0]/fs,2), 'ele':0, 'azi':self._theta[wav_cnt]*180/np.pi, 'dist':3})
            self._room.simulate(reference_mic=0, snr=10)
            self._room.mic_array.to_wav(os.path.join(self._outdir, 'sound', 'split{}_ir0_ov1_{}.wav'.format(int(var_cnt/50), var_cnt%50)), norm=True, bitdepth=np.int16)
            keys = ['sound_event_recording', 'start_time', 'end_time', 'ele', 'azi', 'dist']
            with open(os.path.join(self._outdir, 'label', 'split{}_ir0_ov1_{}.csv'.format(int(var_cnt/50), var_cnt%50)), 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                for data in tags_file:
                    writer.writerow(data)