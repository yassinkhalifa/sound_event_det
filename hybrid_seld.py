import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import numpy as np
import matplotlib.pyplot as plot
from keras.models import load_model
import time
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from IPython import embed
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from HybridSELD.helper_classes import DataGenerator_SED
from HybridSELD.helper_classes import DataGenerator_VSED
from HybridSELD.sed_model import get_model
from HybridSELD import utilities
from HybridSELD import evaluation_metrics


ex = Experiment('Hybrid SEDNet with LogMel Features')
#ex.observers.append(MongoObserver.create(url='192.168.1.102:27017',db_name='SELDNet'))
#ex.captured_out_filter = apply_backspaces_and_linefeeds

################# param #################
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
hdf5_folder_name = '/data/datasets/dcase_2019/task_3/logmel_features/logmel/features/foa'

@ex.config
def cfg():
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

    

@ex.automain
def main(params):
    for split_cnt in range(params['splits']):
        dataGen = DataGenerator_SED(params, hdf5_folder_name, split_cnt+1)
        valid_features, valid_target_events, valid_target_doas, valid_target_dists = utilities.reshape_features(dataGen.validation_features_list, dataGen.validation_target_events_list, dataGen.validation_target_doas_list, dataGen.validation_target_dists_list, chunklen)
        valid_dataGen = DataGenerator_VSED(params, valid_features, valid_target_events, hdf5_folder_name)
        best_model_name = os.path.join(params['log_dir'], 'SED_best_model_split_{}.h5'.format(split_cnt))

        model = get_model(params)
        
        best_ER = 99999
        best_epoch = -1
        patience_cnt = 0
        Error_Rate = np.zeros(params['nb_epochs'])
        tr_loss = np.zeros(params['nb_epochs'])
        val_loss = np.zeros(params['nb_epochs'])
        sed_metric = np.zeros((params['nb_epochs'], 2))
        for epoch_cnt in range(params['nb_epochs']):
            start_time = time.time()
            hist = model.fit_generator(generator=dataGen, validation_data=valid_dataGen, epochs=1, workers=16, use_multiprocessing=True)
            tr_loss[epoch_cnt] = hist.history.get('loss')[-1]
            val_loss[epoch_cnt] = hist.history.get('val_loss')[-1]
            
            valid_pred = model.predict_generator(generator=valid_dataGen, workers=16, use_multiprocessing=True)
            valid_pred = evaluation_metrics.reshape_3Dto2D(valid_pred > 0.5)
            sed_metric[epoch_cnt, :] = evaluation_metrics.compute_sed_scores(valid_pred, evaluation_metrics.reshape_3Dto2D(valid_target_events), frames_per_1s)
            patience_cnt += 1
            if sed_metric[epoch_cnt,0] < best_ER:
                best_ER = sed_metric[epoch_cnt, 0]
                best_epoch = epoch_cnt
                model.save(best_model_name)
                patience_cnt = 0
            print(
                'epoch_cnt: %d, time: %.2fs, tr_loss: %.2f, val_loss: %.2f, '
                'ER_overall: %.2f, F1_overall: %.2f' %
                (
                    epoch_cnt, time.time() - start_time, tr_loss[epoch_cnt], val_loss[epoch_cnt],
                    sed_metric[epoch_cnt, 0], sed_metric[epoch_cnt, 1]
                )
            )