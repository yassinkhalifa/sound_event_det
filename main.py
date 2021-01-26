import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
import numpy as np
import matplotlib.pyplot as plot
from keras.models import load_model
from SimSELDNet.keras_model import get_model
import time
from sacred import Experiment
from keras.layers import Bidirectional, Conv2D, DepthwiseConv2D, MaxPooling2D, Input
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import keras
from IPython import embed
from SimSELDNet.helper_classes import DataGenerator
from SimSELDNet import evaluation_metrics
from SimSELDNet import data_preparation
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



ex = Experiment('SELDnet A raw')

@ex.config
def cfg():
    params = {
        'dataset':'foa',
        'mode':'dev',
        'log_dir':'./logs',
        'res_dir':'./results',
        'sequence_len':100,
        'fs':48000,
        'win_len':1920,
        'hop_len':960,
        'parsing_len':96000,
        'fft_len':2048,
        'max_frames':3000,
        'freq_bins':1024,
        'nb_channels':4,
        'nb_classes':5,
        #'train_splits':[[3, 4], [4, 1], [1, 2], [2, 3]],
        #'val_splits':[[2], [3], [4], [1]],
        #'test_splits':[[1], [2], [3], [4]],
        'train_splits':[[0], [1]],
        'val_splits':[[1], [0]],
        'test_splits':[[1], [0]],
        'loss_weights':[1., 50.],                                 # [sed, doa] weight for scaling the DNN outputs
        'nb_epochs':100,
        'batch_size':10,
        'patience':2
    }
    network_params = {
        'kernel_size':(1, 15),
        'strides':(1,1),
        'pool_size':[8, 8, 4],
        'nb_cnn2d_filt':64,
        'depth_mul':[4, 4, 1],
        'dropout_rate':0.5,
        'rnn_size':[128, 128],
        'fcn_size':[128]
    }


@ex.automain
def main(params, network_params):
    datadir= '../features/data'
    labelsdir= '../features/labels'
    datadir_= '../redataset/outdata/sound'
    labelsdir_= '../redataset/outdata/label'
    feat_class = data_preparation.DataPreparation(datadir = datadir_, labelsdir = labelsdir_, outputdir = '../features')
    avg_scores_val = []
    avg_scores_test = []
    data_preparation.create_folder(params['log_dir'])

    for split_cnt, split in enumerate(params['test_splits']):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        print('---------------------------------------------------------------------------------------------------')
        
        unique_name = '{}_{}_split{}'.format(params['dataset'], params['mode'], split)
        model_name = os.path.join(params['log_dir'], '{}_model.h5'.format(unique_name))

        
        # Load train and validation data
        print('Loading training dataset:')
        train_data_gen = DataGenerator(datadir=datadir, 
            labelsdir=labelsdir, 
            shuffle=True, perfile=False, train=True, 
            splits=params['train_splits'][split_cnt], 
            win_len = params['win_len'], 
            hop_len = int(params['win_len']/2),
            parsing_len = params['parsing_len'],
            nfft = 2*params['freq_bins'],  
            nb_channels = params['nb_channels'], 
            nb_classes = params['nb_classes'], 
            batch_size = params['batch_size'])
        
        print('Loading validation dataset:')
        valid_data_gen = DataGenerator(datadir=datadir, 
            labelsdir=labelsdir,
            shuffle=False, perfile=False, train=False, 
            splits=params['val_splits'][split_cnt], 
            win_len = params['win_len'],
            hop_len = int(params['win_len']/2),
            parsing_len = params['parsing_len'],
            nfft = 2*params['freq_bins'],
            nb_channels = params['nb_channels'], 
            nb_classes = params['nb_classes'], 
            batch_size = params['batch_size'])
        val_gt_sed, val_gt_doa = valid_data_gen.get_val_labels()
        val_gt_sed = evaluation_metrics.reshape_3Dto2D(val_gt_sed)
        val_gt_doa = evaluation_metrics.reshape_3Dto2D(val_gt_doa)
        model = get_model(params, network_params)
        
        best_seld_metric = 99999
        best_epoch = -1
        patience_cnt = 0
        seld_metric = np.zeros(params['nb_epochs'])
        tr_loss = np.zeros(params['nb_epochs'])
        val_loss = np.zeros(params['nb_epochs'])
        doa_metric = np.zeros((params['nb_epochs'], 6))
        sed_metric = np.zeros((params['nb_epochs'], 2))

        for epoch_cnt in range(params['nb_epochs']):
            start_time = time.time()
            print('Epoch No. {}'.format(epoch_cnt+1))
            #hist = model.fit_generator(generator=train_data_gen, validation_data=valid_data_gen, epochs=1)
            hist = model.fit_generator(generator=train_data_gen,
                validation_data=valid_data_gen,
                epochs=5, workers=2, use_multiprocessing=False)

            tr_loss[epoch_cnt] = hist.history.get('loss')[-1]
            val_loss[epoch_cnt] = hist.history.get('val_loss')[-1]

            pred = model.predict_generator(generator=valid_data_gen, workers=2, use_multiprocessing=False)

            # Calculate the metrics
            sed_pred = evaluation_metrics.reshape_3Dto2D(pred[0]) > 0.5
            doa_pred = evaluation_metrics.reshape_3Dto2D(pred[1])

            sed_metric[epoch_cnt, :] = evaluation_metrics.compute_sed_scores(sed_pred, val_gt_sed, int(params['fs'] / float(params['hop_len'])))
            doa_metric[epoch_cnt, :] = evaluation_metrics.compute_doa_scores_regr(doa_pred, val_gt_doa, sed_pred, val_gt_sed)
            seld_metric[epoch_cnt] = evaluation_metrics.compute_seld_metric(sed_metric[epoch_cnt, :], doa_metric[epoch_cnt, :])
            
            patience_cnt += 1
            if seld_metric[epoch_cnt] < best_seld_metric:
                best_seld_metric = seld_metric[epoch_cnt]
                best_epoch = epoch_cnt
                model.save(model_name)
                patience_cnt = 0

            print(
                'epoch_cnt: %d, time: %.2fs, tr_loss: %.2f, val_loss: %.2f, '
                'ER_overall: %.2f, F1_overall: %.2f, '
                'doa_error_pred: %.2f, good_pks_ratio:%.2f, '
                'seld_score: %.2f, best_seld_score: %.2f, best_epoch : %d\n' %
                (
                    epoch_cnt, time.time() - start_time, tr_loss[epoch_cnt], val_loss[epoch_cnt],
                    sed_metric[epoch_cnt, 0], sed_metric[epoch_cnt, 1],
                    doa_metric[epoch_cnt, 0], doa_metric[epoch_cnt, 1],
                    seld_metric[epoch_cnt], best_seld_metric, best_epoch
                )
            )

            if patience_cnt > params['patience']:
                break

        avg_scores_val.append([sed_metric[best_epoch, 0], sed_metric[best_epoch, 1], doa_metric[best_epoch, 0],
                               doa_metric[best_epoch, 1], best_seld_metric])
        print('\nResults on validation split:')
        print('\tUnique_name: {} '.format(unique_name))
        print('\tSaved model for the best_epoch: {}'.format(best_epoch))
        print('\tSELD_score: {}'.format(best_seld_metric))
        print('\tDOA Metrics: DOA_error: {}, frame_recall: {}'.format(doa_metric[best_epoch, 0],
                                                                      doa_metric[best_epoch, 1]))
        print('\tSED Metrics: ER_overall: {}, F1_overall: {}\n'.format(sed_metric[best_epoch, 0],
                                                                       sed_metric[best_epoch, 1]))
        
        # ------------------  Calculate metric scores for unseen test split ---------------------------------
        print('Loading testing dataset:')
        test_data_gen = DataGenerator(datadir=datadir, 
            labelsdir=labelsdir, 
            shuffle=False, perfile=True, train=False, 
            splits=params['test_splits'][split_cnt], 
            win_len = params['win_len'], 
            hop_len = int(params['win_len']/2), 
            parsing_len = params['parsing_len'],
            nfft = 2*params['freq_bins'],
            nb_channels = params['nb_channels'], 
            nb_classes = params['nb_classes'], 
            batch_size = params['batch_size'])

        print('\nLoading the best model and predicting results on the testing split')
        model = load_model(model_name)
        pred_test = model.predict_generator(generator=test_data_gen, workers=2, use_multiprocessing=False)

        test_sed_pred = evaluation_metrics.reshape_3Dto2D(pred_test[0]) > 0.5
        test_doa_pred = evaluation_metrics.reshape_3Dto2D(pred_test[1])

        dcase_dump_folder = os.path.join(params['res_dir'], '{}_{}_split{}'.format(params['dataset'], params['mode'], params['test_splits'][split_cnt]))
        data_preparation.create_folder(params['res_dir'])
        data_preparation.create_folder(dcase_dump_folder)
        test_filelist = test_data_gen._current_files_list
        dc=0
        for file_cnt, test_file in enumerate(test_filelist):
            output_file = os.path.join(dcase_dump_folder, test_file.replace('.npy', '.csv'))
            temp_labels = np.load(os.path.join(labelsdir, test_file))
            temp_max_frames = temp_labels.shape[0]
            output_dict = evaluation_metrics.regression_label_format_to_output_format(
                    feat_class,
                    test_sed_pred[dc:dc + temp_max_frames, :],
                    test_doa_pred[dc:dc + temp_max_frames, :] * 180 / np.pi
            )
            dc = dc + temp_max_frames
            evaluation_metrics.write_output_format_file(output_file, output_dict)

        
        test_gt_sed, test_gt_doa = test_data_gen.get_val_labels()
        test_gt_sed = evaluation_metrics.reshape_3Dto2D(test_gt_sed)
        test_gt_doa = evaluation_metrics.reshape_3Dto2D(test_gt_doa)
        
        test_sed_loss = evaluation_metrics.compute_sed_scores(test_sed_pred, test_gt_sed, int(params['fs'] / float(params['hop_len'])))
        test_doa_loss = evaluation_metrics.compute_doa_scores_regr(test_doa_pred, test_gt_doa, test_sed_pred, test_gt_sed)
        test_metric_loss = evaluation_metrics.compute_seld_metric(test_sed_loss, test_doa_loss)

        avg_scores_test.append([test_sed_loss[0], test_sed_loss[1], test_doa_loss[0], test_doa_loss[1], test_metric_loss])
        print('Results on test split:')
        print('\tSELD_score: {},  '.format(test_metric_loss))
        print('\tDOA Metrics: DOA_error: {}, frame_recall: {}'.format(test_doa_loss[0], test_doa_loss[1]))
        print('\tSED Metrics: ER_overall: {}, F1_overall: {}\n'.format(test_sed_loss[0], test_sed_loss[1]))

    print('\n\nValidation split scores per fold:\n')
    for cnt in range(len(params['test_splits'])):
        print('\tSplit {} - SED ER: {} F1: {}; DOA error: {} frame recall: {}; SELD score: {}'.format(cnt, avg_scores_val[cnt][0], avg_scores_val[cnt][1], avg_scores_val[cnt][2], avg_scores_val[cnt][3], avg_scores_val[cnt][4]))

    print('\n\nTesting split scores per fold:\n')
    for cnt in range(len(params['test_splits'])):
        print('\tSplit {} - SED ER: {} F1: {}; DOA error: {} frame recall: {}; SELD score: {}'.format(cnt, avg_scores_test[cnt][0], avg_scores_test[cnt][1], avg_scores_test[cnt][2], avg_scores_test[cnt][3], avg_scores_test[cnt][4]))