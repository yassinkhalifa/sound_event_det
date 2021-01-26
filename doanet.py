import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
import numpy as np
import matplotlib.pyplot as plot
from keras.models import load_model
import time
from sacred import Experiment
from sacred.observers import SqlObserver
from keras.layers import Bidirectional, Conv2D, DepthwiseConv2D, MaxPooling2D, Input
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import keras
from IPython import embed
from DOANet.keras_model import get_model
from DOANet.helper_classes import DataGenerator
from OrigSELDNet import evaluation_metrics
from OrigSELDNet import data_preparation
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)


ex = Experiment('DOANet Spectrogram')

@ex.config
def cfg():
    params = {
        'dataset':'foa',
        'mode':'dev',
        'log_dir':'./doanet_logs',
        'res_dir':'./doanet_results',
        'sequence_len':100,
        'win_len':1920,
        'max_frames':1500,
        'nb_freq_bins':1024,
        'nb_channels':4,
        'nb_classes':11,
        'train_splits':[[3, 4], [4, 1], [1, 2], [2, 3]],
        'val_splits':[[2], [3], [4], [1]],
        'test_splits':[[1], [2], [3], [4]],
        'nb_epochs':100,
        'batch_size':15,
        'patience':10
    }
    network_params = {
        'specCNN_filt':[64, 64, 64, 64],
        'pool_size_cnn':[8, 8, 4, 2],
        'dropout_rate':0,
        'specRNN_size':[64, 64],
        'specfcn_size':614,
        'spspecCNN_filt':[16, 16],
        'pool_size_spcnn':2,
        'spspecfcn_size':[32, 324],
        'spspecRNN_size':[16, 16]
    }


@ex.automain
def main(params, network_params):
    dataset_name = 'foa'
    datadir = '../dataset'
    featlabelsdir = '../dataset/spec_feat'

    prepare_class = data_preparation.DataPreparation(dataset_dir=datadir, feat_label_dir=featlabelsdir, dataset=dataset_name, is_eval=False)
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
        train_data_gen = DataGenerator(datadir='../dataset/spec_feat/foa_dev_norm', 
            labelsdir='../dataset/spec_feat/foa_dev_label', 
            shuffle=True, perfile=False, train=True, 
            splits=params['train_splits'][split_cnt], 
            win_len = params['win_len'], 
            nb_channels = params['nb_channels'], 
            nb_classes = params['nb_classes'], 
            batch_size = params['batch_size'], 
            seq_len = params['sequence_len'])
        
        print('Loading validation dataset:')
        valid_data_gen = DataGenerator(datadir='../dataset/spec_feat/foa_dev_norm', 
            labelsdir='../dataset/spec_feat/foa_dev_label', 
            shuffle=False, perfile=False, train=False, 
            splits=params['val_splits'][split_cnt], 
            win_len = params['win_len'], 
            nb_channels = params['nb_channels'], 
            nb_classes = params['nb_classes'], 
            batch_size = params['batch_size'], 
            seq_len = params['sequence_len'])
        val_gt_sed, val_gt_doa = valid_data_gen.get_val_labels()
        val_gt_sed = evaluation_metrics.reshape_3Dto2D(val_gt_sed)
        val_gt_doa = evaluation_metrics.reshape_3Dto2D(val_gt_doa)

        # rescaling the reference elevation data from [-180 180] to [-def_elevation def_elevation] for scoring purpose
        # the doa labels are coming from dataGen class in radians and the conversion here is performed to change the 
        # scale and keep them in radians too

        nb_classes = params['nb_classes']
        def_elevation = train_data_gen._default_ele
        val_gt_doa[:, nb_classes:] = val_gt_doa[:, nb_classes:] / (180. / def_elevation)

        model = get_model(params, network_params)
        
        best_seld_metric = 99999
        best_epoch = -1
        patience_cnt = 0
        tr_loss = np.zeros(params['nb_epochs'])
        val_loss = np.zeros(params['nb_epochs'])
        doa_metric = np.zeros((params['nb_epochs'], 6))

        for epoch_cnt in range(params['nb_epochs']):
            start_time = time.time()
            print('Epoch No. {}'.format(epoch_cnt+1))
            hist = model.fit_generator(generator=train_data_gen,
                validation_data=valid_data_gen,
                epochs=5, workers=2, use_multiprocessing=False)

            tr_loss[epoch_cnt] = hist.history.get('loss')[-1]
            val_loss[epoch_cnt] = hist.history.get('val_loss')[-1]

            pred = model.predict_generator(generator=valid_data_gen, workers=2, use_multiprocessing=False)

            # Calculate the metrics
            doa_pred = evaluation_metrics.reshape_3Dto2D(pred)

            # rescaling the elevation data from [-180 180] to [-def_elevation def_elevation] for scoring purpose
            # the same here, all in radians

            doa_pred[:, params['nb_classes']:] = doa_pred[:, params['nb_classes']:] / (180. / valid_data_gen._default_ele)

            doa_metric[epoch_cnt, :] = evaluation_metrics.compute_doa_scores_regr(doa_pred, val_gt_doa, val_gt_sed, val_gt_sed)
            
            patience_cnt += 1
            if doa_metric[epoch_cnt, 0] < best_seld_metric:
                best_seld_metric = doa_metric[epoch_cnt, 0]
                best_epoch = epoch_cnt
                model.save(model_name)
                patience_cnt = 0

            print(
                'epoch_cnt: %d, time: %.2fs, tr_loss: %.2f, val_loss: %.2f, '
                'doa_error_pred: %.2f, good_pks_ratio:%.2f, best_epoch : %d\n' %
                (
                    epoch_cnt, time.time() - start_time, tr_loss[epoch_cnt], val_loss[epoch_cnt],
                    doa_metric[epoch_cnt, 0], doa_metric[epoch_cnt, 1], best_epoch
                )
            )

            if patience_cnt > params['patience']:
                break

        avg_scores_val.append([doa_metric[best_epoch, 0],
                               doa_metric[best_epoch, 1], best_seld_metric])
        print('\nResults on validation split:')
        print('\tUnique_name: {} '.format(unique_name))
        print('\tSaved model for the best_epoch: {}'.format(best_epoch))
        print('\tDAO_error: {}'.format(best_seld_metric))
        print('\tDOA Metrics: DOA_error: {}, frame_recall: {}'.format(doa_metric[best_epoch, 0],
                                                                      doa_metric[best_epoch, 1]))
        
        # ------------------  Calculate metric scores for unseen test split ---------------------------------
        print('Loading testing dataset:')
        test_data_gen = DataGenerator(datadir='../dataset/spec_feat/foa_dev_norm', 
            labelsdir='../dataset/spec_feat/foa_dev_label',
            shuffle=False, perfile=True, train=False, 
            splits=params['test_splits'][split_cnt], 
            win_len = params['win_len'], 
            nb_channels = params['nb_channels'], 
            nb_classes = params['nb_classes'], 
            batch_size = params['batch_size'], 
            seq_len = params['sequence_len'])

        print('\nLoading the best model and predicting results on the testing split')
        model = load_model(model_name)
        pred_test = model.predict_generator(generator=test_data_gen, workers=2, use_multiprocessing=False)

        #test_sed_pred = evaluation_metrics.reshape_3Dto2D(pred_test[0]) > 0.5
        test_doa_pred = evaluation_metrics.reshape_3Dto2D(pred_test)

        # rescaling the elevation data from [-180 180] to [-def_elevation def_elevation] for scoring purpose
        test_doa_pred[:, params['nb_classes']:] = test_doa_pred[:, params['nb_classes']:] / (180. / test_data_gen._default_ele)

        dcase_dump_folder = os.path.join(params['res_dir'], '{}_{}_split{}'.format(params['dataset'], params['mode'], params['test_splits'][split_cnt]))
        data_preparation.create_folder(params['res_dir'])
        data_preparation.create_folder(dcase_dump_folder)
        
        test_gt_sed, test_gt_doa = test_data_gen.get_val_labels()
        test_gt_sed = evaluation_metrics.reshape_3Dto2D(test_gt_sed)
        test_gt_doa = evaluation_metrics.reshape_3Dto2D(test_gt_doa)
        # rescaling the reference elevation from [-180 180] to [-def_elevation def_elevation] for scoring purpose
        test_gt_doa[:, params['nb_classes']:] = test_gt_doa[:, params['nb_classes']:] / (180. / test_data_gen._default_ele)

        test_doa_loss = evaluation_metrics.compute_doa_scores_regr(test_doa_pred, test_gt_doa, test_gt_sed, test_gt_sed)

        avg_scores_test.append([test_doa_loss[0], test_doa_loss[1]])
        print('Results on test split:')
        print('\tDOA_error: {},  '.format(test_doa_loss[0]))
        print('\tDOA Metrics: DOA_error: {}, frame_recall: {}'.format(test_doa_loss[0], test_doa_loss[1]))

    print('\n\nValidation split scores per fold:\n')
    for cnt in range(len(params['test_splits'])):
        print('\tSplit {} - DOA error: {} frame recall: {}'.format(cnt, avg_scores_val[cnt][0], avg_scores_val[cnt][1]))

    print('\n\nTesting split scores per fold:\n')
    for cnt in range(len(params['test_splits'])):
        print('\tSplit {} - DOA error: {} frame recall: {}'.format(cnt, avg_scores_test[cnt][0], avg_scores_test[cnt][1]))
