import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
import numpy as np
import matplotlib.pyplot as plot
from keras.models import load_model
from RawSELDNet.keras_model import get_model
import time
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from keras.layers import Bidirectional, Conv2D, DepthwiseConv2D, MaxPooling2D, Input
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.utils.io_utils import HDF5Matrix
import keras
from IPython import embed
from RawSELDNet.helper_classes import DataGenerator
from RawSELDNet import evaluation_metrics
from RawSELDNet import data_preparation
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ex = Experiment('RawSELDnet Experiment')
ex.observers.append(MongoObserver.create(url='192.168.1.102:27017',db_name='SELDNet'))
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    params = {
        'dataset':'foa',
        'mode':'dev',
        'log_dir':'./rawseldnet_logs',
        'res_dir':'./rawseldnet_results',
        'fs':48000,
        'win_len':0.25,
        'hop_len':0.125,
        'max_frames': 479,
        'nb_channels':4,
        'nb_classes':11,
        'train_splits':[[3, 4], [4, 1], [1, 2], [2, 3]],
        'val_splits':[[2], [3], [4], [1]],
        'test_splits':[[1], [2], [3], [4]],
        'loss_weights':[1., 50.],                                 # [sed, doa] weight for scaling the DNN outputs
        'nb_epochs':100,
        'batch_size':64,
        'patience':10
    }
    network_params = {
        'nb_cnn_filt':[64, 128, 256, 512, 512],
        'pool_size':[2, 2, 2, 2, 2],
        'rnn_size':[128, 128],
        'fnn_size':[128, 128],
        'dropout_rate':0.0
    }

@ex.capture
def log_performance(epoch_cnt, tr_loss, val_loss, ER_overall, F1_overall, doa_error_pred, good_pks_ratio, seld_score):
    ex.log_scalar("epoch_cnt", epoch_cnt)
    ex.log_scalar("train_loss", tr_loss)
    ex.log_scalar("val_loss", val_loss)
    ex.log_scalar("ER_overall", ER_overall)
    ex.log_scalar("F1_overall", F1_overall)
    ex.log_scalar("doa_error_pred", doa_error_pred)
    ex.log_scalar("good_pks_ratio", good_pks_ratio)
    ex.log_scalar("seld_score", seld_score)

@ex.automain
def main(params, network_params):
    #datadir = '../dataset/raw_feat/data'
    #labelsdir = '../dataset/raw_feat/label'
    #outdir = '../dataset/raw_feat'
    datadir = '../../datasets/dcase_2019/task_3/raw_feat/data'
    labelsdir = '../../datasets/dcase_2019/task_3/raw_feat/labels'
    outdir = '../../datasets/dcase_2019/task_3/raw_feat'

    prepare_class = data_preparation.DataPreparation(datadir=datadir, labelsdir=labelsdir, outputdir=outdir)
    avg_scores_val = []
    avg_scores_test = []
    data_preparation.create_folder(params['log_dir'])

    frame_data = HDF5Matrix(os.path.join(datadir, 'dcase19_dataset.h5'), 'frame_data')
    frame_label = HDF5Matrix(os.path.join(datadir, 'dcase19_dataset.h5'), 'frame_label')
    frame_label = frame_label.data.value
    frame_fileorder = HDF5Matrix(os.path.join(datadir, 'dcase19_dataset.h5'), 'frame_fileorder')
    frame_fileorder = frame_fileorder.data.value
    frame_file = np.load(os.path.join(datadir, 'frame_files.npy'))

    for split_cnt, split in enumerate(params['test_splits']):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        print('---------------------------------------------------------------------------------------------------')
        
        unique_name = '{}_{}_split{}'.format(params['dataset'], params['mode'], split)
        model_name = os.path.join(params['log_dir'], '{}_model.h5'.format(unique_name))

        
        # Load train and validation data
        print('Loading training dataset:')
        train_data_gen = DataGenerator(h5_file=frame_data, 
            frame_label = frame_label,
            frame_file = frame_file,
            frame_fileorder = frame_fileorder, 
            shuffle=True, perfile=False, train=True,
            splits=params['train_splits'][split_cnt], 
            max_frames=params['max_frames'], 
            fs=params['fs'],
            win_len_s=params['win_len'], 
            hop_len_s=params['hop_len'], 
            nb_channels = params['nb_channels'], 
            nb_classes = params['nb_classes'], 
            batch_size = params['batch_size'])
        
        print('Loading validation dataset:')
        valid_data_gen = DataGenerator(h5_file=frame_data, 
            frame_label = frame_label,
            frame_file = frame_file,
            frame_fileorder = frame_fileorder, 
            shuffle=False, perfile=False, train=False,
            splits=params['val_splits'][split_cnt],
            max_frames=params['max_frames'], 
            fs=params['fs'],
            win_len_s=params['win_len'], 
            hop_len_s=params['hop_len'], 
            nb_channels = params['nb_channels'], 
            nb_classes = params['nb_classes'], 
            batch_size = params['batch_size'])
        val_gt_sed, val_gt_doa = valid_data_gen.get_val_labels()

        # rescaling the reference elevation data from [-180 180] to [-def_elevation def_elevation] for scoring purpose
        nb_classes = params['nb_classes']
        def_elevation = train_data_gen._default_ele
        val_gt_doa[:, nb_classes:] = val_gt_doa[:, nb_classes:] / (180. / def_elevation)

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
                epochs=5, workers=16, use_multiprocessing=True)

            tr_loss[epoch_cnt] = hist.history.get('loss')[-1]
            val_loss[epoch_cnt] = hist.history.get('val_loss')[-1]

            pred = model.predict_generator(generator=valid_data_gen, workers=16, use_multiprocessing=True)

            # Calculate the metrics
            sed_pred = pred[0] > 0.5
            doa_pred = pred[1]
            # rescaling the elevation data from [-180 180] to [-def_elevation def_elevation] for scoring purpose
            doa_pred[:, params['nb_classes']:] = doa_pred[:, params['nb_classes']:] / (180. / valid_data_gen._default_ele)

            sed_metric[epoch_cnt, :] = evaluation_metrics.compute_sed_scores(sed_pred, val_gt_sed, prepare_class.get_frame_res())
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
            log_performance(epoch_cnt, tr_loss[epoch_cnt], val_loss[epoch_cnt],
                    sed_metric[epoch_cnt, 0], sed_metric[epoch_cnt, 1],
                    doa_metric[epoch_cnt, 0], doa_metric[epoch_cnt, 1],
                    seld_metric[epoch_cnt])
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
        test_data_gen = DataGenerator(h5_file=frame_data, 
            frame_label = frame_label,
            frame_file = frame_file,
            frame_fileorder = frame_fileorder, 
            shuffle=False, perfile=True, train=False, 
            splits=params['test_splits'][split_cnt],
            max_frames=params['max_frames'], 
            fs=params['fs'],
            win_len_s=params['win_len'], 
            hop_len_s=params['hop_len'], 
            nb_channels = params['nb_channels'], 
            nb_classes = params['nb_classes'], 
            batch_size = params['batch_size'])

        print('\nLoading the best model and predicting results on the testing split')
        model = load_model(model_name)
        pred_test = model.predict_generator(generator=test_data_gen, workers=16, use_multiprocessing=True)

        test_sed_pred = pred_test[0] > 0.5
        test_doa_pred = pred_test[1]

        # rescaling the elevation data from [-180 180] to [-def_elevation def_elevation] for scoring purpose
        test_doa_pred[:, params['nb_classes']:] = test_doa_pred[:, params['nb_classes']:] / (180. / test_data_gen._default_ele)

        dcase_dump_folder = os.path.join(params['res_dir'], '{}_{}_split{}'.format(params['dataset'], params['mode'], params['test_splits'][split_cnt]))
        data_preparation.create_folder(params['res_dir'])
        data_preparation.create_folder(dcase_dump_folder)
        test_filelist = test_data_gen._frame_file[test_data_gen._custom_idxs]

        for file_cnt, test_file in enumerate(test_filelist):
            output_file = os.path.join(dcase_dump_folder, test_file.replace('.npy', '.csv'))
            dc = file_cnt*params['max_frames']
            output_dict = evaluation_metrics.regression_label_format_to_output_format(
                    prepare_class,
                    test_sed_pred[dc:dc + params['max_frames'], :],
                    test_doa_pred[dc:dc + params['max_frames'], :] * 180 / np.pi
            )
            evaluation_metrics.write_output_format_file(output_file, output_dict)

        
        test_gt_sed, test_gt_doa = test_data_gen.get_val_labels()
        test_gt_sed = evaluation_metrics.reshape_3Dto2D(test_gt_sed)
        test_gt_doa = evaluation_metrics.reshape_3Dto2D(test_gt_doa)
        # rescaling the reference elevation from [-180 180] to [-def_elevation def_elevation] for scoring purpose
        test_gt_doa[:, params['nb_classes']:] = test_gt_doa[:, params['nb_classes']:] / (180. / test_data_gen._default_ele)

        test_sed_loss = evaluation_metrics.compute_sed_scores(test_sed_pred, test_gt_sed, prepare_class.get_frame_res())
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