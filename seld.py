import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
import numpy as np
import matplotlib.pyplot as plot
from keras.models import load_model
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
import keras
import keras.backend as K
from IPython import embed
from OrigSELDNet.helper_classes import DataGenerator
from OrigSELDNet import evaluation_metrics
from OrigSELDNet import data_preparation
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

def biternion_loss(btrnTrue, btrnPred):
    #btrnTrue and btrnPred are the Biternion representations of the reference angles and predicted ones respectively
    btrndotprod = K.batch_dot(btrnTrue, btrnPred, axes=[1, 2])
    btrnloss = K.mean(1-btrndotprod)
    return btrnloss

keras.backend.set_image_data_format('channels_first')

def get_model(params, network_params):
    x = Input(shape=(2*params['nb_channels'], params['sequence_len'], params['nb_freq_bins']))
    ten_cnn = x
    #CNN
    for cnn_cnt, cnn_pool in enumerate(network_params['pool_size']):
        ten_cnn = Conv2D(filters=network_params['nb_cnn2d_filt'], kernel_size=(3, 3), padding='same')(ten_cnn)
        ten_cnn = BatchNormalization()(ten_cnn)
        ten_cnn = Activation('relu')(ten_cnn)
        ten_cnn = MaxPooling2D(pool_size=(1, cnn_pool))(ten_cnn)
        ten_cnn = Dropout(network_params['dropout_rate'])(ten_cnn)
    ten_cnn = Permute((2, 1, 3))(ten_cnn)
    #RNN
    ten_rnn = Reshape((params['sequence_len'], -1))(ten_cnn)
    for nb_rnn in network_params['rnn_size']:
        ten_rnn = Bidirectional(
            GRU(nb_rnn, activation='tanh', dropout=network_params['dropout_rate'], recurrent_dropout=network_params['dropout_rate'],
                return_sequences=True),
            merge_mode='mul'
        )(ten_rnn)
    # FCN - SED
    sed = ten_rnn
    for nb_fcn in network_params['fcn_size']:
        sed = TimeDistributed(Dense(nb_fcn))(sed)
        sed = Dropout(network_params['dropout_rate'])(sed)
    sed = TimeDistributed(Dense(params['nb_classes']))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    #FCN - DOA
    doa = ten_rnn
    for nb_fcn in network_params['fcn_size']:
        doa = TimeDistributed(Dense(nb_fcn))(doa)
        doa = Dropout(network_params['dropout_rate'])(doa)
    doa = TimeDistributed(Dense(4*params['nb_classes']))(doa)
    doa = Reshape((params['sequence_len'], 2*params['nb_classes'], 2))(doa)
    doa = BatchNormalization()(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    model = Model(inputs=x, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', biternion_loss], loss_weights=params['loss_weights'])
    model.summary()
    return model

ex = Experiment('Original SELDnet with Biternion Loss')
ex.observers.append(MongoObserver.create(url='192.168.1.102:27017',db_name='SELDNet'))
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    params = {
        'dataset':'foa',
        'mode':'dev',
        'log_dir':'./seldnet_logs',
        'res_dir':'./seldnet_results',
        'sequence_len':100,
        'win_len':1920,
        'max_frames':1500,
        'nb_freq_bins':1024,
        'nb_channels':4,
        'nb_classes':11,
        'train_splits':[[3, 4], [4, 1], [1, 2], [2, 3]],
        'val_splits':[[2], [3], [4], [1]],
        'test_splits':[[1], [2], [3], [4]],
        'loss_weights':[1., 50.],                                 # [sed, doa] weight for scaling the DNN outputs
        'nb_epochs':100,
        'batch_size':15,
        'patience':10
    }
    network_params = {
        'nb_cnn2d_filt':64,
        'strides':(1,1),
        'pool_size':[8, 8, 4],
        'dropout_rate':0,
        'rnn_size':[128, 128],
        'fcn_size':[128]
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
                epochs=1, workers=2, use_multiprocessing=False)

            tr_loss[epoch_cnt] = hist.history.get('loss')[-1]
            val_loss[epoch_cnt] = hist.history.get('val_loss')[-1]

            pred = model.predict_generator(generator=valid_data_gen, workers=2, use_multiprocessing=False)

            # Calculate the metrics
            sed_pred = evaluation_metrics.reshape_3Dto2D(pred[0]) > 0.5
            doa_pred = evaluation_metrics.reshape_3Dto2D(evaluation_metrics.compute_angle_from_btrn(pred[1]))



            # rescaling the elevation data from [-180 180] to [-def_elevation def_elevation] for scoring purpose
            # the same here, all in radians

            doa_pred[:, params['nb_classes']:] = doa_pred[:, params['nb_classes']:] / (180. / valid_data_gen._default_ele)

            sed_metric[epoch_cnt, :] = evaluation_metrics.compute_sed_scores(sed_pred, val_gt_sed, prepare_class._frame_res)
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

        test_sed_pred = evaluation_metrics.reshape_3Dto2D(pred_test[0]) > 0.5
        test_doa_pred = evaluation_metrics.reshape_3Dto2D(evaluation_metrics.compute_angle_from_btrn(pred_test[1]))

        # rescaling the elevation data from [-180 180] to [-def_elevation def_elevation] for scoring purpose
        test_doa_pred[:, params['nb_classes']:] = test_doa_pred[:, params['nb_classes']:] / (180. / test_data_gen._default_ele)

        dcase_dump_folder = os.path.join(params['res_dir'], '{}_{}_split{}'.format(params['dataset'], params['mode'], params['test_splits'][split_cnt]))
        data_preparation.create_folder(params['res_dir'])
        data_preparation.create_folder(dcase_dump_folder)
        test_filelist = test_data_gen._current_files_list

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

        test_sed_loss = evaluation_metrics.compute_sed_scores(test_sed_pred, test_gt_sed, prepare_class._frame_res)
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
