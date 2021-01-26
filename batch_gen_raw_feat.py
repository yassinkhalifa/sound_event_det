import os
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
import numpy as np
from RawSELDNet import data_preparation
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


datadir = '../dataset/foa_dev'
labelsdir = '../dataset/metadata_dev'
outdir = '../dataset/raw_feat'

#datadir = '../../datasets/dcase_2019/task_3/foa_dev'
#labelsdir = '../../datasets/dcase_2019/task_3/metadata_dev'
#outdir = '../../datasets/dcase_2019/task_3/raw_feat'

prepare_class = data_preparation.DataPreparation(datadir=datadir, labelsdir=labelsdir, outputdir=outdir, fs=48000, winlen_s = 0.25, hoplen_s = 0.125)

prepare_class.split_data()