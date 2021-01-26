import os
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
import numpy as np
from OrigSELDNet import data_preparation
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

dataset_name = 'foa'
datadir = '../dataset'
featlabelsdir = '../dataset/spec_feat'

prepare_class = data_preparation.DataPreparation(dataset_dir=datadir, feat_label_dir=featlabelsdir, dataset=dataset_name, is_eval=False)

prepare_class.extract_all_feature()
prepare_class.preprocess_features()
prepare_class.extract_all_labels()