import os
import sys
import numpy as np
import time
from IPython import embed
import data_preparation
import warnings

feat_class = data_preparation.DataPreparation(datadir = '../dataset/pyroomsimulated/sound', labelsdir = '../dataset/pyroomsimulated/label', outputdir = '../dataset/pyroomsimulated/features')

feat_class.extract()