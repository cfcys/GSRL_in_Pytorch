# -*- coding: utf-8 -*-
"""
Created on Mon May  1 17:47:13 2023

@author: ryxax
"""

# -*- coding: utf-8 -*-
import os
import argparse
from keras.callbacks import EarlyStopping
from keras import losses
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import Adadelta
from keras.optimizers import Adadelta
from sklearn.metrics.pairwise import paired_distances as dist
from Hyper import imgDraw, listClassification, resnet99_avg_recon
import libmr
import numpy as np
import rscls
import glob
from scipy import io
from copy import deepcopy
import time

os.environ['CUDA_VISIBLE_DEVICES']='1'
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--numTrain', type=int, default=20)  # number of training samples per class,small than 40
# parser.add_argument('--dataset', type=str, default='data/Trento/trento_im.npy')  # dataset name/home/amax/data/xibo/OpenSSLR/PaviaU/paviau_gt_wopen.txt
parser.add_argument('--dataset', type=str, default='data/Houston/houston_im.npy')  # dataset name/home/amax/data/xibo/OpenSSLR/PaviaU/paviau_gt_wopen.txt

# parser.add_argument('--gt', type=str, default='data/Trento/trento_raw_gt.npy')  # only known training samples included
parser.add_argument('--gt', type=str, default='data/Houston/houston_raw_gt.npy')  # only known training samples included

parser.add_argument('--batch_size', type=int, default=16)  # only known training samples included
parser.add_argument('--output', type=str, default='output/')  # save path for output files
args = parser.parse_args()

# generate output dir
early_stopping = EarlyStopping(monitor='loss', patience=1000)
key = args.dataset.split('/')[-1].split('_')[0]
spath = args.output + key + '_' + str(args.numTrain) + '/'
os.makedirs(spath, exist_ok=True)

gt = np.load(args.gt).astype('int')
listClassification(gt)

# pre_closed = np.argmax(pre_all, axis=-1) + 1  # baseline: closed
imgDraw(gt, spath + key + '_region', path='./', show=True)