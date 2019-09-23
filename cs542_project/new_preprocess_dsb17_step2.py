import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=0)
parser.add_argument('-e', type=int, default=-1)
args = parser.parse_args()

import os

import numpy as np
import csv
from glob import glob

import tqdm

DATA_FILE_FILTER = './data/stage1/*'
DATA_DIR = './data/preprocessed_masks/'
SAVE_DIR = './data/preprocessed_masks/lung_region/'

def get_lung_range(lung_mask, axis, minsize=128):
    nonzeros = np.nonzero(np.any(lung_mask, axis=axis))[0]
    begin, end = np.min(nonzeros), np.max(nonzeros)+1
    length = end-begin
    if length < minsize:
        begin = max(0, begin-(minsize-length)//2)
        end = begin+minsize
    return begin, end

# Load list
file_paths = glob(DATA_FILE_FILTER)
print('number of files:', len(file_paths))
file_paths = file_paths[args.b:args.e]
print('number of files in range %d - %d: %d' % (args.b, args.e, len(file_paths)))

os.makedirs(SAVE_DIR, exist_ok=True)
for n in tqdm.trange(len(file_paths)):
    path = file_paths[n]
    uid = os.path.basename(path)
    
    lung_img = np.load(os.path.join(DATA_DIR, uid + '_lung_img.npy'))
    lung_mask = np.load(os.path.join(DATA_DIR, uid + '_lung_mask.npy'))
    
    begin_z, end_z = get_lung_range(lung_mask, axis=(1, 2))
    begin_y, end_y = get_lung_range(lung_mask, axis=(0, 2))
    begin_x, end_x = get_lung_range(lung_mask, axis=(0, 1))
    
    np.savez(os.path.join(SAVE_DIR, uid+'_lung_region.npz'),
         lung_img=lung_img[begin_z:end_z, begin_y:end_y, begin_x:end_x],
         lung_mask=lung_mask[begin_z:end_z, begin_y:end_y, begin_x:end_x])