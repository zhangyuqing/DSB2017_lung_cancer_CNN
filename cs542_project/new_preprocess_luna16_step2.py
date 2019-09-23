import numpy as np
import os
import csv

ANN_FILE = './data_luna16/annotations.csv'
DATA_DIR = './data_luna16/preprocessed_masks/'
SAVE_DIR = './data_luna16/preprocessed_masks/unet_train/'

def get_lung_range(lung_mask, axis, minsize=128):
    nonzeros = np.nonzero(np.any(lung_mask, axis=axis))[0]
    begin, end = np.min(nonzeros), np.max(nonzeros)+1
    length = end-begin
    if length < minsize:
        begin = max(0, begin-(minsize-length)//2)
        end = begin+minsize
    return begin, end

def load_luna16_annotations(ann_file):
    '''
    Load the LUNA16 annotations into a dictionary:
        {<uid>: [[X1, Y1, Z1, D1], [X2, Y2, Z2, D2], ...]}
        where the key is the uid string like
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.122621219961396951727742490470'
        and the value is a list of length-4 list, containing X, Y, Z coordinate and diameter
        of each nodule in the patient, represented in world coordinate
    '''
    sample_dict = {}
    with open(ann_file) as f:
        for uid, X, Y, Z, D in csv.reader(f):
            # skip the first line
            if uid == 'seriesuid':
                continue

            if uid not in sample_dict:
                sample_dict[uid] = [[float(X), float(Y), float(Z), float(D)]]
            else:
                sample_dict[uid].append([float(X), float(Y), float(Z), float(D)])

    return sample_dict

uid2anns = load_luna16_annotations(ANN_FILE)
os.makedirs(SAVE_DIR, exist_ok=True)

for n, uid in enumerate(uid2anns):
    print('processing %d / %d' % (n, len(uid2anns)))

    lung_img = np.load(os.path.join(DATA_DIR, uid + '_lung_img.npy'))
    lung_mask = np.load(os.path.join(DATA_DIR, uid + '_lung_mask.npy'))
    nodule_mask = np.load(os.path.join(DATA_DIR, uid + '_nodule_mask.npy'))

    begin_z, end_z = get_lung_range(lung_mask, axis=(1, 2))
    begin_y, end_y = get_lung_range(lung_mask, axis=(0, 2))
    begin_x, end_x = get_lung_range(lung_mask, axis=(0, 1))
    image = lung_img[begin_z:end_z, begin_y:end_y, begin_x:end_x]
    label = nodule_mask[begin_z:end_z, begin_y:end_y, begin_x:end_x]

    np.savez(os.path.join(SAVE_DIR, uid+'_unet_train.npz'),
         image=image, label=label)
