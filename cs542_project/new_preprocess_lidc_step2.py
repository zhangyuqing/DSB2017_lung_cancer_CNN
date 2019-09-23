import xmltodict
import numpy as np
from glob import glob
import SimpleITK as sitk
import os
import csv

# Match the LIDC annotations to LUNA16 CT scans
LIDC_PREPROCESSED_FILE = './data_luna16/lidc_xml_preprocessed.npy'

DATA_FILE_FILTER = './data_luna16/subset*/*.mhd'
ANN_FILE = './data_luna16/annotations.csv'
SAVE_FILE = './data_luna16/malignant_nodule_ranges.npz'

MALIGNANCY_THRESHOLD = 4  # 4, 5 will be treated as cancer

'''
This funciton reads a '.mhd' file using SimpleITK and return the image array,
origin and spacing of the image.
'''
def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

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

def load_lidc_annotations(ann_file):
    '''
    Load the LIDC preprocessed file into a dictionary
    '''
    lidc_list = list(np.load(ann_file))
    sample_dict = {ann['uid']: [] for ann in lidc_list}
    for ann in lidc_list:
        sample_dict[ann['uid']] += ann['readings']
    return sample_dict

# Load Annotations
uid2anns = load_luna16_annotations(ANN_FILE)
print('number of patients:', len(uid2anns))

# Load CT scan paths
scan_files = glob(DATA_FILE_FILTER)
uid2scan_path = {os.path.basename(path).replace('.mhd', ''): path
                 for path in scan_files}
print('number of files:', len(uid2scan_path))

uid2readings = load_lidc_annotations(LIDC_PREPROCESSED_FILE)
print('number of annotations in LIDC:', len(uid2readings))

# load the origin and spacing of each LUNA16 scans
uids2ct_scan_shape = {}
uids2origin = {}
uids2spacing = {}
for uid in uid2anns:
    ct_scan, origin, spacing = load_itk(uid2scan_path[uid])
    uids2ct_scan_shape[uid] = ct_scan.shape
    uids2origin[uid] = origin
    uids2spacing[uid] = spacing

np.savez('./data_luna16/origin_and_spacing.npy.npz',
         uids2ct_scan_shape=uids2ct_scan_shape,
         uids2origin=uids2origin,
         uids2spacing=uids2spacing)

def convert_roi_to_coord_range(roi, origin, spacing, shape):
    '''
    Crop a rectangular region using the maximum coordinate range
    '''
    Z_origin = origin[0]
    Z_spacing, Y_spacing, X_spacing = spacing
    z_list = []
    xy_list = []
    for layer in roi:
        # convert Z word coordinate to voxel coordinate
        z = (float(layer['imageZposition']) - Z_origin)
        assert(z >= 0)
        z_list.append(z)
        xy_list += layer['edgeCoords']
    z_list = np.array(z_list, dtype=np.int32)
    z_min, z_max = np.min(z_list), np.max(z_list)

    new_shape = np.round(shape * spacing)
    upper_offset = (512 - new_shape[1]) // 2

    xy_list = np.array(xy_list, np.float32)
    xy_list *= np.array([X_spacing, Y_spacing])
    xy_list += upper_offset
    x_min, y_min = np.min(xy_list, axis=0).astype(np.int32)
    x_max, y_max = np.max(xy_list, axis=0).astype(np.int32)

    return z_min, y_min, x_min, z_max, y_max, x_max

uid2_malignant_nodule_ranges = {uid: [] for uid in uid2anns}
for uid in uid2_malignant_nodule_ranges:
    readings = uid2readings[uid]
    for r in readings:
        if r['malignancy'] < MALIGNANCY_THRESHOLD:
            continue
        roi = r['roi']
        coords_range = convert_roi_to_coord_range(roi, uids2origin[uid], uids2spacing[uid], uids2ct_scan_shape[uid])
        uid2_malignant_nodule_ranges[uid].append(coords_range)

np.savez(SAVE_FILE, uid2_malignant_nodule_ranges=uid2_malignant_nodule_ranges)
