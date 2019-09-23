import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=0)
parser.add_argument('-e', type=int, default=-1)
args = parser.parse_args()

import os

import SimpleITK as sitk
import numpy as np
import scipy.ndimage

import csv
from glob import glob

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import dicom
import scipy.misc

import tqdm

DATA_FILE_FILTER = './data/stage1/*'
SAVE_DIR = './data/preprocessed_masks'

def get_segmented_lungs(im, plot=False):

    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image.
    '''
    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)

    # just return the binary image
    binary = binary.astype(np.bool)
    return binary
#     '''
#     Step 8: Superimpose the binary mask on the input image.
#     '''
#     get_high_vals = binary == 0
#     im[get_high_vals] = 0
#     if plot == True:
#         plots[7].axis('off')
#         plots[7].imshow(im, cmap=plt.cm.bone)

#     return im

def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])

def load_scan(path):
    import dicom
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    # deal with some "dirty samples" here
    if 'b8bb02d229361a623a4dc57aa0e5c485' in path:
        # see https://www.kaggle.com/c/data-science-bowl-2017/discussion/28061
        slices = slices[107:]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) # order slices by ImagePosition 3rd variable (z axis)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image

def load_and_preproces(path):
    slices = load_scan(path)
    lung_img_raw = get_pixels_hu(slices)
    lung_img = resample(lung_img_raw, slices)

    # Segment the lung structure
    lung_img = lung_img + 1024
    lung_mask = segment_lung_from_ct_scan(lung_img)
    lung_img = lung_img - 1024

    return lung_img, lung_mask

# Load list
file_paths = glob(DATA_FILE_FILTER)
print('number of files:', len(file_paths))
file_paths = file_paths[args.b:args.e]
print('number of files in range %d - %d: %d' % (args.b, args.e, len(file_paths)))

os.makedirs(SAVE_DIR, exist_ok=True)
for n in tqdm.trange(len(file_paths)):
    path = file_paths[n]
    lung_img, lung_mask = load_and_preproces(path)

    base_save_path = os.path.join(SAVE_DIR, os.path.basename(path))
    np.save(base_save_path+'_lung_img.npy', lung_img)
    np.save(base_save_path+'_lung_mask.npy', lung_mask)
