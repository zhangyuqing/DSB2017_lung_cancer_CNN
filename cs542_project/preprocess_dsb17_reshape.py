import numpy as np
import os
import scipy.ndimage

# Step 1: load CT scans
def load_scan(path):
    import dicom
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    # order slices by ImagePosition 3rd variable (z axis)
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

# Step 2: get HU values from scans
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

# Step 3: resize (downsample) all samples to a fixed size
def downsample_to_fixsize(image, new_shape):
    # Determine current pixel spacing
    old_shape = image.shape
    downsample_factor = np.array(new_shape) / np.array(old_shape)
    image_downsampled =         scipy.ndimage.zoom(image, downsample_factor)        .astype(np.float32)
    assert(image_downsampled.shape == new_shape)
    return image_downsampled

# Step 4: truncate and normalize the data to range (0, 1)
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def full_preprocessing(path, new_shape):
    return normalize(downsample_to_fixsize(get_pixels_hu(load_scan(path)), new_shape))

################################################################################
# Some constants
INPUT_FOLDER = './data/stage1/'
SAVE_FOLDER = './data/preprocessed_reshaped/'
SAVE_MEAN_FILE = './data/preprocessed_reshaped_mean.npy'
os.makedirs(SAVE_FOLDER, exist_ok=True)
patients = os.listdir(INPUT_FOLDER)
patients.sort()

new_shape = (50, 50, 50)
mean_pix = np.zeros(new_shape, np.float32)
mean_count = 0
for patient_id in range(len(patients)):
    print('processing %d / %d' % (patient_id+1, len(patients)))

    load_file = INPUT_FOLDER + patients[patient_id]
    save_file = SAVE_FOLDER + patients[patient_id] + '.npz'

    pix_resampled = full_preprocessing(load_file, new_shape)

    # accumulate pixel mean
    mean_pix += pix_resampled
    mean_count += 1

    # save data to disk
    np.savez(
        save_file,
        pix_resampled=pix_resampled)

# Save pixel mean
mean_pix /= mean_count
np.save(SAVE_MEAN_FILE, mean_pix)
