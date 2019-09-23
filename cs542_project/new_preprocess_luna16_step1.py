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

ANN_FILE = './data_luna16/annotations.csv'
DATA_FILE_FILTER = './data_luna16/subset*/*.mhd'
SAVE_DIR = './data_luna16/preprocessed_masks'

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

'''
This function is used to convert the world coordinates to voxel coordinates using
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

'''
This function is used to convert the voxel coordinates to world coordinates using
the origin and spacing of the ct_scan.
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

def seq(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([start + step*i for i in range(n+1)])
    else:
        return([])

'''
This function is used to create spherical regions in binary masks
at the given locations and radius.
'''
def draw_circles(image,cands,origin,spacing):
    #make empty matrix, which will be filled with the mask
    RESIZE_SPACING = [1, 1, 1]
    image_mask = np.zeros(image.shape, np.bool)

    #run over all the nodules in the lungs
    for ca in cands:
        #get middel x-,y-, and z-worldcoordinate of the nodule
        coord_x, coord_y, coord_z = ca[0:3]
        radius = np.ceil(ca[3])/2
        image_coord = np.array((coord_z,coord_y,coord_x))

        #determine voxel coordinate given the worldcoordinate
        image_coord = world_2_voxel(image_coord,origin,spacing)

        #determine the range of the nodule
        noduleRange = seq(-radius, radius, RESIZE_SPACING[0])

        #create the mask
        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    coords = world_2_voxel(np.array((coord_z+z,coord_y+y,coord_x+x)),origin,spacing)
                    if (np.linalg.norm(image_coord-coords) * RESIZE_SPACING[0]) < radius:
                        image_mask[int(np.round(coords[0])),
                                   int(np.round(coords[1])),
                                   int(np.round(coords[2]))] = True

    return image_mask

'''
This function takes the path to a '.mhd' file as input and
is used to create the nodule masks and segmented lungs after
rescaling to 1mm size in all directions. It saved them in the .npz
format. It also takes the list of nodule locations in that CT Scan as
input.
'''
def create_nodule_mask(imagePath, cands):
    img, origin, spacing = load_itk(imagePath)

    #calculate resize factor
    RESIZE_SPACING = [1, 1, 1]
    resize_factor = spacing / RESIZE_SPACING
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / img.shape
    new_spacing = spacing / real_resize

    #resize image
    lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)

    # Segment the lung structure
    lung_img = lung_img + 1024
    lung_mask = segment_lung_from_ct_scan(lung_img)
    lung_img = lung_img - 1024

    #create nodule mask
    nodule_mask = draw_circles(lung_img,cands,origin,new_spacing)

    lung_img_512 = np.zeros((lung_img.shape[0], 512, 512), lung_img.dtype) - 1024
    lung_mask_512 = np.zeros((lung_mask.shape[0], 512, 512), lung_mask.dtype)
    nodule_mask_512 = np.zeros((nodule_mask.shape[0], 512, 512), nodule_mask.dtype)

    original_shape = lung_img.shape
    for z in range(lung_img.shape[0]):
        offset = (512 - original_shape[1])
        upper_offset = int(np.round(offset/2))
        lower_offset = offset - upper_offset

        new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)

        lung_img_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_img[z,:,:]
        lung_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_mask[z,:,:]
        nodule_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = nodule_mask[z,:,:]

    return lung_img_512, lung_mask_512, nodule_mask_512

#     # save images.
#     np.save(imageName + '_lung_img.npz', lung_img_512)
#     np.save(imageName + '_lung_mask.npz', lung_mask_512)
#     np.save(imageName + '_nodule_mask.npz', nodule_mask_512)

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

# Load Annotations
uid2anns = load_luna16_annotations(ANN_FILE)
print('number of patients:', len(uid2anns))


# Load CT scan paths
scan_files = glob(DATA_FILE_FILTER)
uid2scan_path = {os.path.basename(path).replace('.mhd', ''): path
                 for path in scan_files}
print('number of files:', len(uid2scan_path))

# Check that every annotated patient has CT scan
for uid in uid2anns:
    assert(uid in uid2scan_path)
print('Check passed: every annotated patient has CT scan')

os.makedirs(SAVE_DIR, exist_ok=True)
for n, uid in enumerate(uid2anns):
    print('processing %d / %d' % (n, len(uid2anns)))
    cands = uid2anns[uid]
    scan_path = uid2scan_path[uid]
    lung_img_512, lung_mask_512, nodule_mask_512 = create_nodule_mask(scan_path, cands)

    base_save_path = os.path.join(SAVE_DIR, uid)
    np.save(base_save_path+'_lung_img.npy', lung_img_512)
    np.save(base_save_path+'_lung_mask.npy', lung_mask_512)
    np.save(base_save_path+'_nodule_mask.npy', nodule_mask_512)
