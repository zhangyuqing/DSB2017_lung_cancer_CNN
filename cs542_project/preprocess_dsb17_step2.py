import os
import glob
import numpy as np

old_preprocessed_dir = './data/preprocessed/'
new_preprocessed_dir = './data/preprocessed_aligned/'

fnames = glob.glob(old_preprocessed_dir + '*.npz')
N = len(fnames)
print('sample number:', N)

os.makedirs(new_preprocessed_dir, exist_ok=True)

print('data keys:', np.load(fnames[0]).keys())

stats = np.zeros((N, 3, 3)) # min, mean and max of the lung part
for n, fname in enumerate(fnames):
    print('pass 1, processing %d / %d' % (n+1, N))

    # use segmented_lungs_fill_dilated to calculate lungs center and size
    d = np.load(fname)
    Zs, Ys, Xs = np.nonzero(d['segmented_lungs_fill_dilated'])
    d.close()
    # min, mean and max of the lung
    stats[n, :, :] = [[np.min(Zs), np.round(np.mean(Zs)), np.max(Zs)+1],
                      [np.min(Ys), np.round(np.mean(Ys)), np.max(Ys)+1],
                      [np.min(Xs), np.round(np.mean(Xs)), np.max(Xs)+1]]

# distance between max and mean; mean and min
dist_stats = np.zeros((N, 3, 2))
# the size of each lung (distance w.r.t. the center)
dist_stats[:,:,0] = stats[:,:,1] - stats[:,:,0]
dist_stats[:,:,1] = stats[:,:,2] - stats[:,:,1]

# maximum distance between the center point and the edges
dist_max = np.max(dist_stats, axis=0, keepdims=True)
print('maximum distance from centroid', dist_max, sep='\n')

new_shape = dist_max[0, :, 0] +  dist_max[0, :, 1]
print('new image shape in Z-Y-X:', new_shape)

# the new adjusted position of the beginning and ending of the lung
new_edges = np.zeros_like(dist_stats)
new_edges[:, :, 0] = stats[:, :, 1] - dist_max[:, :, 0]
new_edges[:, :, 1] = stats[:, :, 1] + dist_max[:, :, 1]

def crop_and_pad(old_array, new_edge, fill=0):
    edge_Z1, edge_Y1, edge_X1 = new_edge[:, 0].astype(np.int32)
    edge_Z2, edge_Y2, edge_X2 = new_edge[:, 1].astype(np.int32)
    len_Z, len_Y, len_X = old_array.shape

    # pad or crop at beginning
    old_begin_Z, old_begin_Y, old_begin_X = [max(edge_Z1, 0), max(edge_Y1, 0), max(edge_X1, 0)]
    new_begin_Z, new_begin_Y, new_begin_X = [max(-edge_Z1, 0), max(-edge_Y1, 0), max(-edge_X1, 0)]
    # pad or crop at the end
    old_end_Z, old_end_Y, old_end_X = [min(edge_Z2, len_Z), min(edge_Y2, len_Y), min(edge_X2, len_X)]
    new_end_Z, new_end_Y, new_end_X = [min(edge_Z2, len_Z)-edge_Z1,
                                       min(edge_Y2, len_Y)-edge_Y1,
                                       min(edge_X2, len_X)-edge_X1]

    # copy the data from old array to new array
    new_shape = tuple((new_edge[:, 1] - new_edge[:, 0]).astype(np.int32))
    new_array = np.ones(shape=new_shape, dtype=old_array.dtype)
    new_array[...] = fill
    new_array[new_begin_Z:new_end_Z, new_begin_Y:new_end_Y, new_begin_X:new_end_X] =         old_array[old_begin_Z:old_end_Z, old_begin_Y:old_end_Y, old_begin_X:old_end_X]
    return new_array

for n, fname in enumerate(fnames):
    print('pass 2, processing %d / %d: %s' % (n+1, N, fname))

    # use segmented_lungs_fill_dilated to calculate lungs center and size
    old_d = np.load(fname)
    pix_resampled_old = old_d['pix_resampled']
    pix_resampled_new = crop_and_pad(pix_resampled_old, new_edges[n], fill=np.min(pix_resampled_old))
    segmented_lungs_fill_dilated_old = old_d['segmented_lungs_fill_dilated']
    segmented_lungs_fill_dilated_new = crop_and_pad(segmented_lungs_fill_dilated_old, new_edges[n],
                                                    fill=np.min(segmented_lungs_fill_dilated_old))
    old_d.close()
    new_fname = fname.replace(old_preprocessed_dir, new_preprocessed_dir)
    np.savez(new_fname, pix_resampled=pix_resampled_new,
             segmented_lungs_fill_dilated=segmented_lungs_fill_dilated_new)
