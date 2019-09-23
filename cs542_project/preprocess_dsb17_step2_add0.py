from glob import glob
import numpy as np # linear algebra
import matplotlib.pyplot as plt

SAVE_FOLDER = './data/preprocessed_add0/'
fnames = glob('./data/preprocessed/*.npz')

size_mat = np.arange(len(fnames)*3).reshape(len(fnames),3)
np.shape(size_mat)

for i in range(len(fnames)):
    data = np.load(fnames[i])
    full_image = data['pix_resampled']

    size_mat[i, ] = np.shape(full_image)

max_sizes = np.amax(size_mat, axis=0)

for i in range(len(fnames)):
    save_file = SAVE_FOLDER + fnames[i][20:]
    print(save_file)

    full_image_new = np.zeros(max_sizes)

    data = np.load(fnames[i])
    full_image = data['pix_resampled']
    this_size = size_mat[i, ]

    full_image_new[int((max_sizes[0]-this_size[0])/2):int((max_sizes[0]-this_size[0])/2+this_size[0]),
                   int((max_sizes[1]-this_size[1])/2):int((max_sizes[1]-this_size[1])/2+this_size[1]),
                   int((max_sizes[2]-this_size[2])/2):int((max_sizes[2]-this_size[2])/2+this_size[2])] = full_image

    #print(this_size)
    #print(int((max_sizes[1]-this_size[1])/2))
    #print(len(np.arange(np.floor((max_sizes[1]-this_size[1])/2),(np.floor((max_sizes[1]-this_size[1])/2)+this_size[1]))))

    np.savez(
        save_file,
        pix_resampled_add0=full_image_new)
