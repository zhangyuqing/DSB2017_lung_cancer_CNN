# CS 542 Project (code release)

This code release contains the code for our CS 542 project on **Lung Cancer Detection from CT Scan Images Using Deep Convolutional Neural Networks**.

Team member: Yuqing Zhang, Rahul Bazaz, Howard Fan, Maulik Shah

## Prerequisits

1. A [CUDA compatible GPU](https://developer.nvidia.com/cuda-gpus) with at least 3.5 computation power and 12 GB memory.
2. At least 100 GB free disk space to store the data and intermediate results.
3. Install Python 3. We recommend using [Anaconda](https://www.continuum.io/downloads) to install all the dependencies at once.
4. Install TensorFlow v1.0.0 or higher, following the instructions [here](https://www.tensorflow.org/).

## Dataset download

1. Download the Data Science Bowl 2017 dataset from https://www.kaggle.com/c/data-science-bowl-2017, and put the unzipped data under `data/`
2. Download the LUNA-16 dataset from https://luna16.grand-challenge.org/home/, and put the unzipped data under `data_luna16/`.

## Preprocessing

1. Preprocess the DSB-17 dataset
```
python new_preprocess_dsb17_step1.py
python new_preprocess_dsb17_step2.py
```

2. Preprocess the LUNA-16 dataset
```
python new_preprocess_luna16_step1.py
python new_preprocess_luna16_step2.py
```

## Train U-Net models on LUNA-16 for nodule segmentation

1. Train the U-Net models for nodule segmentation along all three axes
```
python train_unet_nodule_segmentation.py --axis 0 --gpu_id 0
python train_unet_nodule_segmentation.py --axis 1 --gpu_id 0
python train_unet_nodule_segmentation.py --axis 2 --gpu_id 0
```
The trained model together with TensorBoard logging will be stored under `train/`.

2. Predict the nodule probability on CT scans in the DSB-17 dataset
```
python predict_dsb17_nodule_segmentation_step1.py --epoch 100 --gpu_id 0
python predict_dsb17_nodule_segmentation_step2.py --epoch 100 --gpu_id 0
```
The scripts above uses GPU 0 on your machine. If you want to use another GPU, set `--gpu_id` flag accordingly.

## Lung cancer classification on DSB-17 over nodule clusters

1. Train and evaluate the 3D convolutional neural network over nodule clusters
```
python run_training_exp20_nodule_classifier.py --gpu_id 0
```
The script above does evaluation during training. The trained model together with TensorBoard logging will be stored under `train/`.

## Other files

### Experiments

We tried 24 different experiments in this project. A few of them worked, while others did not work. All the Python files in the format `run_training_exp*.py` and `run_test_exp*.py` are the training and test code for each experiment. In the later experiments we merged the test code into the training script so there were no separate test files for them.

These experiments rely on different preprocessing methods on the DSB-17 dataset (those `preprocess_dsb17_*.py` files). You may run all the preprocessing we tried at once:
```
preprocess_dsb17_step1.py
preprocess_dsb17_step2.py
preprocess_dsb17_step2_add0.py
preprocess_dsb17_reshape.py
preprocess_dsb17_reshape2.py
```

### Visualization

This code release also contains some IPython Notebooks (those `visualize_*.ipynb` files) that were used to visualize the results. You may run them directly with IPython Notebook.
