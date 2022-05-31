#Import statements
import warnings
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import tensorflow as tf
import torch # torch is used to get GPU information
from sklearn.model_selection import train_test_split
from utils import piecewise_constant

# Python Imaging Library is used for opening image files
import PIL
import PIL.Image

import pathlib # required to access the image folders

# Variable to store number of available GPUs
num_of_gpu = 0

# Determine the number of GPUs available
if torch.cuda.is_available():    
    # Tell torch to use the GPU    
    device = torch.device("cuda")
    
    # Get the number of GPUs
    num_of_gpu = torch.cuda.device_count()
    print("Number of available GPU(s) %d." % num_of_gpu)

    print("GPU Name: ", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU")
    device = torch.device("cpu")
    
    
print("\n_____________________________________________________")

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("\nNumber of GPU(s): {}".format(strategy.num_replicas_in_sync))
print("\n_____________________________________________________")

#*********************************** Creating a Data Pipeline ***********************************

'''
We create a data pipeline to perform the following tasks:
- Load the dataset: from the local storage into the program's memory
- Prepare the dataset for training
'''



# ****************************** Load Data from Storage ******************************


'''
We use TensorFlow's Data API tf.data to load 
and preprocess a large dataset efficiently (faster).

This API loads the data as tf Dataset object, which is required for 
distributed training using multiple GPUs. 
'''


# Path of the local data directory
SD_train_dir= '/work/netthinker/shared/StanfordDataset/StanfordOriginal/train'
SD_test_dir = '/work/netthinker/shared/StanfordDataset/StanfordOriginal/test'

SDL_train_dir = '/work/netthinker/shared/StanfordDataset/StanfordSubsetRL/train'
'''
Approach for Loading the Images and their Labels:
- Create a list tf Dataset containing the path to the image files
- From this list tf dataset, get the (image, label) pairs
'''


'''
For creating image file list Dataset (for training & testing images) 
- First, we need to get the image file names and paths by accessig the image folders. 

We use Path and glob methods in the pathlib module for accessing the image folder. 
'''

SD_train_data_dir = pathlib.Path(SD_train_dir)
SD_train_file_paths = list(SD_train_data_dir.glob('*/*')) # Get the list of all training image file paths
print("\nNumber of training files: ", len(SD_train_file_paths))

SD_test_data_dir = pathlib.Path(SD_test_dir)
SD_test_file_paths = list(SD_test_data_dir.glob('*/*')) # Get the list of all test image file paths
print("Number of test files: ", len(SD_test_file_paths))

print("\nPath of the first training image: ", str(SD_train_file_paths[0]))
PIL.Image.open(str(SD_train_file_paths[0]))

SD_train_file_paths, SD_val_file_paths = train_test_split(SD_train_file_paths,test_size=0.1)

SDL_train_data_dir = pathlib.Path(SDL_train_dir)
SDL_train_file_paths = list(SDL_train_data_dir.glob('*/*')) # Get the list of all training image file paths
print("\nNumber of training files: ", len(SD_train_file_paths))

print("\nPath of the first training image: ", str(SD_train_file_paths[0]))
PIL.Image.open(str(SD_train_file_paths[0]))

SDL_train_file_paths, SDL_test_file_paths = train_test_split(SDL_train_file_paths,test_size=0.1)
SDL_train_file_paths, SDL_val_file_paths = train_test_split(SDL_train_file_paths,test_size=0.1)

'''
There are two options to create the image file list Dataset.

- Option 1: Create the file list Dataset by using the Dataset.from_tensor_slices() method
- Option 2: Create the file list Dataset by using the Dataset.list_files() method

Note that, before using Option 2, if the filenames have already been globbed, 
then re-globbing every filename with the list_files() method may result 
in poor performance with remote storage systems.

Since, we have already globbed the file names & paths in the first step, we do not use Option 2.
However, the code for option 2 is provided (commented out).
'''


#_______________________Option 1: "from_tensor_slices" method____________________________

SD_train_fnames=[]
for fname in SD_train_file_paths:
    SD_train_fnames.append(str(fname))

print(len(SD_train_fnames))

SD_test_fnames=[]
for fname in SD_test_file_paths:
    SD_test_fnames.append(str(fname))

print(len(SD_test_fnames))

SD_val_fnames=[]
for fname in SD_val_file_paths:
    SD_val_fnames.append(str(fname))

print(len(SD_val_fnames))

SDL_train_fnames=[]
for fname in SDL_train_file_paths:
    SDL_train_fnames.append(str(fname))

print(len(SDL_train_fnames))

SDL_test_fnames=[]
for fname in SDL_test_file_paths:
    SDL_test_fnames.append(str(fname))

print(len(SDL_test_fnames))

SDL_val_fnames=[]
for fname in SDL_val_file_paths:
    SDL_val_fnames.append(str(fname))

print(len(SDL_val_fnames))

# Create the image file list Dataset (for SD training & testing images) 
SD_train_dataset = tf.data.Dataset.from_tensor_slices(SD_train_fnames[:40])
SD_test_dataset = tf.data.Dataset.from_tensor_slices(SD_test_fnames[:40])
SD_val_dataset = tf.data.Dataset.from_tensor_slices(SD_val_fnames[:40])

SD_train_dataset_size = SD_train_dataset.cardinality().numpy()
SD_test_dataset_size = SD_test_dataset.cardinality().numpy()
SD_val_dataset_size = SD_val_dataset.cardinality().numpy()

print("\nNumber of samples in training dataset: ", SD_train_dataset_size)
print("Number of samples in test dataset: ", SD_test_dataset_size)
print("Number of samples in validation dataset: ", SD_val_dataset_size)

# Create the image file list Dataset (for SDL training & testing images) 
SDL_train_dataset = tf.data.Dataset.from_tensor_slices(SDL_train_fnames[:30])
SDL_test_dataset = tf.data.Dataset.from_tensor_slices(SDL_test_fnames[:30])
SDL_val_dataset = tf.data.Dataset.from_tensor_slices(SDL_val_fnames[:30])

SDL_train_dataset_size = SDL_train_dataset.cardinality().numpy()
SDL_test_dataset_size = SDL_test_dataset.cardinality().numpy()
SDL_val_dataset_size = SDL_val_dataset.cardinality().numpy()

print("\nNumber of samples in training dataset: ", SDL_train_dataset_size)
print("Number of samples in test dataset: ", SDL_test_dataset_size)
print("Number of samples in validation dataset: ", SDL_val_dataset_size)


# ****************************** Functions for Preparing Dataset for Training ******************************


'''
To prepare the Dataset for training, we need to:
- Get the (image, label) pairs from the list Dataset object (that contains list of file paths)
- Load these pairs in memory
- Create mini-batches for training and validation

We define the "prepare_dataset" function to accomplish this.
The "prepare_dataset" function requires to define some sub-routines, which we do first.
'''



'''
Get class names as a list
The class names will be used by the "get_label" function
NOTE: it only works if the data is structured in nested directories as specified 
at the beginning.
Following code works for the CIFAR-10 dataset. 
We may need to adapt it for a new dataset.
'''
SD_CLASS_NAMES = np.array([item.name for item in SD_train_data_dir.glob('*') if item.name != ".DS_Store"])
print("\nClass names:\n", SD_CLASS_NAMES)

SDL_CLASS_NAMES = np.array([item.name for item in SDL_train_data_dir.glob('*') if item.name != ".DS_Store"])
print("\nClass names:\n", SDL_CLASS_NAMES)


'''
Function to get the label of an image (file) from its path and class names
The labels are one-hot encoded
'''
def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory (i.e., name of the class)
  return tf.cast(parts[-2] == SD_CLASS_NAMES, dtype=tf.int32)
  

'''
This function reads a PNG-encoded image into a uint8 tensor
- Converts the uint8 tensor into float32
- Scales the tensor into the range [0,1] automatically while converting to float32
  Thus, we don't need to scale the images separately
- Resizes the image
Note: if the image encoding is other than PNG, then a suitable decode method should be used.
'''   
def process_image(image, image_height=100, image_width=100):  
    # Decode a PNG-encoded image to a uint8 tensor
    image = tf.image.decode_png(image, channels=3)

    # Convert the unit8 tensor to floats in the [0,1] range
    # Images that are represented using floating point values are expected to have values in the range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Resize image
    image = tf.image.resize(image, [image_height, image_width])
    
    return image
    
'''
This function returns a one-hot encoded labeled float32 tensor of an image 
based on its file path
'''
def load_labeled_data(file_path):
    img = tf.io.read_file(file_path)
    img = process_image(img)
    label = get_label(file_path)
    return img, label 

'''
Function to prepare the dataset for training.

We use the method chaining technique on the Dataset object to:
- Load (image, label) pairs into the Dataset object
- Cache the data
- Shuffle data
- Create mini-batches
- Prefetch a mini-batch

First, we call the Dataset's map() method that applies the data loading function on each sample of the dataset.
The returned new object will emit transformed images and their one-hot encoded labels in the original order.

Second, we cache the data locally.
This will save some operations (like file opening and data reading) from being executed during each epoch.

Third, we randomly shuffle the dataset using the shuffle method.

The shuffle method randomly shuffles the elements of the dataset.
First, it fills a buffer with the dataset with buffer_size elements. 
Then, it randomly samples elements from this buffer, 
replacing the selected elements with new elements. 
For perfect shuffling, the buffer_size should be greater than or equal to the size of the dataset. 
However, for large datasets, this isn't possible. So, we will use a large enough buffer_size.

Fourth, we batch the dataset. In the batch() method, we set "drop_remainder" to True 
so that the size of the training set is divisible by the batch_size. 
It is done by removing enough training examples. 

Finally, we prefetch a batch to decouple the time when data is produced from the time when data is consumed.
The transformation uses a background thread and an internal buffer to prefetch elements 
from the input dataset ahead of the time they are requested. 
The number of elements to prefetch should be equal to (or possibly greater than) 
the number of batches consumed by a single training step. 
Instead of manually tuning this value, we set it to tf.data.AUTOTUNE, 
which will prompt the tf.data runtime to tune the value dynamically at runtime.
'''

def prepare_dataset(ds):

    images = np.array([])
    labels = np.array([])

    for img in ds:
        image, label = load_labeled_data(img)
        images = np.append(images, image)
        labels = np.append(labels, label)

    print(len(images))
    print(len(labels))
    return images, labels



# ****************************** Prepare Dataset for Training ******************************                                  


# Perform data preprocessing by the CPU
# It is efficient as it ensures that the GPUs will be used only for model training
with tf.device('/cpu:0'):
    print("Loading Images")
    print("Loading SD Images")
    SD_train_images, SD_train_labels = prepare_dataset(SD_train_dataset)
    SD_test_images, SD_test_labels = prepare_dataset(SD_test_dataset)
    SD_val_images, SD_val_labels = prepare_dataset(SD_val_dataset)
    
    print("Length of SD Image Sets")
    print("SD Train: " + str(len(SD_train_images)))
    print("SD Test: " + str(len(SD_test_images)))
    print("SD Val: " + str(len(SD_val_images)))
    
    print("Loading SDL Images")
    SDL_train_images, SDL_train_labels = prepare_dataset(SDL_train_dataset)
    SDL_test_images, SDL_test_labels = prepare_dataset(SDL_test_dataset)
    SDL_val_images, SDL_val_labels = prepare_dataset(SDL_val_dataset)

    print("Length of SDL Image Sets")
    print("SDL Train: " + str(len(SDL_train_images)))
    print("SDL Test: " + str(len(SDL_test_images)))
    print("SDL Val: " + str(len(SDL_val_images)))
    
    print("Filling SdL Images")
    SDL_train_images = np.append(SDL_train_images, SDL_train_images[:10], axis = 0)
    SDL_train_labels = np.append(SDL_train_labels, SDL_train_labels[:10])
    SDL_val_images = np.append(SDL_val_images, SDL_val_images[:10], axis = 0)
    SDL_val_labels = np.append(SDL_val_labels, SDL_val_labels[:10])

    print("Length of New SDL Image Sets")
    print("SDL Train: " + str(len(SDL_train_images)))
    print("SDL Test: " + str(len(SDL_test_images)))
    print("SDL Val: " + str(len(SDL_val_images)))