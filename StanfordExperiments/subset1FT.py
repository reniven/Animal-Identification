# Use comet ml to track the progress of the training and see training statistics
from comet_ml import Experiment

# Create an experiment with the api key:
experiment = Experiment(
    api_key="d1doFSsP6roSDbhpchOqHAc8G",
    project_name="Stanford Experiments",
    workspace="",
    log_code = True,
    auto_metric_logging=True,
    auto_param_logging=True,
)

#Import statements
import warnings
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch # torch is used to get GPU information

import tensorflow as tf
# import tensorflow_addons as tfa

# Python Imaging Library is used for opening image files
import PIL
import PIL.Image

import pathlib # required to access the image folders

# Import the deep learning models and utility functions

from models import resNet50
from utils import piecewise_constant
from sklearn.model_selection import train_test_split

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
train_dir='/work/netthinker/shared/StanfordDataset/StanfordSubsetCrop/train'
test_dir='/work/netthinker/shared/StanfordDataset/StanfordSubsetCrop/test'


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

train_data_dir = pathlib.Path(train_dir)
train_file_paths = list(train_data_dir.glob('*/*')) # Get the list of all training image file paths
print("\nNumber of training files: ", len(train_file_paths))

test_data_dir = pathlib.Path(test_dir)
test_file_paths = list(test_data_dir.glob('*/*')) # Get the list of all test image file paths
print("Number of test files: ", len(test_file_paths))

print("\nPath of the first training image: ", str(train_file_paths[0]))
PIL.Image.open(str(train_file_paths[0]))

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

train_fnames=[]
for fname in train_file_paths:
    train_fnames.append(str(fname))

print(len(train_fnames))

test_fnames=[]
for fname in test_file_paths:
    test_fnames.append(str(fname))

print(len(test_fnames))

# Create the image file list Dataset (for training & testing images) 
train_dataset = tf.data.Dataset.from_tensor_slices(train_fnames)
test_dataset = tf.data.Dataset.from_tensor_slices(test_fnames)


#_______________________Option 2: "list_flies" method (NOT USED)__________________________

# #Get a training dataset of all files (list of all image paths) matching the glob pattern
# train_dataset = tf.data.Dataset.list_files(str(train_data_dir/'*/*'))
# 
# #Get a test dataset of all files (list of all image paths) matching the glob pattern
# test_dataset = tf.data.Dataset.list_files(str(test_data_dir/'*/*'))


train_dataset_size = train_dataset.cardinality().numpy()
test_dataset_size = test_dataset.cardinality().numpy()

print("\nNumber of samples in training dataset: ", train_dataset_size)
print("Number of samples in test dataset: ", test_dataset_size)



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
CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if item.name != ".DS_Store"])
print("\nClass names:\n", CLASS_NAMES)


'''
Function to get the label of an image (file) from its path and class names
The labels are one-hot encoded
'''
def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory (i.e., name of the class)
  return tf.cast(parts[-2] == CLASS_NAMES, dtype=tf.int32)
  

'''
This function reads a PNG-encoded image into a uint8 tensor
- Converts the uint8 tensor into float32
- Scales the tensor into the range [0,1] automatically while converting to float32
  Thus, we don't need to scale the images separately
- Resizes the image
Note: if the image encoding is other than PNG, then a suitable decode method should be used.
'''   
def process_image(image, image_height=200, image_width=200):  
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

def prepare_dataset(ds, mini_batch, shuffle=False, augment=False, buffer_size=0):
    '''
    Get each file path and name into the Dataset object.
    To do this, we use Dataset's map() method.
    It returns (image,label) pairs.
    The map() method applies the argument function (i.e., load_labeled_data) 
    to each element of the dataset, 
    and returns a new dataset containing the transformed elements (i.e., image, label), 
    in the same order as they appeared in the input.
    '''
    ds = ds.map(load_labeled_data, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache the data
    ds = ds.cache()
    
    # Shuffle the data
    if shuffle:
        ds = ds.shuffle(buffer_size)
      
    # Repeat the shuffled dataset  
    ds = ds.repeat()

    # Batch all datasets
    ds = ds.batch(mini_batch, drop_remainder=True)

    # Use data augmentation only on the training set
    # When training is done using GPUs,
    #    for efficiency, we should perform data augmentation inside the model 
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Repeats the batch so each original value is seen "count" times
    # Increasing "count" will increase the number of steps per epoch
    #ds = ds.repeat(count=?) 
    
    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    
# ****************************** Prepare Dataset for Training ******************************                                  

# Determine the size of mini-batch for training
# It should be a multiple of BATCH_SIZE_PER_REPLICA 
# The multiplication factor is determined by the number of available GPUs (or strategy.num_replicas_in_sync)

BATCH_SIZE_PER_REPLICA = 64

if(num_of_gpu > 0):
    size_of_mini_batch = BATCH_SIZE_PER_REPLICA*num_of_gpu
else:
    size_of_mini_batch = BATCH_SIZE_PER_REPLICA # Uses the CPU, as no GPU is available


# Size of test mini-batch 
size_of_mini_batch_test = size_of_mini_batch

print("\nSize of training mini-batch: ", size_of_mini_batch)
print("Size of test mini-batch: ", size_of_mini_batch_test)


# Used by the "shuffle" method. For small dataset, it should be equal or larger than training set
buffer_size =   train_dataset_size
print("\nBuffer Size: ", buffer_size)



# Perform data preprocessing by the CPU
# It is efficient as it ensures that the GPUs will be used only for model training
with tf.device('/cpu:0'):
    train_loader = prepare_dataset(train_dataset, size_of_mini_batch, shuffle=True, augment=False, buffer_size=buffer_size)
    test_loader = prepare_dataset(test_dataset, size_of_mini_batch_test)
        
    # ****************************** Create Model ******************************

'''
Delete the TensorFlow graph before creating a new model, otherwise memory overflow will occur.
'''
tf.keras.backend.clear_session()

'''
To reproduce the same result by the model in each iteration, we use fixed seeds for random number generation. 
'''
np.random.seed(42)
tf.random.set_seed(42)
    
savedModel = tf.keras.models.load_model('/work/netthinker/ele3/Research/StanfordExperiments/standfordSubsetRL')

savedModel.layers.pop()
for layer in savedModel.layers:
    layer.trainable=False
last = savedModel.layers[-2].output
x = tf.keras.layers.Dense(60, activation="softmax", name='prediction')(last)
finetuned_model = tf.keras.Model(savedModel.input, x)

optimizer = opt = tf.keras.optimizers.Nadam(learning_rate=0.001)
loss_object = tf.keras.losses.CategoricalCrossentropy()
finetuned_model.compile(loss=loss_object,
            optimizer=optimizer,
              metrics=[tf.keras.metrics.CategoricalAccuracy()])    
finetuned_model.summary()

'''
Train the model
'''
print("\nTraining started ...")

no_of_epochs = 300

history1 = finetuned_model.fit(train_loader, 
                    epochs=no_of_epochs,
                    steps_per_epoch= train_dataset_size // size_of_mini_batch,
                    validation_steps= test_dataset_size // size_of_mini_batch_test,
                    verbose=1,
                    validation_data=test_loader)
                    
# ****************************** Test Model ******************************

numOfEpochs = len(history1.history['loss'])
print("Epochs: ", numOfEpochs)


train_acc = history1.history['categorical_accuracy']
val_acc = history1.history['val_categorical_accuracy']


train_loss = history1.history['loss']
val_loss = history1.history['val_loss']

model_name = ""
    
print("\n{} Train Accuracy: {:.3f}".format(model_name, train_acc[numOfEpochs - 1]))
print("{} Train Loss: {:.3f}".format(model_name, train_loss[numOfEpochs - 1]))

print("\n{} Validation Accuracy: {:.3f}".format(model_name, val_acc[numOfEpochs - 1]))
print("{} Validation Loss: {:.3f}".format(model_name, val_loss[numOfEpochs - 1]))