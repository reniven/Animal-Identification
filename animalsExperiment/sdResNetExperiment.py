
# ******************* Deep Learning Experiment: TensorFlow (Keras) *********************

'''
This python program uses TensorFlow (Keras) for training a Deep Learning model 
to perform image classification.
It uses computing resources (e.g., GPU) from a HCC cluster (e.g., crane),

When multiple GPUs are available, it uses the distributed training technique.
- It utilizes multiple GPUs on a single node as instructed by the SLURM .sh script.
- For distributed processing, the data needs to be stored in the TensorFlow "Dataset" format.

More information on distributed training: 
https://keras.io/guides/distributed_training/
'''


# *************************************** Dataset **************************************
'''
This program assumes that the dataset (i.e., images) is stored locally 
and structured in nested directories as follows:
- train
   - class_name_1
   - class_name_2
   ...
- test
   - class_name_1
   - class_name_2
   ...

Specifically, there should be two root directories named "train" and "test".
The sub-directories should be named after the class.

The tf.Data API is used to load data as a Dataset object 
from local directories as (image, label) pairs. 

'''

# *******Distributed Training: Single-host & Multi-device Synchronous Training************

'''
We use the tf.distribute API to train a TensorFlow Keras model on multiple GPUs 
installed on a single machine (node). 

Specifically, we use the tf.distribute.Strategy with tf.keras
The tf.distribute.Strategy is integrated into tf.keras, which is a high-level API to build and train models. 
By integrating into tf.keras backend, it is seamless for use 
to distribute the training written in the Keras training framework.

More information on the distributed training with Keras:
https://www.tensorflow.org/tutorials/distribute/keras


Parallelism Technique:
Via the tf.distribute API, we implement the synchronous data parallelism technique. 
In this technique, a single model gets replicated on multiple devices or multiple machines. 
Each of them processes different batches of data, then they merge their results. 
The different replicas of the model stay in sync after each batch they process. 
Synchronicity keeps the model convergence behavior identical 
to what you would see for single-device training.


How to use the the tf.distribute API for distributed training:
To do single-host, multi-device synchronous training with a TensorFlow Keras model, 
we use the tf.distribute.MirroredStrategy API. 

Following are the 3 simple steps.

- Instantiate a MirroredStrategy.
By default, the strategy will use all GPUs available on a single machine.

- Use the strategy object to open a scope, and within this scope, 
create all the Keras objects you need that contain variables. 
More specifically, within the distribution scope:
- Create the model
- Compile the model (by defining the optimizer and metrics)

- Train the model via the fit() method as usual (outside the scope).

Note: we need to use tf.data.Dataset objects to load data in a multi-device 
or distributed workflow.
'''

# Use comet ml to track the progress of the training and see training statistics
from comet_ml import Experiment

# Create an experiment with the api key:
experiment = Experiment(
    api_key="d1doFSsP6roSDbhpchOqHAc8G",
    project_name="ResNet-32 Experiments - 06/30/2021",
    workspace="reniven",
    log_code = True,
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
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from utils import data_augmentation_layer

# Python Imaging Library is used for opening image files
import PIL
import PIL.Image

import pathlib # required to access the image folders

# Import the deep learning models and utility functions

from models import resNet32
from utils import piecewise_constant

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



# ****************************** Load Data ******************************

(train_ds, test_ds), metadata  = tfds.load('stanford_dogs', split = ['train', 'test'], with_info=True,
    as_supervised=True)

train_dataset_size = len(train_ds)
test_dataset_size = len(test_ds)

rescale = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

IMG_SIZE = 150

resize_and_rescale = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

def prepare_dataset(ds, mini_batch, shuffle=False, augment=False, buffer_size=0,  hotEncode = False, num_classes = 0):
    '''
    Get each file path and name into the Dataset object.
    To do this, we use Dataset's map() method.
    It returns (image,label) pairs.
    The map() method applies the argument function (i.e., load_labeled_data) 
    to each element of the dataset, 
    and returns a new dataset containing the transformed elements (i.e., image, label), 
    in the same order as they appeared in the input.
    '''
    if hotEncode:
        ds = ds.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)),
              num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
              num_parallel_calls=tf.data.AUTOTUNE)

    # Cache the data
    ds = ds.cache()
    
    # Shuffle the data
    if shuffle:
        ds = ds.shuffle(buffer_size)
      
    # Repeat the shuffled dataset  
    ds = ds.repeat(2)

    # Batch all datasets
    ds = ds.batch(mini_batch, drop_remainder=True)

    # Use data augmentation only on the training set
    # When training is done using GPUs,
    #    for efficiency, we should perform data augmentation inside the model 
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Repeats the batch so each original value is seen "count" times
    # Increasing "count" will increase the number of steps per epoch
    #ds = ds.repeat(count=?) 
    
    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Function to scale and batch test data
def prepare_test(ds, hotEncode = False, num_classes = 0):
    
    if hotEncode:
        ds = ds.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)),
              num_parallel_calls=tf.data.AUTOTUNE)
    # Rescale the data
    ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch all datasets
    ds = ds.batch(size_of_mini_batch_test, drop_remainder=True)
    
    return ds

# ****************************** Prepare Dataset for Training ******************************

# Determine the size of mini-batch for training
# It should be a multiple of BATCH_SIZE_PER_REPLICA 
# The multiplication factor is determined by the number of available GPUs (or strategy.num_replicas_in_sync)

BATCH_SIZE_PER_REPLICA = 512
experiment.log_parameter("batch_size", BATCH_SIZE_PER_REPLICA)

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
    train_loader = prepare_dataset(train_ds, size_of_mini_batch, shuffle=True, augment=False, buffer_size=buffer_size, hotEncode=True, num_classes=120)
    test_loader = prepare_dataset(test_ds, size_of_mini_batch_test, hotEncode=True, num_classes=120)
    



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


'''
Define the optimizer
'''
optimizer = opt = tf.keras.optimizers.Nadam(learning_rate=0.001)




# Open a strategy scope to create and compile the model
with strategy.scope():
    
    # When you create the model, make sure to do data augmentation inside it
    model = resNet32((IMG_SIZE, IMG_SIZE, 3), 120,         
               resize = True,
               random_crop = True,
               increased_size = 160,
               original_size = 150,
               zoom_height_factor =0.6,
               rotation_angle_degree = 20,
               random_flip = True,
               brightness_proba = 0.8,
               brightness_factor = 0.5,
               translation_height_factor = 0,
               translation_width_factor = 0,
               contrast_proba=1.0,
               contrast_factor=0,
               gamma_proba = 0.8,
               hue_proba =0.8,
               blur_proba = 0.8,
               gaussian_filter_shape = 10,
               gaussian_sigma = 0.5,
               cutout_proba = 0.8,
               mask_size=10)
    model.summary()
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss_object,
              optimizer=optimizer,
              metrics=[tf.keras.metrics.CategoricalAccuracy()])        
        
	


        
# ****************************** Train Model ******************************

model_name = ""
model_name_format = ""


'''
Create model checkpoint "callback" object to save only the best performing models
'''
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_name_format, save_best_only=True)


'''
Create a TensorBoard "callback" object for the learning schedule
'''
#lschedule_cb = tf.keras.callbacks.LearningRateScheduler(piecewise_constant, verbose=0) # define the piecewise_constant function first
#lschedule_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=10, min_lr=0.000001, mode='auto', verbose=1)


'''
Train the model
'''
print("\nTraining started ...")

no_of_epochs = 200

history = model.fit(train_ds, 
                    epochs=no_of_epochs,
                    steps_per_epoch=train_dataset_size // size_of_mini_batch,
                    validation_steps=test_dataset_size // size_of_mini_batch_test,
                    verbose=1,
                    validation_data=train_ds)


'''
Save the model
'''
#model.save(model_name_format)






# ****************************** Test Model ******************************

# Load the saved model for making predictions
#model = tf.keras.models.load_model(model_name_format)

numOfEpochs = len(history.history['loss'])
print("Epochs: ", numOfEpochs)


train_acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']


train_loss = history.history['loss']
val_loss = history.history['val_loss']

print("\n{} Train Accuracy: {:.3f}".format(model_name, train_acc[numOfEpochs - 1]))
print("{} Train Loss: {:.3f}".format(model_name, train_loss[numOfEpochs - 1]))

print("\n{} Validation Accuracy: {:.3f}".format(model_name, val_acc[numOfEpochs - 1]))
print("{} Validation Loss: {:.3f}".format(model_name, val_loss[numOfEpochs - 1]))