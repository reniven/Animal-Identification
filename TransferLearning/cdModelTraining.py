
# ******************* Deep Learning Experiment: TensorFlow (Keras) *********************


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

from models import resNet50
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

train_ds, info_train = tfds.load('cats_vs_dogs', split='train[:80%]', with_info=True,  as_supervised=True)
test_ds, info_val = tfds.load('cats_vs_dogs', split='train[80%:]', with_info=True,  as_supervised=True)

train_dataset_size = len(train_ds)
test_dataset_size = len(test_ds)

IMG_SIZE = 200

resize_and_rescale = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

# ****************************** Load Data ******************************



def prepare_dataset(ds, mini_batch, epochs, shuffle=False, augment=False, buffer_size=0,  hotEncode = False, num_classes = 0):
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

    '''
    Repeat the shuffled dataset
    '''  
    ds = ds.repeat(count=epochs)
      
    # Repeat the shuffled dataset  
    #ds = ds.repeat()

    # Batch all datasets
    ds = ds.batch(mini_batch, drop_remainder=True)

    # Use data augmentation only on the training set
    # When training is done using GPUs,
    #    for efficiency, we should perform data augmentation inside the model 
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Repeats the batch so each original value is seen "count" times
    # Increasing "count" will increase the number of steps per epoch
    #ds = ds.repeat(2) 
    
    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# ****************************** Prepare Dataset for Training ******************************

# Determine the size of mini-batch for training
# It should be a multiple of BATCH_SIZE_PER_REPLICA 
# The multiplication factor is determined by the number of available GPUs (or strategy.num_replicas_in_sync)

BATCH_SIZE_PER_REPLICA = 64
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
buffer_size =   2000
print("\nBuffer Size: ", buffer_size)



# Perform data preprocessing by the CPU
# It is efficient as it ensures that the GPUs will be used only for model training
with tf.device('/cpu:0'):
    train_loader = prepare_dataset(train_ds, size_of_mini_batch, 300, shuffle=True, augment=False, buffer_size=buffer_size, hotEncode=True, num_classes=2)
    test_loader = prepare_dataset(test_ds, size_of_mini_batch_test, 300, hotEncode=True, num_classes=2)
    no_of_steps_per_epoch = train_dataset_size//size_of_mini_batch
    total_no_of_steps = train_loader.cardinality().numpy()
    print("Data available to run for %d epochs" % (total_no_of_steps//no_of_steps_per_epoch))
    print("Unlimited data available: ", (train_loader.cardinality() == tf.data.INFINITE_CARDINALITY).numpy())
    



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
    model = resNet50((IMG_SIZE, IMG_SIZE, 3), 2,  augmentation = True,
                        resize = True,
                        increased_size = 250,
                        random_crop = True,
                        original_size = 200,
                        rotation_angle_degree = 20,
                        zoom_height_factor = 0.6,
                        gamma_proba = 0.8,
                        cutout_proba = 0.8,
                        mask_size = 60,
                        random_flip = True)
    model.summary()
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_object,
              optimizer=optimizer,
              metrics=[tf.keras.metrics.CategoricalAccuracy()])        
        
	


        
# ****************************** Train Model ******************************

model_name = "cdModel"

'''
The file path to the Saved_Models diretory in which the best model will be serialized
'''

ROOT_DIR = os.path.abspath(os.curdir)

saved_model_directory = os.path.join(ROOT_DIR, 'Saved_Models')
saved_model_path = os.path.join(saved_model_directory, model_name)

'''
Create model checkpoint "callback" object to save only the best performing models
'''
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=saved_model_path, save_best_only=True)

'''
Create a TensorBoard "callback" object for the learning schedule
'''
lschedule_cb = tf.keras.callbacks.LearningRateScheduler(piecewise_constant, verbose=0) # define the piecewise_constant function first


'''
Train the model
'''
print("\nTraining started ...")

no_of_epochs = 300

history = model.fit(train_loader, 
                    epochs=no_of_epochs,
                    steps_per_epoch=train_dataset_size // size_of_mini_batch,
                    validation_steps=test_dataset_size // size_of_mini_batch_test,
                    verbose=1,
                    validation_data=test_loader,
                    callbacks=[lschedule_cb])


'''
Save the model
'''
model.save(model_name)






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