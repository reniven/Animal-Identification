# Use comet ml to track the progress of the training and see training statistics
from comet_ml import Experiment

# Create an experiment with the api key:
record = False

if(record):
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
SD_train_dataset = tf.data.Dataset.from_tensor_slices(SD_train_fnames)
SD_test_dataset = tf.data.Dataset.from_tensor_slices(SD_test_fnames)
SD_val_dataset = tf.data.Dataset.from_tensor_slices(SD_val_fnames)

SD_train_dataset_size = SD_train_dataset.cardinality().numpy()
SD_test_dataset_size = SD_test_dataset.cardinality().numpy()
SD_val_dataset_size = SD_val_dataset.cardinality().numpy()

print("\nNumber of samples in training dataset: ", SD_train_dataset_size)
print("Number of samples in test dataset: ", SD_test_dataset_size)
print("Number of samples in validation dataset: ", SD_val_dataset_size)

# Create the image file list Dataset (for SDL training & testing images) 
SDL_train_dataset = tf.data.Dataset.from_tensor_slices(SDL_train_fnames)
SDL_test_dataset = tf.data.Dataset.from_tensor_slices(SDL_test_fnames)
SDL_val_dataset = tf.data.Dataset.from_tensor_slices(SDL_val_fnames)

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
print("\nSD Class Length:\n", str(len(SD_CLASS_NAMES)))

SDL_CLASS_NAMES = np.array([item.name for item in SDL_train_data_dir.glob('*') if item.name != ".DS_Store"])
print("\nSDL Class names:\n", str(len(SDL_CLASS_NAMES)))


'''
Function to get the label of an image (file) from its path and class names
The labels are one-hot encoded
'''
def get_label(file_path, classes):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory (i.e., name of the class)
  return tf.cast(parts[-2] == classes, dtype=tf.int32)
  

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
def load_labeled_data(file_path, classes):
    img = tf.io.read_file(file_path)
    img = process_image(img)
    label = get_label(file_path, classes)
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

def prepare_dataset(ds, classes):
    
    print("Length of Initial Dataset")
    print(len(ds))
    images = []
    labels = []
    
    for img in ds:
        image, label = load_labeled_data(img, classes)
        images.append(image)
        labels.append(label)


    print("Length of Training Images: " + str(len(images)))
    print("Length of Training Labels: " + str(len(labels)))
    print()
    return images, labels



# ****************************** Prepare Dataset for Training ******************************                                  


# Perform data preprocessing by the CPU
# It is efficient as it ensures that the GPUs will be used only for model training
print("hello")
print("Loading Images")
print("Loading SD Images")
SD_train_images, SD_train_labels = prepare_dataset(SD_train_dataset, SD_CLASS_NAMES)
SD_test_images, SD_test_labels = prepare_dataset(SD_test_dataset, SD_CLASS_NAMES)
SD_val_images, SD_val_labels = prepare_dataset(SD_val_dataset, SD_CLASS_NAMES)
    
print("Length of SD Image Sets")
print("SD Train: " + str(len(SD_train_images)))
print("SD Test: " + str(len(SD_test_images)))
print("SD Val: " + str(len(SD_val_images)))
    
print("Loading SDL Images")
SDL_train_images, SDL_train_labels = prepare_dataset(SDL_train_dataset, SDL_CLASS_NAMES)
SDL_test_images, SDL_test_labels = prepare_dataset(SDL_test_dataset, SDL_CLASS_NAMES)
SDL_val_images, SDL_val_labels = prepare_dataset(SDL_val_dataset, SDL_CLASS_NAMES)

print("Length of SDL Image Sets")
print("SDL Train: " + str(len(SDL_train_images)))
print("SDL Test: " + str(len(SDL_test_images)))
print("SDL Val: " + str(len(SDL_val_images)))
    
print("Filling SDL Images")
SDL_train_images = np.append(SDL_train_images, SDL_train_images[:3510], axis = 0)
SDL_train_labels = np.append(SDL_train_labels, SDL_train_labels[:3510], axis = 0)
SDL_val_images = np.append(SDL_val_images, SDL_val_images[:390], axis = 0)
SDL_val_labels = np.append(SDL_val_labels, SDL_val_labels[:390], axis = 0)

print("Length of New SDL Image Sets")
print("SDL Train Image Set: " + str(len(SDL_train_images)))
print("SDL Train Label Set: " + str(len(SDL_train_labels)))
print("SDL Test Image Set: " + str(len(SDL_test_images)))
print("SDL Test Label Set: " + str(len(SDL_test_labels)))
print("SDL Val Image Set: " + str(len(SDL_val_images)))
print("SDL Val Label Set: " + str(len(SDL_val_labels)))

#Convert datasets to numpy arrays
SD_train_images = np.array(SD_train_images, np.float32)
SDL_train_images = np.array(SDL_train_images, np.float32)
SD_train_labels = np.array(SD_train_labels, np.float32)
SDL_train_labels = np.array(SDL_train_labels, np.float32)
SD_val_images = np.array(SD_val_images, np.float32)
SD_val_labels = np.array(SD_val_labels, np.float32)
SDL_val_images = np.array(SDL_val_images, np.float32)
SDL_val_labels = np.array(SDL_val_labels, np.float32)

original_size = 100
increased_size = 120
zoom_height_factor = 0.5 # This value may vary depending on the level of required augmentation
rotation_factor = 0.1 # This value may vary depending on the level of required augmentation


'''
Following augmentations are performed.
- Resize (increase the size)
- Random zoom
- Random rotation
- Random horizontal flip
- Random crop (restore the original size)
'''  

# Create a tf.Keras "Layer" for data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(increased_size, increased_size),
    tf.keras.layers.experimental.preprocessing.RandomZoom(zoom_height_factor, fill_mode='nearest'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(rotation_factor, fill_mode='nearest'),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomCrop(original_size, original_size),
])

class Residual_Block(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        
        '''
        Add two Conv layers
            - After the first Conv layer perform batch normalization (BN) and ReLU activation
            - After the second Conv layer, perform only BN

        Don't use bias neurons in the Conv layers (set the use_bias to False).
        Because the Conv layer is followed by a BN layer that adds a bias.
        The BN "shift" parameter shifts the output of the layer (thus acts like a bias). 
        '''
        self.main_layers = [
            # The stride of the first Conv layer is specified by the designer: could be 1 or 2
            tf.keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding="SAME", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            # The stride of the second Conv layer is always 1
            tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding="SAME", use_bias=False),
            tf.keras.layers.BatchNormalization()]
        self.skip_layers = []
        # If the stride of the first Conv layer is 2,
        #    then add a 1 x 1 Conv layer on the skip channel, followed by batch normalization
        if strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, kernel_size=1, strides=strides, padding="SAME", use_bias=False),
                tf.keras.layers.BatchNormalization()]
                
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
    # Required for the custom object's serialization
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "conv_layers": self.main_layers,
            "skip_layers": self.skip_layers
        })
        return config

def resNet50(input_shape, augmentation = False, **kwargs):
    
    resnet = tf.keras.models.Sequential(name='ResNet-50')
    
    resnet.add(tf.keras.layers.InputLayer(input_shape=input_shape))
     # Data augmentation layer
    if(augmentation):
        resnet.add(data_augmentation)
    
    '''
    Set the use_bias to False because the following BN layer adds a bias. 
    The BN "shift" parameter shifts the output of the layer (thus acts like a bias).
    '''
    resnet.add(tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding='same', 
                     input_shape=input_shape, use_bias=False))
    resnet.add(tf.keras.layers.BatchNormalization())
    resnet.add(tf.keras.layers.Activation("relu"))
    resnet.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))

    # Add the residual blocks
    prev_filters = 64
    for filters in [64] * 6 + [128] * 6 + [256] * 6 + [512] * 6:
        strides = 1 if filters == prev_filters else 2
        resnet.add(Residual_Block(filters, strides=strides))
        prev_filters = filters

    # Perform global average pooling
    resnet.add(tf.keras.layers.GlobalAvgPool2D())
    resnet.add(tf.keras.layers.Flatten())

    return resnet

def resNet32(input_shape, augmentation = False, **kwargs):
    
    resnet = tf.keras.models.Sequential(name='ResNet-50')
    
    resnet.add(tf.keras.layers.InputLayer(input_shape=input_shape))
     # Data augmentation layer
    if(augmentation):
        resnet.add(data_augmentation)
    
    '''
    Set the use_bias to False because the following BN layer adds a bias. 
    The BN "shift" parameter shifts the output of the layer (thus acts like a bias).
    '''
    resnet.add(tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding='same', 
                     input_shape=input_shape, use_bias=False))
    resnet.add(tf.keras.layers.BatchNormalization())
    resnet.add(tf.keras.layers.Activation("relu"))
    resnet.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))

    # Add the residual blocks
    prev_filters = 64
    for filters in [64] * 5 + [128] * 5 + [256] * 5 :
        strides = 1 if filters == prev_filters else 2
        resnet.add(Residual_Block(filters, strides=strides))
        prev_filters = filters

    # Perform global average pooling
    resnet.add(tf.keras.layers.GlobalAvgPool2D())
    resnet.add(tf.keras.layers.Flatten())

    return resnet



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
Create a Functional model. 
- Input Layer 1 (input1): It instantiates an input tensor for buildng the model 
- Branch 1: Two Hidden Layers: Dense hidden layer with the ReLU activation function

- Input Layer 2 (input2): It instantiates an input tensor for buildng the model 
- Branch 2: Two Hidden Layers: Dense hidden layer with the ReLU activation function
'''

# Open a strategy scope to create and compile the model
with strategy.scope():

    resnet50 = resNet50(input_shape=(100, 100, 3), augmentation=True)
    resnet32 = resNet50(input_shape=(100, 100, 3), augmentation=True)


    '''
    Combine the final hidden layers of the two parallel branches
    '''
    combined_hidden = tf.keras.layers.concatenate([resnet50.outputs[0], resnet32.outputs[0]], axis = 1)

    '''
    The combined hidden layer passes the signal to two output layers 
    (for multiclass and binary classification, respectively).
    - Output Layer 1: Dense output layer with 10 neurons. Since it's a multi-class classification, we use "softmax"  
    - Output Layer 2: Dense output layer with 2 neurons. Since it's a binary classification, we use "sigmoid"  
    '''

    denseLayer = tf.keras.layers.Dense(300)(combined_hidden)
    denseLayer2 = tf.keras.layers.Dense(300)(combined_hidden)
    #denseLayer2 = tf.keras.layers.Dense(300)(denseLayer)
    output1 = tf.keras.layers.Dense(120, activation="softmax")(denseLayer)
    output2 = tf.keras.layers.Dense(120, activation="softmax")(denseLayer2)

    # Create a Model by specifying its input and outputs
    model = tf.keras.models.Model(inputs=[resnet50.input, resnet32.input], outputs=[output1, output2])


    # Display the model summary
    model.summary()

    # Display the model graph
    tf.keras.utils.plot_model(model, show_shapes=True)

    # Define the optimizer
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1, momentum=0.1)


    '''
    Compile the model.
    Since we are using two different types of loss functions, we specify those using a list.
    '''
    model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"],
              optimizer=optimizer,
              metrics=["accuracy"])

    # Create a callback object of early stopping
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                  min_delta=0, 
                                  patience=10, 
                                  verbose=1, 
                                  mode='auto',
                                  restore_best_weights=True)

lschedule_cb = tf.keras.callbacks.LearningRateScheduler(piecewise_constant, verbose=0) # define the piecewise_constant function first

# ****************************** Train Model ******************************

model_name = "multioutput"

ROOT_DIR = os.path.abspath(os.curdir)

saved_model_directory = os.path.join(ROOT_DIR, 'Saved_Models')
saved_model_path = os.path.join(saved_model_directory, model_name)

'''
Create model checkpoint "callback" object to save only the best performing models
'''
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=saved_model_path, save_best_only=True)

lschedule_cb = tf.keras.callbacks.LearningRateScheduler(piecewise_constant, verbose=0) # define the piecewise_constant function first

'''
Train the model.
We need to specify
- two inputs (train & validation) for passing through two parallel branches
- two types of labels for training and validation using lists
'''
history = model.fit([SD_train_images, SD_train_images], [SD_train_labels, SD_train_labels],
                    epochs=300,
                    batch_size = 64,
                    verbose=1,
                    validation_data= ([SD_val_images, SD_val_images], [SD_val_labels, SD_val_labels]),
                    callbacks=[lschedule_cb])

#validation_data= ([np.array(SD_val_images, np.float32), np.array(SDL_val_images, np.float32)], [np.array(SD_val_labels, np.float32), np.array(SDL_val_labels, np.float32)]),
# For evaluation, we need to pass two input matrices for the two parallel branches in the model
numOfEpochs = len(history.history['loss'])
print("Epochs: ", numOfEpochs)

# For evaluation, we need to pass two input matrices for the two parallel branches in the model
train_evaluation = model.evaluate([SD_train_images, SDL_train_images], [SD_train_labels, SDL_train_labels], verbose=0)
test_evaluation = model.evaluate([SD_test_images, SDL_test_images], [SD_test_labels, SDL_test_labels], verbose=0)

print("\nTrain Evaluation: ", train_evaluation)
print("Test Evaluation: ", test_evaluation)


train_loss_multiclass = train_evaluation[1]
train_loss_binary = train_evaluation[2]
train_accuracy_multiclass = train_evaluation[3]
train_accuracy_binary = train_evaluation[4]

test_loss_multiclass = test_evaluation[1]
test_loss_binary = test_evaluation[2]
test_accuracy_multiclass = test_evaluation[3]
test_accuracy_binary = test_evaluation[4]


print("\n******************** Original Classification ********************************************")


print("\nOriginal Classification - Train Accuracy: ", train_accuracy_multiclass)
print("Original Classification - Test Accuracy: ", test_accuracy_multiclass)

print("\nOriginal Classification - Train Loss: ", train_loss_multiclass)
print("Original Classification - Test Loss: ", test_loss_multiclass)

# For evaluation, we need to pass two input matrices for the two parallel branches in the model
# model.predict(X_test) method return 10 probabilities per class for each instance (Dimension Nx10)
y_test_predicted = model.predict([SD_test_images, SDL_test_images])
y_test_predicted_multiclass = np.argmax(y_test_predicted[0], axis=1) # get the label/index of the highest probability class
y_test_predicted_binary = np.argmax(y_test_predicted[1], axis=1) # get the label/index of the highest probability class



# model.predict_classes(X_test) method returns the index (class label) with largest probability (1D array)
#y_test_predicted= model.predict_classes(X_test)


# For evaluation, we need to pass two input matrices for the two parallel branches in the model
y_train_predicted = model.predict([SD_train_images, SDL_train_images])
y_train_predicted_multiclass = np.argmax(y_train_predicted[0], axis=1) # get the label/index of the highest probability class
y_train_predicted_binary = np.argmax(y_train_predicted[1], axis=1) # get the label/index of the highest probability class


print("\nTest Confusion Matrix (Original):")
print(confusion_matrix(SD_test_labels, y_test_predicted_multiclass))

print("\nClassification Report (Original):")
print(classification_report(SD_test_labels, y_test_predicted_multiclass))



print("\n******************** Subset 3 Classification ********************************************")



print("\nSubset 3 Classification - Train Accuracy: ", train_accuracy_binary)
print("Subset 3 Classification - Test Accuracy: ", test_accuracy_binary)

print("\Subset 3 Classification - Train Loss: ", train_loss_binary)
print("Subset 3 Classification - Test Loss: ", test_loss_binary)


print("\nTest Confusion Matrix (Subset 3):")
print(confusion_matrix(SDL_test_labels, y_test_predicted_binary))

print("\nClassification Report (Subset 3):")
print(classification_report(SDL_test_labels, y_test_predicted_binary))