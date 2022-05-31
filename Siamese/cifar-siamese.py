import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import tensorflow as tf
import tensorflow_addons as tfa

# Load the MNIST dataset (train and test subsets)
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Cast the datatype ofthe features
X_train_full = X_train_full.astype('float32')
X_test = X_test.astype('float32')



# Create validation dataset as well as scale the data
X_valid, X_train = X_train_full[:10000] / 255.0, X_train_full[10000:] / 255.0
y_valid, y_train = y_train_full[:10000], y_train_full[10000:]
X_test = X_test / 255.0


print("No. of Training Samples: ", X_train.shape)
print("No. of Training Labels: ", y_train.shape)

print("\nNo. of Validation Samples: ", X_valid.shape)
print("No. of Validation Labels: ", y_valid.shape)

print("\nNo. of Testing Samples: ", X_test.shape)
print("No. of Testing Labels: ", y_test.shape)

print("\nX type: ", X_train.dtype)
print("y type: ", y_train.dtype)

# Get the shape of the samples
input_shape = X_train.shape[1:]
print("\nInput Shape: ", input_shape)

'''
This function rotates an image randomly by 
either degree "rotation_angle_degree" or "rotation_angle_degree"+90
It returns the rotated image and its "label"
    label = 0: rotation degree = "rotation_angle_degree"
    label = 1: rotation degree = "rotation_angle_degree" + 90
'''
def rotate_randomly(x, rotation_angle_degree):
    
    rand_val = np.random.random_sample()
    label = -1
    if(rand_val < 0.9 and rand_val > 0.7):
      rotation_angle_radian = 0 * (3.1416/ 180)
      x = tfa.image.rotate(x, rotation_angle_radian, fill_mode='nearest', interpolation='bilinear')
      label = 0
    elif(rand_val < 0.7 and rand_val > 0.5):
      rotation_angle_radian = rotation_angle_degree * (3.1416/ 180)
      x = tfa.image.rotate(x, rotation_angle_radian, fill_mode='nearest', interpolation='bilinear')
      label = 1
    elif(rand_val < 0.5 and rand_val > 0.3):
      rotation_angle_radian = (rotation_angle_degree  + 90) * (3.1416/ 180)
      x = tfa.image.rotate(x, rotation_angle_radian, fill_mode='nearest', interpolation='bilinear')
      label = 2
    else:
      rotation_angle_radian = (rotation_angle_degree + 180) * (3.1416/ 180)
      x = tfa.image.rotate(x, rotation_angle_radian, fill_mode='nearest', interpolation='bilinear')
      label = 3
    return x, label



'''
This function takes a data marix X and returns another matrix X_rot of the same shape
X_rot contains rotated versions (by 90 degree or 180 degree) of the images in X
It also returns a label matrix, which assigns 0/1 to each rotated image in X_rot as follows.
    label = 0; if rotation = 90 degree
    label = 1; if rotation = 180 degree
'''
def create_rotated_images_labels(X):
    
    '''
    Create a data matrix to store rotated version of the samples in X 
    '''
    X_rot = X.copy()
    
    '''
    Create a binary label matrix to store the labels of the rotated samples
    in X_rot
    '''
    y_rot = np.zeros((X_rot.shape[0], 4))
    
    
    for i in range(X.shape[0]):
        '''
        Rotate an image randomly by either degree 90 or 180
        label = 0; if rotation = 90 degree
        label = 1; if rotation = 180 degree
        '''
        rotated_image, label = rotate_randomly(X[i], 90)

        # Store the rotated image in X_rot
        X_rot[i] = rotated_image

        # Store the label for the rotated image in y_rot
        if(label == 0):
            y_rot[i][0] = 1
        elif(label == 1):
            y_rot[i][1] = 1
        elif(label == 2):
            y_rot[i][2] = 1
        else:
            y_rot[i][3] = 1
            
    return X_rot, y_rot

def flip_randomly(x, rotation_angle_degree):
    
    rand_val = np.random.random_sample()
    label = -1
    if(rand_val < 0.9 and rand_val > 0.7):
      rotation_angle_radian = 0 * (3.1416/ 180)
      x = tfa.image.rotate(x, rotation_angle_radian, fill_mode='nearest', interpolation='bilinear')
      label = 0
    elif(rand_val < 0.7 and rand_val > 0.5):
      rotation_angle_radian = rotation_angle_degree * (3.1416/ 180)
      x = tfa.image.rotate(x, rotation_angle_radian, fill_mode='nearest', interpolation='bilinear')
      label = 1
    elif(rand_val < 0.5 and rand_val > 0.3):
      rotation_angle_radian = (rotation_angle_degree  + 90) * (3.1416/ 180)
      x = tfa.image.rotate(x, rotation_angle_radian, fill_mode='nearest', interpolation='bilinear')
      label = 2
    else:
      rotation_angle_radian = (rotation_angle_degree + 180) * (3.1416/ 180)
      x = tfa.image.rotate(x, rotation_angle_radian, fill_mode='nearest', interpolation='bilinear')
      label = 3
    return x, label

'''
Create data & label matrices of rotated images for train, test, and validation samples
'''
X_train_rot, y_train_rot = create_rotated_images_labels(X_train)
X_test_rot, y_test_rot = create_rotated_images_labels(X_test)
X_valid_rot, y_valid_rot = create_rotated_images_labels(X_valid)

'''
This function creates pairs of similar and dissimilar images & assigns new labels
'''   
def create_similar_dissimilar_pairs(X_rot, y_rot):
    '''
    Shuffling will done on the last 50% samples
    Get the begin and end index for sample shuffling
    '''
    index_begin = X_rot.shape[0] // 2
    index_end = X_rot.shape[0]    

    #print(index_begin)
    #print(index_end)

    '''
    Get a set of indices (from the last 50% samples) 
    that will be used to shuffle the corresponding samples
    '''
    indices = np.arange(index_begin, index_end)

    '''
    Shuffle the idices of last 50% samples
    '''
    np.random.shuffle(indices)


    '''
    Create a data and label matrix to store both the unshuffled
    and shuffled samples
    '''
    X_rot_suffled = X_rot.copy()
    y_rot_suffled = y_rot.copy()

    '''
    Create a label list as follows.
    - Label = 1: sample is unshuffled
    - Label = 0: sample is shuffled
    '''
    y_distance = np.ones(X_rot.shape[0])


    '''
    Using the shuffled indices for the last 50% samples,
    create a new data and label matrix that store both unshuffled 
    and shuffled samples and their labels
    Also update the label list to set whether a sample is shuffled or unshuffled
    '''
    for i in range(X_rot.shape[0]):
        if(i < index_begin):
            X_rot_suffled[i] = X_rot[i]
            y_rot_suffled[i] = y_rot[i]

        else:
            X_rot_suffled[i] = X_rot[indices[i-index_begin]]
            y_rot_suffled[i] = y_rot[indices[i-index_begin]]
            y_distance[i] = 0
            
    return X_rot_suffled, y_rot_suffled, y_distance 

    '''
Create data and label matrices for storing the rotated images (last 50% images are shuffled)
'''
X_train_rot_shuffled, y_train_rot_shuffled, y_train_distance = create_similar_dissimilar_pairs(X_train_rot, y_train_rot)
X_test_rot_shuffled, y_test_rot_shuffled, y_test_distance = create_similar_dissimilar_pairs(X_test_rot, y_test_rot)
X_valid_rot_shuffled, y_valid_rot_shuffled, y_valid_distance = create_similar_dissimilar_pairs(X_valid_rot, y_valid_rot)


'''
Visualize an original image and its rotated version
For the first half of the dataset, an image will have its rotated version at the same index.
'''

index = 4989
image_original = X_valid[index]
image_rotated = X_valid_rot_shuffled[index]

plt.figure(figsize=(2,2))
plt.imshow(image_original, interpolation="nearest", cmap="Greys")
plt.xticks(())
plt.yticks(())
plt.show()

plt.figure(figsize=(2,2))
plt.imshow(image_rotated, interpolation="nearest", cmap="Greys")
plt.xticks(())
plt.yticks(())
plt.show()

def piecewise_constant(epoch):
  #initial_lrate = lr
  #print(initial_lrate)
  if(epoch < 50):
    lrate = 1e-1
  elif(epoch < 100):
    lrate = 1e-1*0.1
  elif(epoch < 130):
    lrate = 1e-1*0.01
  else:
    lrate = 1e-1*0.001
  return lrate

lschedule_cb = tf.keras.callbacks.LearningRateScheduler(piecewise_constant, verbose=0) # define the piecewise_constant function first

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
y_valid = tf.keras.utils.to_categorical(y_valid)

print("Shape of the training labels: ", y_train.shape)

'''
This function creates a base MLP network (shared layers) for learning shared representations
It includes two Dense layers with 300 and 100 neurons, respectively
'''
def create_base_network(input_shape):
    input_ = tf.keras.Input(shape=input_shape)
    
    # Since the network is MLP, we need to flatten the input data
    input_flatten = tf.keras.layers.Flatten()(input_)
    hidden1 = tf.keras.layers.Dense(300, activation="relu")(input_flatten)
    hidden2 = tf.keras.layers.Dense(100, activation="relu")(hidden1)
    hidden2 = tf.keras.layers.Dropout(0.4)(hidden2)
    return tf.keras.models.Model(input_, hidden2)


'''
This function creates a base CNN (shallow) network (shared layers) for learning shared representations
'''
def create_base_network_cnn_shallow(input_shape):
    
    input_shape = (32, 32, 3)
    
    initializer = 'he_normal'
    activation_func = 'relu'
    
    input_ = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer=initializer, padding='same', use_bias=False)(input_)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_func)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer=initializer, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_func)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_func)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    return tf.keras.models.Model(input_, x)



'''
This function creates a base CNN (deep) network (shared layers) for learning shared representations
'''
def create_base_network_cnn_deep(input_shape):
    
    input_ = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer=initializer, padding='same', use_bias=False)(input_)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_func)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer=initializer, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_func)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    
    x = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer=initializer, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_func)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_func)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    return tf.keras.models.Model(input_, x)



'''
This function computes the distance between two representations (feature vectors)
'''
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

'''
This function defines a contrastive loss function using the following equation.
Contrastive loss = mean((1-true_value) * square(prediction) + true_value * square( max(margin-prediction, 0) ))
'''
def contrastive_loss(y_true, y_pred):            
    margin = 1
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
    return tf.math.reduce_mean(
        (1 - y_true) * square_pred + (y_true) * margin_square
    )

'''
Delete the TensorFlow graph before creating a new model, otherwise memory overflow will occur.
'''
tf.keras.backend.clear_session()

'''
To reproduce the same result by the model in each iteration, we use fixed seeds for random number generation. 
'''
np.random.seed(42)
tf.random.set_seed(42)

input_shape = (32, 32, 3)

'''
Create the base network for learning shared representations
'''
base_network = create_base_network_cnn_shallow(input_shape)


'''
Define the shape of the input for both tasks
'''
input_a = tf.keras.Input(shape=input_shape)
input_b = tf.keras.Input(shape=input_shape)

'''
Pass both inputs through the base network
It will create shared reprsentations
'''
processed_a = base_network(input_a)
processed_b = base_network(input_b)



'''
Add two task-specific Dense layers for the two inputs
'''
projection_head_1 = tf.keras.layers.Dense(50, activation="relu")(processed_a)
projection_head_2 = tf.keras.layers.Dense(50, activation="relu")(processed_b)

'''
Add a Lambda layer to compute the Euclidean distance between the representations of the input pairs
at the task-specific layer
'''
distance = tf.keras.layers.Lambda(euclidean_distance, output_shape=None)([projection_head_1, projection_head_2])


'''
Add a classification layer
'''
output1 = tf.keras.layers.Dense(10, activation="softmax")(projection_head_1)
output2 = tf.keras.layers.Dense(4, activation="softmax")(projection_head_2)


'''
Create a Model by specifying its input and outputs
'''
model_1 = tf.keras.models.Model(inputs=[input_a, input_b], outputs=[distance, output1, output2], name="Experiment-1")


'''
Display the model summary
'''
model_1.summary()

'''
Display the model graph
'''
#tf.keras.utils.plot_model(model_1, show_shapes=True)

'''
Define the optimizer
'''
optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1, momentum=0.1)
#optimizer="adam"


'''
Compile the model.
Since labels for both tasks are categorical, we use the same loss function.
Otherwise we have to use specify the loss functions using a list.

Choice of loss function:
- contrastive_loss (we defined it earlier)
- categorical_crossentropy
'''
model_1.compile(loss=[contrastive_loss, "categorical_crossentropy", "categorical_crossentropy"],
              optimizer=optimizer,
              metrics=["accuracy"])


'''
Create a callback object of early stopping
'''
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_dense_4_loss',
                                  min_delta=0, 
                                  patience=15, 
                                  verbose=1, 
                                  mode='auto',
                                  restore_best_weights=True)

'''
Train the model
We need to specify three types of labels: 
(i) labels indicating whether a pair of images include the augmented version 
    of the same image (label = 1) or not (label = 0)
(ii) labels of the original task
(iii) labels of the augmented task
'''
history_1 = model_1.fit([X_train, X_train_rot_shuffled], [y_train_distance, y_train, y_train_rot_shuffled], 
                    batch_size=64, 
                    epochs=150,
                    verbose=1,
                    validation_data=([X_valid, X_valid_rot_shuffled], [y_valid_distance, y_valid, y_valid_rot_shuffled]),
                    callbacks=[lschedule_cb]
                    )

numOfEpochs = len(history_1.history['loss'])
print("Epochs: ", numOfEpochs)

print("\n******************** Multiclass Classification ********************************************")


'''
Prediction for test data
'''
# model.predict() method returns C probabilities per class for each instance (Dimension NxC), where C = #classes
y_test_predicted = model_1.predict([X_test, X_test_rot_shuffled])
y_test_predicted_multiclass = np.argmax(y_test_predicted[1], axis=1) # get the label/index of the highest probability class
y_test_predicted_binary = np.argmax(y_test_predicted[2], axis=1) # get the label/index of the highest probability class

'''
Prediction for training data
'''
y_train_predicted = model_1.predict([X_train, X_train_rot_shuffled])
y_train_predicted_multiclass = np.argmax(y_train_predicted[1], axis=1) # get the label/index of the highest probability class
y_train_predicted_binary = np.argmax(y_train_predicted[2], axis=1) # get the label/index of the highest probability class


'''
Get the integer labels for the multiclass data
'''
y_test_multiclass = np.argmax(y_test, axis=1) # get the label/index of the highest probability class
y_train_multiclass = np.argmax(y_train, axis=1) # get the label/index of the highest probability class


'''
Compute the train & test accuracies for the multiclass data
'''
train_accuracy_multiclass = accuracy_score(y_train_predicted_multiclass, y_train_multiclass)
test_accuracy_multiclass_1 = accuracy_score(y_test_predicted_multiclass, y_test_multiclass)


print("\nMulticlass Classification - Train Accuracy: ", train_accuracy_multiclass)
print("Multiclass Classification - Test Accuracy: ", test_accuracy_multiclass_1)


print("\nTest Confusion Matrix (Multiclass):")
print(confusion_matrix(y_test_multiclass, y_test_predicted_multiclass))

print("\nClassification Report (Multiclass):")
print(classification_report(y_test_multiclass, y_test_predicted_multiclass))



print("\n******************** Binary Classification ********************************************")


'''
Get the integer labels for the binary data
'''
y_train_rot_binary = np.argmax(y_train_rot_shuffled, axis=1) # get the label/index of the highest probability class
y_test_rot_binary = np.argmax(y_test_rot_shuffled, axis=1) # get the label/index of the highest probability class


train_accuracy_binary = accuracy_score(y_train_rot_binary, y_train_predicted_binary)
test_accuracy_binary = accuracy_score(y_test_rot_binary, y_test_predicted_binary)

print("\nBinary Classification - Train Accuracy: ", train_accuracy_binary)
print("Binary Classification - Test Accuracy: ", test_accuracy_binary)


print("\nTest Confusion Matrix (Binary):")
print(confusion_matrix(y_test_rot_binary, y_test_predicted_binary))

print("\nClassification Report (Binary):")
print(classification_report(y_test_rot_binary, y_test_predicted_binary))

'''
Delete the TensorFlow graph before creating a new model, otherwise memory overflow will occur.
'''
tf.keras.backend.clear_session()

'''
To reproduce the same result by the model in each iteration, we use fixed seeds for random number generation. 
'''
np.random.seed(42)
tf.random.set_seed(42)

input_shape = (32, 32, 3)

'''
Create the base network for learning shared representations
'''
base_network = create_base_network_cnn_shallow(input_shape)


'''
Define the shape of the input for both tasks
'''
input_a = tf.keras.Input(shape=input_shape)
input_b = tf.keras.Input(shape=input_shape)

'''
Pass both inputs through the base network
It will create shared reprsentations
'''
processed_a = base_network(input_a)
processed_b = base_network(input_b)


'''
Add two task-specific layers (for two inputs)
'''
projection_head_1 = tf.keras.layers.Dense(50, activation="relu")(processed_a)
projection_head_2 = tf.keras.layers.Dense(50, activation="relu")(processed_b)

'''
Add classification layers (for two inputs)
'''
output1 = tf.keras.layers.Dense(10, activation="softmax")(projection_head_1)
output2 = tf.keras.layers.Dense(4, activation="softmax")(projection_head_2)


'''
Create a Model by specifying its input and outputs
'''

model_2 = tf.keras.models.Model(inputs=[input_a, input_b], outputs=[output1, output2], name="Experiment-2")

'''
Display the model summary
'''
model_2.summary()

'''
Display the model graph
'''
#tf.keras.utils.plot_model(model_2, show_shapes=True)


'''
Define the optimizer
'''
optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1, momentum=0.1)
#optimizer="adam"


'''
Compile the model.
Since labels for both tasks are categorical, we use the same loss function.
Otherwise we have to use specify the loss functions using a list.
'''

model_2.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])



'''
Create a callback object of early stopping
'''
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_dense_4_loss',
                                  min_delta=0, 
                                  patience=10, 
                                  verbose=1, 
                                  mode='auto',
                                  restore_best_weights=True)

'''
Train the model
We need to specify two types of labels: 
(i) labels of the original task
(ii) labels of the augmented task
'''
history_2 = model_2.fit([X_train, X_train_rot_shuffled], [y_train, y_train_rot_shuffled], 
                    batch_size=64, 
                    epochs=150,
                    verbose=1,
                    validation_data=([X_valid, X_valid_rot_shuffled], [y_valid, y_valid_rot_shuffled]),
                    callbacks=[lschedule_cb]
                    )
                    
numOfEpochs = len(history_2.history['loss'])
print("Epochs: ", numOfEpochs)

print("\n******************** Multiclass Classification ********************************************")


'''
Prediction for test data
'''
# model.predict() method returns C probabilities per class for each instance (Dimension NxC), where C = #classes
y_test_predicted = model_2.predict([X_test, X_test_rot_shuffled])
y_test_predicted_multiclass = np.argmax(y_test_predicted[1], axis=1) # get the label/index of the highest probability class
y_test_predicted_binary = np.argmax(y_test_predicted[2], axis=1) # get the label/index of the highest probability class

'''
Prediction for training data
'''
y_train_predicted = model_2.predict([X_train, X_train_rot_shuffled])
y_train_predicted_multiclass = np.argmax(y_train_predicted[1], axis=1) # get the label/index of the highest probability class
y_train_predicted_binary = np.argmax(y_train_predicted[2], axis=1) # get the label/index of the highest probability class


'''
Get the integer labels for the multiclass data
'''
y_test_multiclass = np.argmax(y_test, axis=1) # get the label/index of the highest probability class
y_train_multiclass = np.argmax(y_train, axis=1) # get the label/index of the highest probability class


'''
Compute the train & test accuracies for the multiclass data
'''
train_accuracy_multiclass = accuracy_score(y_train_predicted_multiclass, y_train_multiclass)
test_accuracy_multiclass_1 = accuracy_score(y_test_predicted_multiclass, y_test_multiclass)


print("\nMulticlass Classification - Train Accuracy: ", train_accuracy_multiclass)
print("Multiclass Classification - Test Accuracy: ", test_accuracy_multiclass_1)


print("\nTest Confusion Matrix (Multiclass):")
print(confusion_matrix(y_test_multiclass, y_test_predicted_multiclass))

print("\nClassification Report (Multiclass):")
print(classification_report(y_test_multiclass, y_test_predicted_multiclass))



print("\n******************** Binary Classification ********************************************")


'''
Get the integer labels for the binary data
'''
y_train_rot_binary = np.argmax(y_train_rot_shuffled, axis=1) # get the label/index of the highest probability class
y_test_rot_binary = np.argmax(y_test_rot_shuffled, axis=1) # get the label/index of the highest probability class


train_accuracy_binary = accuracy_score(y_train_rot_binary, y_train_predicted_binary)
test_accuracy_binary = accuracy_score(y_test_rot_binary, y_test_predicted_binary)

print("\nBinary Classification - Train Accuracy: ", train_accuracy_binary)
print("Binary Classification - Test Accuracy: ", test_accuracy_binary)


print("\nTest Confusion Matrix (Binary):")
print(confusion_matrix(y_test_rot_binary, y_test_predicted_binary))

print("\nClassification Report (Binary):")
print(classification_report(y_test_rot_binary, y_test_predicted_binary))