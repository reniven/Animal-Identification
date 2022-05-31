import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np 

def prepare_dataset(ds, mini_batch, shuffle=False, augment=False, buffer_size=0,  hotEncode = False, num_classes = 0):

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
    ds = ds.repeat(2) 
    
    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
def piecewise_constant(epoch, lr): 
    initial_lrate = lr  
    
    if(epoch < 150):
    	lrate = initial_lrate
        
    elif(epoch < 250):
        lrate = initial_lrate*0.1  

    elif(epoch < 300):
    	lrate = initial_lrate*0.01
    	
    else:
        lrate = initial_lrate*0.0001 
        
    return lrate

def piecewise_constant2(epoch, lr): 
    initial_lrate = lr  
    
    if(epoch < 50):
        lrate = initial_lrate

    elif(epoch < 150):
    	lrate = initial_lrate * 0.1
        
    elif(epoch < 250):
        lrate = initial_lrate*0.01  

    elif(epoch < 300):
    	lrate = initial_lrate*0.001
    	
    else:
        lrate = initial_lrate*0.00001 
        
    return lrate

'''
A class to increase the learning rate by a "factor" at each iteration.
It stores the changing learning rate and loss at each iteration.
'''

K = tf.keras.backend

class IncreaseLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)
        
        
'''
Function to determine the "factor" to be used in the IncreaseLearningRate (above) object
'''
def computeFactorForIncreasingLearningRate(X, size_mini_batch, no_of_epochs, upper_bound_lr, lower_bound_lr):
    
    # Total number of iterations
    iterations = len(X) // size_mini_batch * no_of_epochs
    
    factor = np.exp(np.log(upper_bound_lr / lower_bound_lr) / iterations)
    
    return factor

class OneCycle(tf.keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, momentum_vals=(0.95, 0.85), start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
        
        self.step_len =  int(self.iterations * (1 - 10.0/100.0)/2)
        
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
    
    def on_batch_begin(self, batch, logs):
        
        # Set Learning Rate
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)
        
        
        # Set Momentum
        if self.iteration == 0:
            return self.high_mom
        elif self.iteration == self.iterations:
            self.iteration = 0
            return self.high_mom
        elif self.iteration > 2 * self.step_len:
            mom = self.high_mom
        elif self.iteration > self.step_len:
            ratio = (self.iteration - self.step_len)/self.step_len
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        else :
            ratio = self.iteration/self.step_len
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        K.set_value(self.model.optimizer.momentum, mom)

'''
Randomly adjust the brightness of the batch images
The "delta" argument of the tf.image.adjust_brightness function should be in the range (-1,1).
Using the "brightness_factor" (0, 1), we create random values for the "delta" between (-1, 1).
Recommended value for the "brightness_factor": 0.1 ~ 0.3
'''
def random_adjust_brighness_tf_image(x, proba, brightness_factor):
    # Select a random value from a uniform distribution between 0 ~ 1
    rand_value = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)
    
    
    # Select a random value from a uniform distribution between -brightness_factor ~ +brightness_factor
    rand_brightness_factor = tf.random.uniform(shape=[], minval=-brightness_factor, 
                                               maxval=+brightness_factor, dtype=tf.float32)

    # Perform a random change in brightness
    return tf.cond(rand_value < proba, lambda: x, lambda: tf.image.adjust_brightness(x, delta=rand_brightness_factor))


'''
Randomly adjust the contrast of the batch images
The "contrast_factor" argument should be >= 1.0 
Recommended value for the "contrast_factor": 1.0 ~ 3.0
'''
def random_adjust_contrast_tf_image(x, proba, contrast_factor):
    # Select a random value from a uniform distribution between 0 ~ 1
    rand_value = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)

    # Perform a random change in contrast
    return tf.cond(rand_value < proba, lambda: x, lambda: tf.image.adjust_contrast(x, contrast_factor=contrast_factor))



'''
Randomly adjust the gamma of the batch images
'''
def random_adjust_gamma_tf_image(x, proba):
    # Select a random value from a uniform distribution between 0 ~ 1
    rand_value = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)

    # Perform a random change in gamma 
    return tf.cond(rand_value < proba, lambda: x, lambda: tf.image.adjust_gamma(x, gamma=1.5, gain=1))



'''
Randomly adjust the hue of the batch images
'''
def random_adjust_hue_tf_image(x, proba):
    # Select a random value from a uniform distribution between 0 ~ 1
    rand_value = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)

    # Perform a random change in hue
    return tf.cond(rand_value < proba, lambda: x, lambda: tf.image.adjust_hue(x, delta=0.1))



'''
Randomly rotate the batch images within the range of a given degree
'''
def random_rotation_tfa(x, rotation_angle_degree):
    rotation_angle_radian = rotation_angle_degree * (3.1416/ 180)
    
    x = tfa.image.rotate(x, np.random.uniform(-rotation_angle_radian, rotation_angle_radian),
                             fill_mode='nearest', interpolation='bilinear')
    return x




'''
Randomly apply gaussian blurring on the batch images
The "filter_shape" and "sigma" of the tfa.image.gaussian_filter2d function should be set carefully.
For higher resoultion images, "filter_shape" should be 10~30 & "sigma" should be 10~20

- filter_shape:
An integer specifying the height and width of the 2-D gaussian filter.
- sigma:
A float specifying the standard deviation in x and y direction the 2-D gaussian filter. 
'''
def random_gaussian_blur_tfa(x, proba, gaussian_filter_shape, gaussian_sigma):
    # Select a random value from a uniform distribution between 0 ~ 1
    rand_value = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)

    # Perform a random gaussian blurring 
    return tf.cond(rand_value < proba, lambda: x, 
                   lambda: tfa.image.gaussian_filter2d(x, filter_shape=gaussian_filter_shape, 
                                                       sigma=gaussian_sigma))


'''
Randomly cutout a patch (mask_size x mask_size) from the batch images
Recommended setting for "mask_size" is one-third of the size of the image.
For example: if image size=150x150, then mask_size=50
'''
def random_random_cutout_tfa(x, proba, mask_size):
    # Select a random value from a uniform distribution between 0 ~ 1
    rand_value = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)

    # Perform a random cutout change
    return tf.cond(rand_value < proba, lambda: x, 
                   lambda: tfa.image.random_cutout(x, mask_size=mask_size, constant_values=0))




'''
IMPORTANT: the parameters of the DataAugmentation class should be set carefully.
An optimal setting must be based on the problem (dataset).
It is STRONGLY recommended to first try each augmentation on some sample images from the problem dataset.
Carefully note what augmentations settings are useful, then use those settings during training.
Below we provide some recommeded setting. However, these should be adapted based on the problem.
Never forget that data augmentation is a contextual trick. It will not be beneficial or even detrimetal
if it is applied without considering the context.


The DataAugmentation class is defined using the tf.Keras' Sequential API.
It creates a data augmentation layer for performing following augmentation to input data.
It uses some TensorFlow Image ops and Addons ops (defined by the relevant functions). 

Following augmentations are applied:
- Resize (increase the size) & Random crop (restore the original size)
  -- Resize must be followed by random crop
  -- Both of these can be controlled by boolean arguments "resize" & "random_crop"
- Adjust brightness with a user-defined probability
  -- Recommended value for the "brightness_factor": 0.1 ~ 0.3
- Adjust contrast with a user-defined probability
  -- Recommended value for the "contrast_factor": 1.0 ~ 3.0
- Adjust gamma with a user-defined probability
  -- The "random_adjust_gamma_tf_image" uses the following setting: gamma=1.5, gain=1
  -- If needed, these values should be changed in the function
- Adjust hue with a user-defined probability
  -- The "random_adjust_hue_tf_image" uses the following setting: delta=0.1
  -- If needed, these values should be changed in the function
- Apply gaussian blur with a user-defined probability
  -- "gaussian_filter_shape" and "gaussian_sigma" should be set based on the resolution of the images
  -- For high resolution images (100 ~ 400): both could be set to 10~20
- Random zoom
  -- Recommended value for "zoom_height_factor" is 0.6
- Random rotation
  -- Rotation angle (in degree) is set by the user
- Random horizontal flip
- Random translation (height & width)
  -- Controlled by "translation_height_factor" and "translation_width_factor"  
- Apply cutout with a user-defined probability
  -- Recommended setting for "mask_size" is one-third of the size of the image.
  -- For example: if image size=150x150, then mask_size=50


NOTE: for the use-defined probability, larger value decreases the likelihood of augmentation.
For example: if probability is set to 1.0, then no augmentation, 
and 0 probability applies 100% augmentation.
'''

class DataAugmentation(tf.keras.layers.Layer):
    def __init__(self, resize=False, increased_size=0, 
                 random_crop=False, original_size=0, 
                 rotation_angle_degree=0, 
                 zoom_height_factor=0,
                 brightness_proba=1.0,
                 brightness_factor=0,
                 contrast_proba=1.0,
                 contrast_factor=0,
                 gamma_proba=1.0, hue_proba=1.0, blur_proba=1.0, 
                 gaussian_filter_shape=0, gaussian_sigma=0,
                 cutout_proba=1.0,
                 mask_size=0,
                 random_flip=False, 
                 translation_height_factor=0,
                 translation_width_factor=0, **kwargs):
        super().__init__(**kwargs)
        
        self.augmentation_layers = [] 
        
        
        if(resize):
            self.augmentation_layers.append(tf.keras.layers.experimental.preprocessing.Resizing(increased_size, increased_size))
            
        if(brightness_proba < 1.0):
            self.augmentation_layers.append(tf.keras.layers.Lambda(lambda x: (random_adjust_brighness_tf_image(x, brightness_proba, brightness_factor))))
            
        if(contrast_proba < 1.0):
            self.augmentation_layers.append(tf.keras.layers.Lambda(lambda x: (random_adjust_contrast_tf_image(x, contrast_proba, contrast_factor))))
        
        if(gamma_proba < 1.0):
            self.augmentation_layers.append(tf.keras.layers.Lambda(lambda x: (random_adjust_gamma_tf_image(x, gamma_proba))))
        
        if(hue_proba < 1.0):
            self.augmentation_layers.append(tf.keras.layers.Lambda(lambda x: (random_adjust_hue_tf_image(x, hue_proba))))
        
        if(blur_proba < 1.0):
            self.augmentation_layers.append(tf.keras.layers.Lambda(lambda x: (random_gaussian_blur_tfa(x, blur_proba,
                                                                                                       gaussian_filter_shape, 
                                                                                                       gaussian_sigma))))       

        if(zoom_height_factor > 0):
            self.augmentation_layers.append(tf.keras.layers.experimental.preprocessing.RandomZoom(zoom_height_factor, width_factor=None, 
                                                                                                  fill_mode='nearest', 
                                                                                                  interpolation='bilinear'))
        
        if(rotation_angle_degree > 0): 
            self.augmentation_layers.append(tf.keras.layers.experimental.preprocessing.RandomRotation(rotation_angle_degree/360, 
                                                                                                      fill_mode='nearest', 
                                                                                                      interpolation='bilinear'))
        
        if(random_flip):
            self.augmentation_layers.append(tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"))
        
        if(random_crop):
            self.augmentation_layers.append(tf.keras.layers.experimental.preprocessing.RandomCrop(original_size, original_size))
        
        
        
        if(translation_height_factor > 0 or translation_width_factor > 0):
            self.augmentation_layers.append(tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=translation_height_factor, 
                                                                     width_factor=translation_width_factor, 
                                                                     fill_mode='nearest',
                                                                     interpolation='bilinear'))

                                                
        if(cutout_proba < 1.0):
            self.augmentation_layers.append(tf.keras.layers.Lambda(lambda x: (random_random_cutout_tfa(x, cutout_proba, 
                                                                                                       mask_size))))
        
 
    '''
    Apply augmentation on the batch input
    '''    
    def call(self, inputs):
        Z = inputs
        for layer in self.augmentation_layers:
            Z = layer(Z)

        return Z
    
    
    '''
    This method is required for the serialization of this custom class (object).
    A custom class is defined by extending the built-in TensorFlow classes (e.g., tf.keras.layers.Layer).
    If data augmentation is done inside the model (during training),
    then for saving the model, the get_config method is required for all custom classses.
    '''
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "augmentation_layers": self.augmentation_layers
        })
        return config
  

'''
This function creates a sequential tf.Keras layer for performing data augmentation
'''
def data_augmentation_layer(**kwargs):
    
    d_augmentation = tf.keras.models.Sequential(name='Data-Augmentation')
    d_augmentation.add(DataAugmentation(**kwargs))

    return d_augmentation  