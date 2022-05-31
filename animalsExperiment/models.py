import numpy as np
import tensorflow as tf
from utils import data_augmentation_layer
'''
Define the VGG block class using Keras' Sequential API 
The VGG_Block class takes two arguments:
- conv_block_number: number of convolutional layers 
- num_of_channels: number of output channels 
'''
class VGG_Block(tf.keras.layers.Layer):
    def __init__(self, conv_block_number, num_of_channels, weight_decay, efficient=False, **kwargs):
        super().__init__(**kwargs)
        self.conv_layers = []
        self.efficient = efficient
        for _ in range(conv_block_number):
            self.conv_layers.append(tf.keras.layers.Conv2D(filters=num_of_channels, kernel_size=(3, 3), strides=1,
             padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
            
            if(self.efficient == True):
                self.conv_layers.append(tf.keras.layers.BatchNormalization())
                self.conv_layers.append(tf.keras.activations.get("relu"))
            else:
                self.conv_layers.append(tf.keras.activations.get("relu"))

        self.pool_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')

        if(self.efficient == True):
            self.last_Conv2D = tf.keras.layers.Conv2D(filters=(num_of_channels/2), kernel_size=(1, 1), strides=1,
             padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
            self.last_BN = tf.keras.layers.BatchNormalization()
            self.last_activation = tf.keras.activations.get("relu")

    def call(self, inputs):
        Z = inputs
        for layer in self.conv_layers:
            Z = layer(Z)
            
        Z = self.pool_layer(Z)

        if(self.efficient == True):
            Z = self.last_Conv2D(Z)
            Z = self.last_BN(Z)
            Z = self.last_activation(Z)
        
        return Z
      
    # Required for the custom object's serialization
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "conv_layers": self.conv_layers,
            "pool_layer": self.pool_layer,
        })
        return config

def vgg_19(conv_blocks, width, height, channels, num_classes, weight_decay, efficient = False, dataAug = False, **kwargs):
    
    vgg_net = tf.keras.models.Sequential(name='VGG-19')
    
    vgg_net.add(tf.keras.layers.InputLayer(input_shape=(width, height, channels)))

    # Data augmentation layer
    if(dataAug):
        vgg_net.add(data_augmentation_layer(**kwargs))
    
    # Conv part
    for (conv_block_number, num_of_channels) in conv_blocks:
            print(conv_block_number)
            print(num_of_channels)
            vgg_net.add(VGG_Block(conv_block_number, num_of_channels, weight_decay, efficient))
    
    if(efficient == True):
        vgg_net.add(tf.keras.layers.GlobalAvgPool2D())
        vgg_net.add(tf.keras.layers.Flatten())
        vgg_net.add(tf.keras.layers.Dense(units=num_classes, activation="softmax"))
    else:
        # Flatten the convnet output to feed it with fully connected layers
        vgg_net.add(tf.keras.layers.Flatten())
    
        # FC part
        vgg_net.add(tf.keras.layers.Dense(units=4096, activation='relu'))
        vgg_net.add(tf.keras.layers.Dropout(0.5))
        vgg_net.add(tf.keras.layers.Dense(units=4096, activation='relu'))
        vgg_net.add(tf.keras.layers.Dropout(0.5))
        vgg_net.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))
    
    return vgg_net

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

def resNet32(input_shape, num_of_output_classes, augmentation = False, **kwargs):
    
    resnet = tf.keras.models.Sequential(name='ResNet-32')
    
    resnet.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    # Data augmentation layer
    if(augmentation):
        resnet.add(data_augmentation_layer(**kwargs))

    '''
    Set the use_bias to False because the following BN layer adds a bias. 
    The BN "shift" parameter shifts the output of the layer (thus acts like a bias).
    '''
    resnet.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', 
                     input_shape=input_shape, use_bias=False))
    resnet.add(tf.keras.layers.BatchNormalization())
    resnet.add(tf.keras.layers.Activation("relu"))
    resnet.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))

    # Add the residual blocks
    prev_filters = 64
    for filters in [64] * 5 + [128] * 5 + [256] * 5:
        strides = 1 if filters == prev_filters else 2
        resnet.add(Residual_Block(filters, strides=strides))
        prev_filters = filters
        
    # Perform global average pooling
    resnet.add(tf.keras.layers.GlobalAvgPool2D())
    resnet.add(tf.keras.layers.Flatten())
    resnet.add(tf.keras.layers.Dense(units=num_of_output_classes, activation="softmax"))
    
    return resnet