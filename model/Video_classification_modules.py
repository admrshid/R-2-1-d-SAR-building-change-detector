import keras
from keras import layers
import einops
import tensorflow as tf

class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        #convolution operations over the spatial then temporal dimension

        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.seq = keras.Sequential([
            # Spatial decomposition
            layers.Conv3D(filters=filters, kernel_size=(1, kernel_size[1], kernel_size[2]), padding=padding),
            # Temporal decomposition
            layers.Conv3D(filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding)])
    def call(self,x):
        return self.seq(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters':self.filters,
            'kernel_size':self.kernel_size,
            'padding':self.padding
        })

        return config

    
class ResidualMain(keras.layers.Layer):

    def __init__(self, filters, kernel_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.seq = keras.Sequential([
            Conv2Plus1D(filters=filters,kernel_size=kernel_size,padding='same'), # padding = 'same' means want to maintain dim
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=filters,kernel_size=kernel_size,padding='same'),
            layers.LayerNormalization()
        ])

    def call(self,x):
        return self.seq(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        })
        return config
    
class Project(keras.layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.units = units
        self.seq = keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
        })
        return config
    
def add_residual_block(input, filters, kernel_size):

    # function to add residual block

    out = ResidualMain(filters, kernel_size)(input)

    res = input

    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)
    
    return layers.add([res, out])

class ResizeVideo(keras.layers.Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        # einops library to resize tensor

        # video input as set of frames

        # output is downsampled size of video

        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)

        videos = einops.rearrange(images, '(b t) h w c -> b t h w c', t = old_shape['t'])
    
        return videos
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'height': self.height,
            'width' : self.width
        })

        return config
    

class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gamma = self.add_weight(shape=(channels,),
                                     initializer='ones',
                                     trainable=True,
                                     name='gamma')
        self.beta = self.add_weight(shape=(channels,),
                                    initializer='zeros',
                                    trainable=True,
                                    name='beta')
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2, 3], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        outputs = self.gamma * normalized + self.beta
        return outputs