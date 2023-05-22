import tensorflow as tf
from typing import Iterable, Tuple

class DecoderBlock(tf.keras.layers.Layer):
    """
    """
    def __init__(self,
                 filters:int,
                 kernel_size:Tuple[int, int],
                 strides:Tuple[int, int],
                 data_format=None,
                 activation=None, 
                 use_batchnorm=True,
                 kernel_initializer="glorot_uniform", 
                 bias_initializer="zeros", 
                 kernel_regularizer=None,
                 bias_regularizer=None, 
                 activity_regularizer=None, 
                 kernel_constraint=None,
                 bias_constraint=None,
                 post_activation = "relu",
                 block_number = None, 
                 **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)


        self.filters                =   filters
        self.kernel_size            =   kernel_size
        self.strides                =   strides
        self.data_format            =   data_format
        self.activation             =   activation
        self.use_bias               =   not (use_batchnorm)
        self.kernel_initializer     =   kernel_initializer
        self.bias_initializer       =   bias_initializer
        self.kernel_regularizer     =   kernel_regularizer
        self.bias_regularizer       =   bias_regularizer
        self.activity_regularizer   =   activity_regularizer
        self.kernel_constraint      =   kernel_constraint
        self.bias_constraint        =   bias_constraint

        self.post_activation_name   =   post_activation

        self.transposed             =   None
        self.bn1                    =   None
        self.conv                   =   None
        self.bn2                    =   None
        self.post_activation        =   None
    
    def build(self, input_shape):
        self.transposed = tf.keras.layers.Conv2DTranspose(
            input_shape             =   input_shape,
            filters                 =   self.filters,
            kernel_size             =   self.kernel_size,
            strides                 =   self.strides,
            padding                 =   "same",
            data_format             =   self.data_format,
            activation              =   self.activation,
            use_bias                =   self.use_bias,
            kernel_initializer      =   self.kernel_initializer,
            bias_initializer        =   self.bias_initializer,
            kernel_regularizer      =   self.kernel_regularizer,
            bias_regularizer        =   self.bias_regularizer,
            activity_regularizer    =   self.activity_regularizer,
            kernel_constraint       =   self.kernel_constraint,
            bias_constraint         =   self.bias_constraint,
            name                    =   "decoder_transposed"
        )

        self.bn1 = tf.keras.layers.BatchNormalization()


        self.conv = tf.keras.layers.Conv2D(
            input_shape             =   input_shape,
            filters                 =   self.filters,
            kernel_size             =   (3, 3),
            strides                 =   (1, 1),
            padding                 =   "same",
            data_format             =   self.data_format,
            activation              =   self.activation,
            use_bias                =   self.use_bias,
            kernel_initializer      =   self.kernel_initializer,
            bias_initializer        =   self.bias_initializer,
            kernel_regularizer      =   self.kernel_regularizer,
            bias_regularizer        =   self.bias_regularizer,
            activity_regularizer    =   self.activity_regularizer,
            kernel_constraint       =   self.kernel_constraint,
            bias_constraint         =   self.bias_constraint,
            name                    =   "decoder_conv",
        )

        self.bn2 = tf.keras.layers.BatchNormalization()

        self.post_activation = tf.keras.layers.Activation(self.post_activation_name)
        

    def call(self, x, training=None):
        x = self.transposed(x)
        x = self.bn1(x, training=training)
        x = self.conv(x)
        x = self.bn2(x, training=training)
        x = self.post_activation(x)
        return x

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1]*self.strides[0], input_shape[2]*self.strides[1], self.filters]

    def get_config(self):
        config = {
            "filters"               :   self.filters,
            "kernel_size"           :   self.kernel_size,
            "strides"               :   self.strides,
            "data_format"           :   self.data_format,
            "activation"            :   self.activation,
            "use_bias"              :   self.use_bias,
            "kernel_initializer"    :   self.kernel_initializer,
            "bias_initializer"      :   self.bias_initializer,
            "kernel_regularizer"    :   self.kernel_regularizer,
            "bias_regularizer"      :   self.bias_regularizer,
            "activity_regularizer"  :   self.activity_regularizer,
            "kernel_constraint"     :   self.kernel_constraint,
            "bias_constraint"       :   self.bias_constraint,
            "post_activation_name"  :   self.post_activation
        }
        base_config = super(DecoderBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Unet(tf.keras.Model):
    def __init__(
            self,
            n_classes,
            backbone:tf.keras.Model,
            decoder_filters:Iterable[int] = (256, 128, 64, 32, 16),
            decoder_kernel_sizes:Iterable[int] = (3, 3, 3, 3, 3),
            **kwargs):
        super(Unet, self).__init__(**kwargs)

        # == encoder == #
        self.backbone = backbone
        
        # == decoder == #
        skip_connections = backbone.output[:-1]
        skip_connections.reverse()
        
        bottleneck_input = backbone.output[-1]

        self.decoder_blocks = []
        for i, skip in enumerate(skip_connections):
            if i == 0:
                strides = (
                    skip.shape[1]//bottleneck_input.shape[1],
                    skip.shape[2]//bottleneck_input.shape[2]
                )
            else:
                strides = (
                    skip.shape[1]//skip_connections[i-1].shape[1],
                    skip.shape[2]//skip_connections[i-1].shape[2]
                )

            self.decoder_blocks.append(
                DecoderBlock(
                    filters = decoder_filters[i], 
                    kernel_size=(decoder_kernel_sizes[i]),
                    strides=strides
                )
            )
        
        self.bottleneck = tf.keras.layers.Conv2D(
            filters=512, 
            kernel_size=(3, 3), 
            padding="same"
        )

        self.final_conv_1x1 = tf.keras.layers.Conv2D(
            filters=n_classes,
            kernel_size=(1, 1),
            padding="same"
        )

    def call(self, inputs):

        backbone_outputs = backbone(inputs)

        skip_connections = backbone_outputs[:-1]
        print(skip_connections)
        skip_connections.reverse()
        

        bottleneck_input = backbone_outputs[-1]

        bottleneck_output = self.bottleneck(bottleneck_input)
        t = bottleneck_output

        for i, block in enumerate(self.decoder_blocks):
            t = block(t)
            t = tf.keras.layers.Concatenate()([t, skip_connections[i]])
            
        
        outputs = self.final_conv_1x1(t)

        return outputs
    

if __name__ == "__main__":
    # == build backbone == #
    vgg = tf.keras.applications.vgg16.VGG16(input_shape=(256, 256, 3), include_top=False)
    output_layers = [
        vgg.get_layer(f"block{i+1}_pool").output for i in range(5)
        ]
    backbone = tf.keras.Model(vgg.input, output_layers, name=f"{vgg.name}_backbone")

    # == build model == #
    model = Unet(2, backbone = backbone)
    model.build(backbone.input.shape)
    model.summary()