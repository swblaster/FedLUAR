import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Resizing
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.regularizers import l2
from transformers import TFDistilBertModel, DistilBertTokenizer

class distilBert ():
    def __init__(self, weight_decay, sample_length, num_classes):
        self.weight_decay = weight_decay
        self.regularizer = l2(self.weight_decay)
        self.num_classes = num_classes
        self.sample_length = sample_length

    def build_model(self):
        distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

        input_ids = Input(shape=(self.sample_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.sample_length,), dtype=tf.int32, name='attention_mask')

        distilbert_output = distilbert_model(input_ids, attention_mask=attention_mask)
        
        cls_token = distilbert_output.last_hidden_state[:, 0, :]  

        dense = Dense(units=2048, activation='relu', kernel_regularizer=self.regularizer)(cls_token)

        y = Dense(self.num_classes, activation="softmax")(dense)
        model = Model(inputs=[input_ids, attention_mask], outputs=y, name="DistilBERT_Model")
        return model
    
class resnet20 ():
    def __init__ (self, weight_decay, num_classes, ratio = 1.0):
        self.weight_decay = weight_decay
        self.regularizer = l2(self.weight_decay)
        self.initializer = tf.keras.initializers.GlorotUniform(seed = int(time.time()))
        self.num_classes = num_classes
        self.ratio = ratio

    def res_block (self, input_tensor, num_filters, strides = (1, 1), projection = False):
        x = Conv2D(num_filters,
                   (3, 3),
                   strides = strides,
                   padding = "same",
                   use_bias = False,
                   kernel_initializer = self.initializer,
                   kernel_regularizer = self.regularizer)(input_tensor)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = Conv2D(num_filters,
                   (3, 3),
                   padding = "same",
                   use_bias = False,
                   kernel_initializer = self.initializer,
                   kernel_regularizer = self.regularizer)(x)
        x = BatchNormalization(gamma_initializer = 'zeros')(x)
        if projection:
            shortcut = Conv2D(num_filters,
                              (1, 1),
                              padding = "same",
                              use_bias = False,
                              kernel_initializer = self.initializer,
                              kernel_regularizer = self.regularizer)(input_tensor)
            shortcut = BatchNormalization()(shortcut)
        elif strides != (1, 1):
            shortcut = Conv2D(num_filters,
                              (1, 1),
                              strides = strides,
                              padding = "same",
                              use_bias = False,
                              kernel_initializer = self.initializer,
                              kernel_regularizer = self.regularizer)(input_tensor)
            shortcut = BatchNormalization()(shortcut)
        else:
            shortcut = input_tensor

        x = x + shortcut
        y = tf.nn.relu(x)
        return y

    def build_model (self):
        x_in = Input(shape = (None, None, 3), name = "input")

        # The first conv layer.
        x = Conv2D(16,
                   (3, 3),
                   strides=(1, 1),
                   name='conv0',
                   padding='same',
                   use_bias=False,
                   kernel_initializer = self.initializer,
                   kernel_regularizer = self.regularizer) (x_in)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        # Residual blocks
        for i in range (3):
            if i == 0:
                x = self.res_block(x, int(16 * self.ratio), projection = True)
            else:
                x = self.res_block(x, int(16 * self.ratio))

        for i in range (3):
            if i == 0:
                x = self.res_block(x, int(32 * self.ratio), strides = (2, 2))
            else:
                x = self.res_block(x, int(32 * self.ratio))

        for i in range (3):
            if i == 0:
                x = self.res_block(x, int(64 * self.ratio), strides = (2, 2))
            else:
                x = self.res_block(x, int(64 * self.ratio))

        # The final average pooling layer and fully-connected layer.
        x = GlobalAveragePooling2D()(x)
        #y = Dense(self.num_classes, activation = 'softmax', name='fully_connected',
        y = Dense(self.num_classes, name='fully_connected',
                  kernel_initializer = self.initializer,
                  kernel_regularizer = self.regularizer,
                  bias_regularizer = self.regularizer)(x)
        return Model(x_in, y, name = "resnet20")

class wideresnet28 ():
    def __init__ (self, weight_decay, num_classes):
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.batch_norm_momentum = 0.99
        self.batch_norm_epsilon = 1e-5

    def res_block (self, input_tensor, num_filters, strides = (1, 1), projection = False):
        x = Conv2D(num_filters,
                   (3, 3),
                   strides = strides,
                   padding = "same",
                   use_bias = False,
                   kernel_regularizer = self.regularizer)(input_tensor)
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               epsilon = self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)
        x = Dropout(0.3)(x)

        x = Conv2D(num_filters,
                   (3, 3),
                   padding = "same",
                   use_bias = False,
                   kernel_regularizer = self.regularizer)(x)
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               gamma_initializer = 'zeros',
                               epsilon = self.batch_norm_epsilon)(x)
        if projection:
            shortcut = Conv2D(num_filters,
                              (1, 1),
                              padding = "same",
                              use_bias = False,
                              kernel_regularizer = self.regularizer)(input_tensor)
            shortcut = BatchNormalization(momentum = self.batch_norm_momentum,
                                          epsilon = self.batch_norm_epsilon)(shortcut)
        elif strides != (1, 1):
            shortcut = Conv2D(num_filters,
                              (1, 1),
                              strides = strides,
                              padding = "same",
                              use_bias = False,
                              kernel_regularizer = self.regularizer)(input_tensor)
            shortcut = BatchNormalization(momentum = self.batch_norm_momentum,
                                          epsilon = self.batch_norm_epsilon)(shortcut)
        else:
            shortcut = input_tensor

        x = x + shortcut
        y = tf.nn.relu(x)
        return y

    def build_model (self):
        self.regularizer = l2(self.weight_decay)
        d = 28
        k = 10
        rounds = int((d - 4) / 6)

        x_in = Input(shape = (None, None, 3), name = "input")

        # The first conv layer.
        x = Conv2D(16,
                   (3, 3),
                   strides=(1, 1),
                   name='conv0',
                   padding='same',
                   use_bias=False,
                   kernel_regularizer = self.regularizer) (x_in)
        x = BatchNormalization(momentum = self.batch_norm_momentum,
                               epsilon = self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)

        # Residual blocks
        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 16 * k, projection = True)
            else:
                x = self.res_block(x, 16 * k)

        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 32 * k, strides = (2, 2))
            else:
                x = self.res_block(x, 32 * k)

        for i in range (rounds):
            if i == 0:
                x = self.res_block(x, 64 * k, strides = (2, 2))
            else:
                x = self.res_block(x, 64 * k)

        # The final average pooling layer and fully-connected layer.
        x = GlobalAveragePooling2D()(x)
        y = Dense(self.num_classes, activation = 'softmax', name='fully_connected',
                  kernel_regularizer = self.regularizer,
                  bias_regularizer = self.regularizer)(x)
        return Model(x_in, y, name = "wideresnet28-10")

class cnn ():
    def __init__ (self, weight_decay, num_classes):
        self.weight_decay = weight_decay
        self.regularizer = l2(self.weight_decay)
        self.num_classes = num_classes

    def build_model (self):
        x_in = Input(shape=(None, None, 1), name="input")

        conv1 = Conv2D(
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation='relu')(x_in)

        pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(conv1)

        conv2 = Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation='relu')(pool1)

        pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(conv2)

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        dense = tf.keras.layers.Dense(units=2048, activation=tf.nn.relu)(pool2_flat)

        logits = tf.keras.layers.Dense(units=self.num_classes)(dense)

        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        #y = predictions["probabilities"]
        y = logits
        return Model(x_in, y, name = "cnn")
    
# FedPARA
class LowRank(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, low_rank, kernel_size=None):
        super(LowRank, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.low_rank = low_rank
        self.kernel_size = kernel_size

        if kernel_size:
            # Convolution layer
            self.T = self.add_weight(
                shape=(low_rank, low_rank, kernel_size, kernel_size),
                initializer=self._init_initializer("T"),
                trainable=True,
                name="T"
            )
        else:
            # FC layer
            self.T = self.add_weight(
                shape=(low_rank, low_rank),
                initializer=self._init_initializer("T"),
                trainable=True,
                name="T"
            )

        self.O = self.add_weight(
            shape=(low_rank, out_channels),
            initializer=self._init_initializer("O"),
            trainable=True,
            name="O"
        )

        self.I = self.add_weight(
            shape=(low_rank, in_channels),
            initializer=self._init_initializer("I"),
            trainable=True,
            name="I"
        )

    def _init_initializer(self, tensor_name):
        def initializer(shape, dtype=None):
            fan = shape[-2] if tensor_name == "O" else shape[-1]
            gain = np.sqrt(2.0)
            std = gain / np.sqrt(fan)
            return tf.random.normal(shape, mean=0.0, stddev=std, dtype=dtype)
        return initializer

    def call(self):
        if self.kernel_size:
            # Conv2D
            W = tf.einsum("xyzw,xo,yi->zwio", self.T, self.O, self.I)
        else:
            # Dense
            W = tf.einsum("xy,yi,xo->io", self.T, self.I, self.O)
        return W
    
class Conv2DLowRank(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same", use_bias=False, ratio=1.0):
        super(Conv2DLowRank, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.ratio = ratio
        self.low_rank = self._calc_low_rank()

        self.W1 = LowRank(in_channels, out_channels, self.low_rank, kernel_size=self.kernel_size[0])
        self.W2 = LowRank(in_channels, out_channels, self.low_rank, kernel_size=self.kernel_size[0])

        if use_bias:
            self.bias = self.add_weight(
                shape=(out_channels,), initializer="zeros", trainable=True, name="bias"
            )
        else:
            self.bias = None

    def _calc_low_rank(self):
        r1 = int(np.ceil(np.sqrt(self.out_channels)))
        r2 = int(np.ceil(np.sqrt(self.in_channels)))
        r = max(r1, r2)
        num_target_params = self.out_channels * self.in_channels * (self.kernel_size[0] ** 2) * self.ratio
        r3 = np.sqrt(
            ((self.out_channels + self.in_channels) ** 2) / (4 * (self.kernel_size[0] ** 4)) +
            num_target_params / (2 * (self.kernel_size[0] ** 2))
        ) - (self.out_channels + self.in_channels) / (2 * (self.kernel_size[0] ** 2))
        return max(r, int(np.ceil(r3)))

    def call(self, inputs, **kwargs):
        W1 = self.W1.call()
        W2 = self.W2.call()
        W = W1 * W2
        output = tf.nn.conv2d(
            inputs, W, strides=[1, self.stride, self.stride, 1], padding=self.padding
        )
        if self.bias is not None:
            output = tf.nn.bias_add(output, self.bias)
        return output


class DenseLowRank(tf.keras.layers.Layer):
    def __init__(self, out_units, ratio):
        super(DenseLowRank, self).__init__()
        self.out_units = out_units
        self.ratio = ratio
        self.W1 = None
        self.W2 = None
        self.bias = None

    def build(self, input_shape):

        in_units = input_shape[-1] 
        low_rank = self._calc_low_rank(in_units)

        self.W1 = LowRank(in_units, self.out_units, low_rank)
        self.W2 = LowRank(in_units, self.out_units, low_rank)

        self.bias = self.add_weight(
            shape=(self.out_units,), initializer="zeros", trainable=True, name="bias"
        )
        super(DenseLowRank, self).build(input_shape) 

    def _calc_low_rank(self, in_units):
        r1 = int(np.ceil(np.sqrt(self.out_units)))
        r2 = int(np.ceil(np.sqrt(in_units)))
        num_target_params = self.out_units * in_units * self.ratio
        r3 = np.sqrt(
            ((self.out_units + in_units) ** 2) / 4 +
            num_target_params / 2
        ) - (self.out_units + in_units) / 2
        return max(r1, r2, int(np.ceil(r3)))

    def call(self, inputs, **kwargs):
        if self.W1 is None or self.W2 is None:
            raise ValueError("Weights are not initialized. Ensure the layer is built.")
        W1 = self.W1.call()
        W2 = self.W2.call()

        W = W1 * W2

        output = tf.matmul(inputs, W)
        output = tf.nn.bias_add(output, self.bias)
        return output
        
class DenseLowRank_distil(tf.keras.layers.Layer):
    def __init__(self, out_units, ratio=1.0, activation=None):
        super(DenseLowRank_distil, self).__init__()
        self.out_units = out_units
        self.ratio = ratio
        self.activation = tf.keras.activations.get(activation)
        self.is_dense = ratio == 1.0  
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        in_units = input_shape[-1]

        if self.is_dense:
            self.kernel = self.add_weight(
                shape=(in_units, self.out_units),
                initializer="glorot_uniform",
                trainable=True,
                name="kernel"
            )
        else:
            # Low-rank 
            low_rank = self._calc_low_rank(in_units)

            self.W1 = self.add_weight(
                shape=(in_units, low_rank),
                initializer="glorot_uniform",
                trainable=True,
                name="W1"
            )
            self.W2 = self.add_weight(
                shape=(low_rank, self.out_units),
                initializer="glorot_uniform",
                trainable=True,
                name="W2"
            )

        self.bias = self.add_weight(
            shape=(self.out_units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )
        super(DenseLowRank_distil, self).build(input_shape)

    def _calc_low_rank(self, in_units):
        target_params = self.out_units * in_units * self.ratio
        return max(1, int(np.ceil(target_params / (in_units + self.out_units))))

    def call(self, inputs, **kwargs):
        if self.is_dense:
            output = tf.matmul(inputs, self.kernel)
        else:
            low_rank_mat = tf.matmul(inputs, self.W1)
            output = tf.matmul(low_rank_mat, self.W2)

        output = tf.nn.bias_add(output, self.bias)

        if self.activation:
            output = self.activation(output)
        return output
    
class distilBertLowRank:
    def __init__(self, weight_decay, sample_length, num_classes, low_rank_ratio=1.0):
        self.weight_decay = weight_decay
        self.regularizer = l2(self.weight_decay)
        self.num_classes = num_classes
        self.sample_length = sample_length
        self.low_rank_ratio = low_rank_ratio

    def replace_dense_layers(self, model):
        main_layer = model.distilbert

        for transformer_layer in main_layer.transformer.layer:
            if hasattr(transformer_layer.ffn, "lin1"):
                transformer_layer.ffn.lin1 = self.replace_with_low_rank(transformer_layer.ffn.lin1)
            if hasattr(transformer_layer.ffn, "lin2"):
                transformer_layer.ffn.lin2 = self.replace_with_low_rank(transformer_layer.ffn.lin2)
        
        return model

    def replace_with_low_rank(self, dense_layer):
        out_units = dense_layer.units
        activation = dense_layer.activation

        new_layer = DenseLowRank_distil(
            out_units=out_units,
            ratio=self.low_rank_ratio,
            activation=activation,
        )
        return new_layer
    
    def build_model(self):
        distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        
        distilbert_model = self.replace_dense_layers(distilbert_model)
        
        input_ids = Input(shape=(self.sample_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.sample_length,), dtype=tf.int32, name='attention_mask')

        distilbert_output = distilbert_model(input_ids, attention_mask=attention_mask)
        cls_token = distilbert_output.last_hidden_state[:, 0, :]  # CLS 토큰 출력

        dense = DenseLowRank_distil(
            out_units=2048,
            ratio=self.low_rank_ratio,
            activation='relu'
        )(cls_token)

        y = Dense(self.num_classes, activation="softmax")(dense)

        # 모델 생성
        model = Model(inputs=[input_ids, attention_mask], outputs=y, name="DistilBERT_LowRank_Model")
        return model


class resnet20_para:
    def __init__(self, weight_decay, num_classes, low_rank_ratio, ratio=1.0):
        self.weight_decay = weight_decay
        self.regularizer = l2(self.weight_decay)
        self.initializer = tf.keras.initializers.GlorotUniform(seed=int(time.time()))
        self.num_classes = num_classes
        self.ratio = ratio
        self.low_rank_ratio = low_rank_ratio

    def conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, low_rank_ratio):
        # If low_rank_ratio is 0, use standard Conv2D
        if low_rank_ratio == 1.0:
            return tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=(stride, stride),
                padding=padding,
                use_bias=use_bias,
                kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer
            )
        else:
            # Use low-rank Conv2D if low_rank_ratio > 0
            return Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_bias=use_bias,
                ratio=low_rank_ratio
            )

    def res_block(self, input_tensor, num_filters, strides=(1, 1), projection=False):
        in_channels = input_tensor.shape[-1]

        # First conv layer in residual block
        x = self.conv_layer(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=strides[0],
            padding="same",
            use_bias=False,
            low_rank_ratio=self.low_rank_ratio
        )(input_tensor)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        # Second conv layer in residual block
        x = self.conv_layer(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding="same",
            use_bias=False,
            low_rank_ratio=self.low_rank_ratio
        )(x)
        x = BatchNormalization(gamma_initializer="zeros")(x)

        # Shortcut connection
        if projection or strides != (1, 1):
            shortcut = self.conv_layer(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=1,
                stride=strides[0],
                padding="same",
                use_bias=False,
                low_rank_ratio=self.low_rank_ratio
            )(input_tensor)
            shortcut = BatchNormalization()(shortcut)
        else:
            shortcut = input_tensor

        x = x + shortcut
        return tf.nn.relu(x)

    def build_model(self):
        x_in = Input(shape=(28, 28, 3), name="input")

        # Initial convolution
        x = self.conv_layer(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding="same",
            use_bias=False,
            low_rank_ratio=self.low_rank_ratio
        )(x_in)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        # Residual blocks
        for i in range(3):
            x = self.res_block(x, int(16 * self.ratio), projection=(i == 0))

        for i in range(3):
            x = self.res_block(x, int(32 * self.ratio), strides=(2, 2) if i == 0 else (1, 1))

        for i in range(3):
            x = self.res_block(x, int(64 * self.ratio), strides=(2, 2) if i == 0 else (1, 1))

        # Final layers
        x = GlobalAveragePooling2D()(x)
        y = Dense(
            self.num_classes,
            kernel_initializer=self.initializer,name='fully_connected',
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
        )(x)
        return Model(x_in, y, name="resnet20")



class cnn_para:
    def __init__(self, weight_decay, num_classes, low_rank_ratio):
        self.weight_decay = weight_decay
        self.regularizer = l2(self.weight_decay)
        self.num_classes = num_classes
        self.low_rank_ratio = low_rank_ratio

    def conv_layer(self, in_channels, filters, kernel_size, padding, activation):
           
        if self.low_rank_ratio == 1.0:
            # Standard Conv2D
            return tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=self.regularizer
            )
        else:
            # Low-rank Conv2D
            return Conv2DLowRank(
                in_channels=in_channels,  
                out_channels=filters,
                kernel_size=kernel_size[0],  
                stride=1,
                padding=padding,
                use_bias=False,
                ratio=self.low_rank_ratio
            )
            
    def dense_layer(self, in_units ,out_units, activation=None):
            
        if self.low_rank_ratio == 1.0:
            # Standard Dense
            return tf.keras.layers.Dense(
                units=out_units,
                activation=activation,
                kernel_regularizer=self.regularizer
            )
        else:
            # Low-rank Dense
            return DenseLowRank(
                out_units=out_units,
                ratio=self.low_rank_ratio
            )
            
    def build_model(self):
        x_in = Input(shape=(28, 28, 1), name="input")

        # # First convolution layer
        # conv1 = self.conv_layer(
        #     in_channels=1,
        #     filters=32,
        #     kernel_size=[5, 5],
        #     padding="same",
        #     activation="relu"
        # )(x_in)

        conv1 = Conv2D(
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation='relu')(x_in)

        pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(conv1)

        # # Second convolution layer
        # conv2 = self.conv_layer(
        #     in_channels=32,
        #     filters=64,
        #     kernel_size=[5, 5],
        #     padding="same",
        #     activation="relu"
        # )(pool1)

        conv2 = Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation='relu')(pool1)
            
        pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(conv2)

        pool2_flat = tf.keras.layers.Flatten()(pool2)

        # Fully connected layers
        dense = self.dense_layer(
            in_units=7 * 7 * 64,
            out_units=2048,
            activation=tf.nn.relu
        )(pool2_flat)

        # logits = self.dense_layer(
        #     in_units=2048,
        #     out_units=self.num_classes
        # )(dense)

        # dense = tf.keras.layers.Dense(units=2048, activation=tf.nn.relu)(pool2_flat)

        logits = tf.keras.layers.Dense(units=self.num_classes)(dense)

        predictions = tf.nn.softmax(logits, name="softmax_tensor")

        return Model(inputs=x_in, outputs=predictions, name="cnn")

class WideResNet28_para:
    def __init__(self, weight_decay, num_classes, low_rank_ratio=1.0):
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.low_rank_ratio = low_rank_ratio
        self.batch_norm_momentum = 0.99
        self.batch_norm_epsilon = 1e-5

    def conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, low_rank_ratio):
        # If low_rank_ratio is 1.0, use standard Conv2D
        if low_rank_ratio == 1.0:
            return tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=(stride, stride),
                padding=padding,
                use_bias=use_bias,
                kernel_regularizer=self.regularizer
            )
        else:
            # Use low-rank Conv2D if low_rank_ratio < 1.0
            return Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_bias=use_bias,
                ratio=low_rank_ratio
            )

    def res_block(self, input_tensor, num_filters, strides=(1, 1), projection=False):
        in_channels = input_tensor.shape[-1]

        # First conv layer in residual block
        x = self.conv_layer(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=strides[0],
            padding="same",
            use_bias=False,
            low_rank_ratio=self.low_rank_ratio
        )(input_tensor)
        x = BatchNormalization(momentum=self.batch_norm_momentum, epsilon=self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)
        x = Dropout(0.3)(x)

        # Second conv layer in residual block
        x = self.conv_layer(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding="same",
            use_bias=False,
            low_rank_ratio=self.low_rank_ratio
        )(x)
        x = BatchNormalization(momentum=self.batch_norm_momentum, gamma_initializer="zeros", epsilon=self.batch_norm_epsilon)(x)

        # Shortcut connection
        if projection or strides != (1, 1):
            shortcut = self.conv_layer(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=1,
                stride=strides[0],
                padding="same",
                use_bias=False,
                low_rank_ratio=self.low_rank_ratio
            )(input_tensor)
            shortcut = BatchNormalization(momentum=self.batch_norm_momentum, epsilon=self.batch_norm_epsilon)(shortcut)
        else:
            shortcut = input_tensor

        x = x + shortcut
        return tf.nn.relu(x)

    def build_model(self):
        self.regularizer = l2(self.weight_decay)
        d = 28
        k = 10
        rounds = int((d - 4) / 6)

        x_in = Input(shape=(None, None, 3), name="input")

        # Initial convolution
        x = self.conv_layer(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding="same",
            use_bias=False,
            low_rank_ratio=self.low_rank_ratio
        )(x_in)
        x = BatchNormalization(momentum=self.batch_norm_momentum, epsilon=self.batch_norm_epsilon)(x)
        x = tf.nn.relu(x)

        # Residual blocks
        for i in range(rounds):
            x = self.res_block(x, 16 * k, projection=(i == 0))

        for i in range(rounds):
            x = self.res_block(x, 32 * k, strides=(2, 2) if i == 0 else (1, 1))

        for i in range(rounds):
            x = self.res_block(x, 64 * k, strides=(2, 2) if i == 0 else (1, 1))

        # Final layers
        x = GlobalAveragePooling2D()(x)
        y = Dense(
            self.num_classes,
            activation="softmax",
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            name="fully_connected"
        )(x)
        return Model(x_in, y, name="wideresnet28-10")
        