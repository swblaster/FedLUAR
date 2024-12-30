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
        y = predictions["probabilities"]
        return Model(x_in, y, name = "cnn")
