from mpi4py import MPI
import tensorflow as tf
import math
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean

class FedPAQ:
    def __init__ (self, model, num_classes, num_workers, average_interval, quantizer_level):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.num_local_workers = num_workers // self.size
        self.average_interval = average_interval

        self.num_t_params = len(model.trainable_variables)
        self.num_nt_params = len(model.non_trainable_variables)
        self.last_t_param = []
        self.last_nt_param = []
        self.quantizer_level = quantizer_level

        # TensorFlow requires one model to be trained with a dedicated optimizer.
        self.local_optimizers = []
        for i in range (self.num_local_workers):
            self.local_optimizers.append(SGD(momentum = 0.9))
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        if self.rank == 0:
            print ("FedPAQ is the local solver!")

    @tf.function
    def quantize(self, x):
        """quantize the tensor x in d level on the absolute value coef wise"""
        if self.quantizer_level == 0:
            return x
        else:
            if tf.math.reduce_all(tf.math.equal(x, 0)):
                return x
            else:
                norm = tf.norm(x)
                level_float = self.quantizer_level * tf.math.abs(x) / norm
                previous_level = tf.math.floor(level_float)
                is_next_level = np.random.rand(*x.shape) <= (level_float - previous_level)
                new_level = previous_level + tf.cast(is_next_level, tf.float32)
                return tf.math.sign(x) * norm * new_level / self.quantizer_level

    @tf.function
    def cross_entropy_batch(self, label, prediction):
        cross_entropy = self.loss_object(label, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def round (self, round_id, models, datasets, local_id):
        model = models[local_id]
        dataset = datasets[local_id]
        optimizer = self.local_optimizers[local_id]

        lossmean = Mean()
        for i in range(self.average_interval):
            images, labels = dataset.next()
            loss, grads = self.local_train_step(model, optimizer, images, labels)
            lossmean(loss)
        return lossmean

    def local_train_step (self, model, optimizer, data, label):
        #init last model
        if len(self.last_t_param) != self.num_t_params:
            for i in range(self.num_t_params):
                self.last_t_param.append(tf.identity(model.trainable_variables[i]))
        if len(self.last_nt_param) != self.num_nt_params:
            for i in range(self.num_nt_params):
                self.last_nt_param.append(tf.identity(model.non_trainable_variables[i]))
        with tf.GradientTape() as tape:
            prediction = model(data, training = True)
            loss = self.cross_entropy_batch(label, prediction)
            regularization_losses = model.losses
            total_loss = tf.add_n(regularization_losses + [loss])
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, grads

    def average_model (self, checkpoint, epoch_id):
        # trainable parameters
        for i in range (self.num_t_params):
            params = []
            for j in range (self.num_local_workers):
                param = checkpoint.models[j].trainable_variables[i]
                last_param = self.last_t_param[i]
                delta = tf.math.subtract(param, last_param)
                delta = self.quantize(delta)
                params.append(delta)
            localsum_param = tf.math.add_n(params)
            globalsum_param = self.comm.allreduce(localsum_param, op = MPI.SUM)
            update = globalsum_param / self.num_workers
            update = self.quantize(update)
            average_param = tf.math.add(update, self.last_t_param[i])
            for j in range (self.num_local_workers):
                checkpoint.models[j].trainable_variables[i].assign(average_param)
            self.last_t_param[i] = average_param

        # non-trainable parameters (BN statistics)
        for i in range (self.num_nt_params):
            params = []
            for j in range (self.num_local_workers):
                param = checkpoint.models[j].non_trainable_variables[i]
                last_param = self.last_nt_param[i]
                delta = tf.math.subtract(param, last_param)
                delta = self.quantize(delta)
                params.append(delta)
            localsum_param = tf.math.add_n(params)
            globalsum_param = self.comm.allreduce(localsum_param, op = MPI.SUM)
            update = globalsum_param / self.num_workers
            update = self.quantize(update)
            average_param = tf.math.add(update, self.last_nt_param[i])
            for j in range (self.num_local_workers):
                checkpoint.models[j].non_trainable_variables[i].assign(average_param)
            self.last_nt_param[i] = average_param
