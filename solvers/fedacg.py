from mpi4py import MPI
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import Mean

class FedACG:
    def __init__ (self, model, num_classes, num_workers, average_interval):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.num_local_workers = num_workers // self.size
        self.average_interval = average_interval
        self.last_param = []

        self.momentum_term = 0.8
        '''
        the performance of FedACG
        remains stable in a range of  values from 0.75 to 0.9. 

        0.75: 65.02
        0.8: 
        '''
        self.global_momentum = [tf.zeros_like(w) for w in model.trainable_variables]
        self.beta =  0.01
        '''
        0.001, 0.01, 0.1, 1
        '''
        # TensorFlow requires one model to be trained with a dedicated optimizer.
        self.local_optimizers = []
        for i in range (self.num_local_workers):
            self.local_optimizers.append(SGD(momentum = 0.9))
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            #self.loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1)
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            
        if self.rank == 0:
            print ("FedACG is the local solver!")

    @tf.function
    def cross_entropy_batch(self, label, prediction):
        cross_entropy = self.loss_object(label, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def round (self, round_id, models, datasets, local_id):
        if len(self.last_param) != len(models[0].trainable_variables):
            for i in range(len(models[0].trainable_variables)):
                self.last_param.append(tf.identity(models[0].trainable_variables[i]))
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
        with tf.GradientTape() as tape:
            prediction = model(data, training = True)
            loss = self.cross_entropy_batch(label, prediction)
            #regularization_losses = model.losses
            #total_loss = tf.add_n(regularization_losses + [loss])
            
            quadratic_term = (self.beta / 2) * tf.add_n([
            tf.norm(model.trainable_variables[i] - self.last_param[i]) ** 2 for i in range(len(model.trainable_variables))
            ])
            total_loss = loss + quadratic_term

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, grads

    def average_model (self, checkpoint, epoch_id):
        # Trainable variables.
        for i in range(len(checkpoint.models[0].trainable_variables)):
            params = []
    
            for j in range(self.num_local_workers):
                param = checkpoint.models[j].trainable_variables[i]
                last_param = self.last_param[i]
                delta = tf.math.subtract(param, last_param)
                params.append(delta)
                    
            localsum_param = tf.math.add_n(params)
            globalsum_param = self.comm.allreduce(localsum_param, op=MPI.SUM)
            update = globalsum_param / self.num_workers
            self.global_momentum[i] = self.momentum_term * self.global_momentum[i] + update 
            global_param = tf.math.add(self.global_momentum[i], self.last_param[i])
            
            for j in range(self.num_local_workers):
                checkpoint.models[j].trainable_variables[i].assign(global_param)
            self.last_param[i] = global_param

        # Non-trainable variables.
        for i in range (len(checkpoint.models[0].non_trainable_variables)):
            local_params = []
            for j in range (self.num_local_workers):
                local_params.append(checkpoint.models[j].non_trainable_variables[i])
            local_params_sum = tf.math.add_n(local_params)

            global_param = self.comm.allreduce(local_params_sum, op = MPI.SUM) / self.num_workers
            for j in range (self.num_local_workers):
                checkpoint.models[j].non_trainable_variables[i].assign(global_param)
