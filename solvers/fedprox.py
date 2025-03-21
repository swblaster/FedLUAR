from mpi4py import MPI
import tensorflow as tf
import math
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean

class FedProx:
    def __init__ (self, num_classes, num_workers, average_interval, mu):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.num_local_workers = int(num_workers / self.size)
        self.average_interval = average_interval
        self.mu = mu
        self.local_optimizers = []
        self.last_t_param = []
        self.last_nt_param = []
        for i in range (self.num_local_workers):
            self.local_optimizers.append(SGD(momentum = 0.9))
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1)
            #self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        if self.rank == 0:
            print ("FedProx is the local solver!")

    @tf.function
    def cross_entropy_batch(self, label, prediction):
        cross_entropy = self.loss_object(label, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def round (self, round_id, models, datasets, local_id):
        if len(self.last_t_param) != len(models[0].trainable_variables):
            for i in range(len(models[0].trainable_variables)):
                self.last_t_param.append(tf.identity(models[0].trainable_variables[i]))
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
            regularization_losses = model.losses
            total_loss = tf.add_n(regularization_losses + [loss])
        grads = tape.gradient(total_loss, model.trainable_variables)

        for i in range(len(model.trainable_variables)):
            grads[i] = grads[i] + self.mu * (model.trainable_variables[i] - self.last_t_param[i])
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, grads
    
    def average_model (self, checkpoints, epoch_id):
        # Trainable variables.
        for i in range(len(checkpoints.models[0].trainable_variables)):
            params = []
            for j in range(self.num_local_workers):
                param = checkpoints.models[j].trainable_variables[i]
                last_param = self.last_t_param[i]
                delta = tf.math.subtract(param, last_param)
                params.append(delta)
                    
            localsum_param = tf.math.add_n(params)
            globalsum_param = self.comm.allreduce(localsum_param, op=MPI.SUM)
            update = globalsum_param / self.num_workers
            global_param = tf.math.add(update, self.last_t_param[i])
            
            for j in range(self.num_local_workers):
                checkpoints.models[j].trainable_variables[i].assign(global_param)
            self.last_t_param[i] = global_param

 
         # Non-trainable variables.
        for i in range (len(checkpoints.models[0].non_trainable_variables)):
            local_params = []
            for j in range (self.num_local_workers):
                local_params.append(checkpoints.models[j].non_trainable_variables[i])
            local_params_sum = tf.math.add_n(local_params)

            global_param = self.comm.allreduce(local_params_sum, op = MPI.SUM) / self.num_workers
            for j in range (self.num_local_workers):
                checkpoints.models[j].non_trainable_variables[i].assign(global_param)
