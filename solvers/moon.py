from mpi4py import MPI
import numpy as np
import math
import time
import tensorflow as tf
from util import cos_sim
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean

class Moon:
    def __init__ (self, models, num_classes, num_workers, average_interval):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.num_local_workers = num_workers // self.size
        self.average_interval = average_interval
        self.tau = 1.5
        '''
        FEMNIST: tau(0.5), mu(1)
        CIFAR10: tau(1.5), mu(1)
        '''
        self.mu = 1
        self.last_t_param = []
        self.last_model = models

        self.local_optimizers = []
        for i in range (self.num_local_workers):
            self.local_optimizers.append(SGD(momentum = 0.9))
        #self.local_optimizer = Adam()
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            #self.loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1)
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        if self.rank == 0:
            print ("MOON is the local solver!")

    @tf.function
    def cross_entropy_batch(self, label, prediction):
        cross_entropy = self.loss_object(label, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def MOON_loss (self, data, label, local_model, global_model, last_model):
        # local model in the local traininig phase
        z = local_model(data, training = True)
        # global model
        z_glob = global_model(data, training = True)
        # local model of last round
        z_prev = last_model(data, training = True)

        l_sup = self.cross_entropy_batch(label, z)

        sim1 = cos_sim(z, z_glob)
        sim2 = cos_sim(z, z_prev)
        numerator = np.exp(sim1 / self.tau)
        denominator = np.exp(sim1 / self.tau) + np.exp(sim2 / self.tau)
        l_con = -1 * math.log(numerator / denominator)

        loss = l_sup + self.mu * l_con
        return loss

    def round (self, round_id, models, datasets, local_id):
        if len(self.last_t_param) != len(models[0].trainable_variables):
            for i in range(len(models[0].trainable_variables)):
                self.last_t_param.append(tf.identity(models[0].trainable_variables[i]))

        model = models[local_id]
        global_model = models[local_id]
        last_model = self.last_model[local_id]

        dataset = datasets[local_id]
        optimizer = self.local_optimizers[local_id]    
       
        lossmean = Mean()
        for i in range(self.average_interval):
            images, labels = dataset.next()
            loss, grads = self.local_train_step(model, optimizer, images, labels, global_model, last_model)
            lossmean(loss)
        
        self.last_model[local_id] = model

        return lossmean

    def local_train_step (self, model, optimizer, data, label, global_model, last_model):
        with tf.GradientTape() as tape:
            loss = self.MOON_loss(data, label, model, global_model, last_model)
            regularization_losses = model.losses
            total_loss = tf.add_n(regularization_losses + [loss])
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, grads
    
    def average_model (self, checkpoints, epoch_id):
        # Trainable variables.
        for i in range(len(checkpoints.models[0].trainable_variables)):
            params = []
            for j in range(self.num_local_workers):
                param = checkpoints.models[j].trainable_variables[i]
                last_t_param = self.last_t_param[i]
                delta = tf.math.subtract(param, last_t_param)
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