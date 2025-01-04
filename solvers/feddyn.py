from mpi4py import MPI
import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import Mean

class FedDyn:
    def __init__ (self, num_classes, num_workers, average_interval):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.num_local_workers = num_workers // self.size
        self.average_interval = average_interval
        self.last_param = []

        self.alpha = 0.3
        '''
        cifar10 -> 0.1, 0.01, 0.001
        femnist -> [0.001, 0.1, 0.3, 1]
        '''
        self.prev_grad = [[] for _ in range(self.num_local_workers)]

        # TensorFlow requires one model to be trained with a dedicated optimizer.
        self.local_optimizers = []
        for i in range (self.num_local_workers):
            self.local_optimizers.append(SGD(momentum = 0.9))
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1)
            #self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        
        if self.rank == 0:
            print ("FedDyn is the local solver!")
        
        self.h = []
      
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
        if len(self.prev_grad[local_id]) == 0:
            self.prev_grad[local_id] = [tf.zeros_like(w) for w in model.trainable_variables]

        for i in range(self.average_interval):
            images, labels = dataset.next()
            loss, grads = self.local_train_step(model, optimizer, images, labels, local_id)
            lossmean(loss)
            self.prev_grad[local_id] += grads

        self.prev_grad[local_id] = [tf.zeros_like(w) for w in model.trainable_variables]
        return lossmean
    
    
    def local_train_step (self, model, optimizer, data, label, local_id):
        with tf.GradientTape() as tape:
            prediction = model(data, training = True)
            loss = self.cross_entropy_batch(label, prediction)
      
            # Linear Penalty
            linear_penalty = [0 for i in range(len(model.trainable_variables))]
            for i in range(len(model.trainable_variables)):
                linear_penalty[i] = -1 * tf.reduce_sum(self.prev_grad[local_id][i] * model.trainable_variables[i])
            
            # Quadratic Penalty
            quad_penalty = [0 for i in range(len(model.trainable_variables))]
            for i in range(len(model.trainable_variables)):
                quad_penalty[i] = (self.alpha/2) * tf.norm(model.trainable_variables[i] - self.last_param[i]) ** 2
            
            total_penalty = quad_penalty + linear_penalty
            total_loss = tf.add_n([loss] + total_penalty)
        
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, grads
    
    def average_model (self, checkpoint, epoch_id):
        if len(self.h) == 0:
            self.h = [tf.zeros_like(w) for w in checkpoint.models[0].trainable_variables]

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
            self.h[i] = self.h[i] - self.alpha * update
            h_term = (1/self.alpha) * self.h[i]
            global_param = tf.math.add(update, self.last_param[i])
            global_param = global_param - h_term

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
