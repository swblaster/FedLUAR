from mpi4py import MPI
import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import Mean
import config as cfg

class PruneFL:
    def __init__ (self, num_classes, num_workers, average_interval):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.num_local_workers = num_workers // self.size
        self.average_interval = average_interval
        self.last_param = []

        self.reconfiguration_iteration = cfg.reconfiguration_iteration
        # The set of i mportance measure
        self.importance_measures = [[] for _ in range(self.num_local_workers)]
        self.global_importance = []
        # Mask vector
        self.mask = []
        
        self.t = cfg.times
        
        # TensorFlow requires one model to be trained with a dedicated optimizer.
        self.local_optimizers = []
        for i in range (self.num_local_workers):
            self.local_optimizers.append(SGD(momentum = 0.9))
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            #self.loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1)
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        if self.rank == 0:
            print ("PruneFL is the local solver!")

    @tf.function
    def cross_entropy_batch(self, label, prediction):
        cross_entropy = self.loss_object(label, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def round (self, round_id, models, datasets, local_id, mask):
        if len(self.last_param) != len(models[0].trainable_variables):
            for i in range(len(models[0].trainable_variables)):
                self.last_param.append(tf.identity(models[0].trainable_variables[i]))
        model = models[local_id]
        grads = [tf.zeros_like(i) for i in models[0].trainable_variables]
        dataset = datasets[local_id]
        optimizer = self.local_optimizers[local_id]    

        lossmean = Mean()
        for i in range(self.average_interval):
            images, labels = dataset.next()
            loss, grad = self.local_train_step(model, optimizer, images, labels, mask)
            lossmean(loss)

            grads = [tf.add(g1, g2) for g1, g2 in zip(grads, grad)]  

        importance = self.compute_importance(grads)
        if len(self.importance_measures[local_id]) == 0:
            self.importance_measures[local_id] = [tf.zeros_like(i) for i in importance]
        
        # Add importance measure (Algorithm1 6)
        self.importance_measures[local_id] = [tf.add(i1, i2) for i1, i2 in zip(self.importance_measures[local_id], importance)]

        return lossmean

    def local_train_step (self, model, optimizer, data, label, mask):
        with tf.GradientTape() as tape:
            prediction = model(data, training = True)
            loss = self.cross_entropy_batch(label, prediction)
            regularization_losses = model.losses
            total_loss = tf.add_n(regularization_losses + [loss])
        grads = tape.gradient(total_loss, model.trainable_variables)
        
        # Mask gradient and update weights parameter (Algorithm1 line 4-5)
        masked_grads = [ g * m if len(g.shape) > 1 else g for g, m in zip(grads, mask)]
        optimizer.apply_gradients(zip(masked_grads, model.trainable_variables))
        return loss, masked_grads

    def average_model (self, checkpoint, epoch_id):
        if len(self.mask) == 0:
            self.mask = [tf.ones_like(w) for w in checkpoint.models[0].trainable_variables]
        total_sum = 0
        # Trainable variables.
        for i in range(len(checkpoint.models[0].trainable_variables)):
            params = []
            for j in range(self.num_local_workers):
                param = checkpoint.models[j].trainable_variables[i]
                last_param = self.last_param[i]
                delta = tf.math.subtract(param, last_param)
                params.append(delta)
                num_ones = tf.reduce_sum(tf.cast(self.mask[i] == 1, tf.int32)).numpy()
                total_sum += num_ones

            localsum_param = tf.math.add_n(params)
            globalsum_param = self.comm.allreduce(localsum_param, op=MPI.SUM)
            update = globalsum_param / self.num_workers
            global_param = tf.math.add(update, self.last_param[i])
            
            # Reconfiguration round
            if epoch_id % self.reconfiguration_iteration == 1:
                if len(self.global_importance) == 0:
                    self.global_importance = [tf.zeros_like(w) for w in checkpoint.models[0].trainable_variables]

                # Initialize importance_sum
                importance_sum = [tf.zeros_like(w) for w in checkpoint.models[0].trainable_variables[i]]
                if len(checkpoint.models[0].trainable_variables[i].shape) > 1:
                
                # Upload importance measure (Algorithm1 line 11)
                    for j in range(self.num_local_workers):
                        importance_sum = tf.add(self.importance_measures[j][i], importance_sum)

                    # Aggregate the received importance measure (Algorithm1 line 12)
                    self.global_importance[i] = self.comm.allreduce(importance_sum, op=MPI.SUM) / self.num_workers
                
                # Find optimal subset of weights parameters
                if len(checkpoint.models[0].trainable_variables[i].shape) > 1:
                    #t = self.t[i]
                    if self.rank == 0:
                        print(f"start reconfiguration for {i}'s layer")
                    pruned_weights, mask = self.reconfiguration(global_param, self.global_importance[i])
                    checkpoint.models[0].trainable_variables[i].assign(pruned_weights)
                    self.mask[i] = mask
                    if self.rank == 0:
                        print(f"reconfiguration for {i}'s layer")
                else:
                    checkpoint.models[0].trainable_variables[i].assign(global_param)

            # Non-reconfiguration round
            else:
                # Send global model
                for j in range(self.num_local_workers):
                    checkpoint.models[j].trainable_variables[i].assign(global_param)
                self.last_param[i] = global_param

        # Reset the set of importance measure (Algorithm 14) 
        if epoch_id % self.reconfiguration_iteration == 1:
            self.importance_measures = [[] for _ in range(self.num_local_workers)]
            self.global_importance = []

        total_comm_params = self.comm.allreduce(total_sum, op=MPI.SUM)

        if self.rank == 0:
            f = open(f"num_comms({cfg.optimizer} {cfg.dataset} {cfg.reconfiguration_iteration} {cfg.comm}).txt", "a")
            f.write(str(total_comm_params) + "\n")
            f.close()

        # Non-trainable variables.
        for i in range (len(checkpoint.models[0].non_trainable_variables)):
            local_params = []
            for j in range (self.num_local_workers):
                local_params.append(checkpoint.models[j].non_trainable_variables[i])
            local_params_sum = tf.math.add_n(local_params)

            global_param = self.comm.allreduce(local_params_sum, op = MPI.SUM) / self.num_workers
            for j in range (self.num_local_workers):
                checkpoint.models[j].non_trainable_variables[i].assign(global_param)
        
        return self.mask
    
    def compute_importance(self, grads):
        # Algorithm line 6
        importance = []
        for grad in grads:
            if len(grad) > 1:
                importance.append(tf.square(grad))
            else: 
                importance.append(tf.constant(0.0, dtype=tf.float32))
        return importance
    
    def reconfiguration(self, weights, importance_measures):
            A = set()
            importance_flat = tf.reshape(importance_measures, [-1])
            S = tf.argsort(importance_flat, direction='DESCENDING')

            if self.rank == 0:
                print(f"total number of parameters: {len(importance_flat)}")
            
            percent = int(len(S) * cfg.comm)

            A = set(S.numpy()[:percent])

            if self.rank == 0:
                print(f"Number of selected parameters (top {(cfg.comm) * 100}%): {len(A)}")

            # Generate mask vector
            mask_flat = tf.zeros_like(importance_flat)
            mask_flat = tf.tensor_scatter_nd_update(
                mask_flat,
                indices=tf.expand_dims(tf.constant(list(A), dtype=tf.int32), axis=1),
                updates=tf.ones(len(A), dtype=tf.float32)
            )

            mask = tf.reshape(mask_flat, weights.shape)
            weights = tf.multiply(weights, mask)

            return weights, mask