import config as cfg
from mpi4py import MPI
import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import Mean

class LBGM:
    def __init__ (self, num_classes, num_workers, average_interval):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.num_local_workers = num_workers // self.size
        self.average_interval = average_interval

        self.last_param = []
        self.last_delta = []
        self.actual_size = 0
        # TensorFlow requires one model to be trained with a dedicated optimizer.
        self.local_optimizers = []
        for i in range (self.num_local_workers):
            #self.local_optimizers.append(Adam(learning_rate=0.00001))
            self.local_optimizers.append(SGD(momentum = 0.9))
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            #self.loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1)
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        if self.rank == 0:
            print ("LBGM(Look-back gradient multiplier) is the local solver!")
        
        self.LBG = [[] for _ in range(self.num_local_workers)]

    @tf.function
    def cross_entropy_batch(self, label, prediction):
        cross_entropy = self.loss_object(label, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def round (self, round_id, models, datasets, local_id):
        if len(self.last_param) != len(models[0].trainable_variables):
            for i in range(len(models[0].trainable_variables)):
                    self.last_param.append(tf.identity(models[0].trainable_variables[i]))
            self.last_delta = [[] for i in range (len(models[0].trainable_variables))]

        model = models[local_id]
        dataset = datasets[local_id]
        optimizer = self.local_optimizers[local_id]

        # Accumulated gradient at worker k
        asg = []

        lossmean = Mean()
        for i in range(self.average_interval):
            images, labels = dataset.next()
            loss, grads = self.local_train_step(model, optimizer, images, labels)
            if len(asg) == 0:
                asg = [tf.zeros_like(g) for g in grads]

            for layer in range(len(grads)):
                asg[layer] = tf.add(asg[layer], grads[layer])

            lossmean(loss)

        if len(self.LBG[local_id]) == 0:
            self.LBG[local_id] = [tf.zeros_like(g) for g in asg]

        if round_id > 0:
            # Calculate LBP error
            ASG_weights = []
            LBG_weights = []

            for i in range(len(asg)):
                #if len(asg[i].shape) > 1:
                ASG_weights.append(asg[i])
                LBG_weights.append(self.LBG[local_id][i])

            flatten_LBG = tf.concat([tf.reshape(tensor, [-1]) for tensor in LBG_weights], axis=0)
            flatten_ASG = tf.concat([tf.reshape(tensor, [-1]) for tensor in ASG_weights], axis=0)

            dot_product = tf.reduce_sum(flatten_LBG * flatten_ASG)

            norm_LBG = tf.norm(flatten_LBG)
            norm_ASG = tf.norm(flatten_ASG)

            cosine_similarity = dot_product / (norm_LBG * norm_ASG)
            cosine_squared = tf.square(cosine_similarity)

            # LBP error
            lbp = 1 - cosine_squared
            
            if self.rank == 0:
                print(f"\nprocess {self.rank}'s {local_id} worker LBP error: {lbp}")

            # Values for calculating LBC
            squarenorm_LBG = tf.square(norm_LBG)
        else:
            dot_product = 0
            lbp = 0
            squarenorm_LBG = 0
            
        self.LBG[local_id] = asg.copy()
        return lossmean, lbp, dot_product, squarenorm_LBG

    def local_train_step (self, model, optimizer, data, label):
        with tf.GradientTape() as tape:
            prediction = model(data, training = True)
            loss = self.cross_entropy_batch(label, prediction)
            regularization_losses = model.losses
            total_loss = tf.add_n(regularization_losses + [loss])
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, grads

    def average_model (self, checkpoint, epoch_id, s_ks, lbcs):
      
        num_params = 0

        for i in range(len(checkpoint.models[0].trainable_variables)):
            params = []
            for j in range(self.num_local_workers):
                param = checkpoint.models[j].trainable_variables[i]
                last_param = self.last_param[i]
                if s_ks[j] == 1: 
                    delta = tf.multiply(lbcs[j], self.last_delta[i])
                    num_params += 1
                else:
                    delta = tf.math.subtract(param, last_param)
                    size = np.prod(param.shape)
                    num_params += size 
                    
                if s_ks[j] == 0:
                    self.last_delta[i] = delta
                      
                params.append(delta)
                    
            localsum_param = tf.math.add_n(params)
            globalsum_param = self.comm.allreduce(localsum_param, op=MPI.SUM)
            update = globalsum_param / self.num_workers
            global_param = tf.math.add(update, self.last_param[i])
            
            for j in range(self.num_local_workers):
                checkpoint.models[j].trainable_variables[i].assign(global_param)
                    
            self.last_param[i] = global_param

        total_comm_params = self.comm.allreduce(num_params, op=MPI.SUM)
        self.actual_size += total_comm_params

        if self.rank == 0:
            f = open(f"num_comms({cfg.optimizer} {cfg.dataset} threshold {cfg.threshold}).txt", "a")
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
