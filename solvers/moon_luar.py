from mpi4py import MPI
import numpy as np
import math
import time
import tensorflow as tf
from util import cos_sim
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
import config as cfg

class Moon_LUAR:
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

        # Recycle score.
        self.num_recycling_layers = cfg.reuse_layer
        self.recycling_layers = []
        self.prev_updates = []
        for i in range (len(models[0].trainable_variables)):
            self.prev_updates.append(np.zeros_like(models[0].trainable_variables[i].numpy()))

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
            print ("MOON + FedLUAR is the local solver!")

        self.num_comms = np.zeros((len(models[0].trainable_variables)))
        self.num_params = np.zeros((len(models[0].trainable_variables)))
        for i in range (len(models[0].trainable_variables)):
            self.num_params[i] = np.prod(models[0].trainable_variables[i].shape)

        self.kernels = []
        for i in range (len(models[0].trainable_variables)):
            if len(models[0].trainable_variables[i].shape) > 1:
                self.kernels.append(i)
        self.kernels = np.array(self.kernels)
        self.score = np.zeros((len(self.kernels)))

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
    
    def average_model (self, checkpoint, epoch_id, num_comms):
        if self.rank == 0:
            print(f"layer: {self.recycling_layers}")
            
        offset = 0
        for i in range (len(checkpoint.models[0].trainable_variables)):
            if len(checkpoint.models[0].trainable_variables[i].shape) > 1:
                if i not in self.recycling_layers: # update
                    local_params = []
                    for j in range(self.num_local_workers):
                        param = checkpoint.models[j].trainable_variables[i]
                        last_param = self.last_t_param[i]
                        delta = tf.math.subtract(param, last_param)
                        local_params.append(delta)
                    localsum_param = tf.math.add_n(local_params)
                    globalsum_param = self.comm.allreduce(localsum_param, op=MPI.SUM)
                    update = globalsum_param / self.num_workers
                    new_param = tf.math.add(update, self.last_t_param[i])

            
                    np_last_t_param = np.array(self.last_t_param[i])
                    self.prev_updates[i] = np.array(update)
                    self.score[offset] = np.linalg.norm(self.prev_updates[i].flatten()) / (np.linalg.norm(np_last_t_param.flatten()) + 1e-6)
                    num_comms[i] += 1

                else: # recycle
                    prev_updates_tensor = tf.convert_to_tensor(self.prev_updates[i])
                    new_param = tf.math.add(self.last_t_param[i], prev_updates_tensor)

                for j in range (self.num_local_workers):
                    checkpoint.models[j].trainable_variables[i].assign(new_param)
                self.last_t_param[i] = new_param
                offset += 1

            else:
                local_params = []
                for j in range(self.num_local_workers):
                    param = checkpoint.models[j].trainable_variables[i]
                    last_param = self.last_t_param[i]
                    delta = tf.math.subtract(param, last_param)
                    local_params.append(delta)
                        
                localsum_param = tf.math.add_n(local_params)
                globalsum_param = self.comm.allreduce(localsum_param, op=MPI.SUM)
                update = globalsum_param / self.num_workers
                global_param = tf.math.add(update, self.last_t_param[i])

                for j in range (self.num_local_workers):
                    checkpoint.models[j].trainable_variables[i].assign(global_param)
                self.last_t_param[i] = global_param

        # Sample the recycling layers for the next round.
        self.sample_recycling_layers(checkpoint, epoch_id)

        # Non-trainable variables.
        for i in range (len(checkpoint.models[0].non_trainable_variables)):
            local_params = []
            for j in range (self.num_local_workers):
                local_params.append(checkpoint.models[j].non_trainable_variables[i])
            local_params_sum = tf.math.add_n(local_params)

            global_param = self.comm.allreduce(local_params_sum, op = MPI.SUM) / self.num_workers
            for j in range (self.num_local_workers):
                checkpoint.models[j].non_trainable_variables[i].assign(global_param)
    
    def sample_recycling_layers (self, checkpoint, epoch_id):
        if self.num_recycling_layers > 0:
            sum_inverse_scores = sum(np.reciprocal(self.score))
            weight = np.reciprocal(self.score) / sum_inverse_scores
            self.recycling_layers = np.random.choice(np.arange(len(self.kernels)), size = self.num_recycling_layers, replace = False, p = weight)
            self.recycling_layers = self.kernels[np.array(self.comm.bcast(self.recycling_layers, root = 0))]
            index = np.delete(np.arange(len(self.num_comms)), self.recycling_layers)
            self.num_comms[index] += 1

    def count_comms (self, checkpoint, num_epochs, num_comms):
        params = checkpoint.models[0].trainable_variables
        total_size = 0
        actual_size = 0
        for i in range (len(params)):
            param = params[i]
            size = np.prod(param.shape)
            actual_size += num_comms[i] / num_epochs * size
            total_size += size
        cost = actual_size / total_size

        if self.rank == 0:
            f = open(f"num_comms({cfg.optimizer} {cfg.dataset} {cfg.reuse_layer}).txt", "a")
            for i in range (len(num_comms)):
                if len(params[i].shape) > 1:
                    f.write("%3d: %d\n" %(i, num_comms[i]))
            f.close()

            f = open(f"comm_cost({cfg.optimizer} {cfg.dataset} {cfg.reuse_layer}).txt", "a")
            f.write("actual: %f total: %f cost: %f\n" %(actual_size, total_size, cost))
            f.close()
            print("record complete")