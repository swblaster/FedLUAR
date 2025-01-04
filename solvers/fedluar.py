from mpi4py import MPI
import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean

class FedLUAR:
    def __init__ (self, model, num_classes, num_workers, average_interval):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.num_local_workers = num_workers // self.size
        self.average_interval = average_interval

        # Recycle score.
        self.num_recycling_layers = 12
        self.recycling_layers = []
        self.prev_params = []
        self.prev_updates = []
        for i in range (len(model.trainable_variables)):
            self.prev_params.append(np.copy(model.trainable_variables[i].numpy()))
            self.prev_updates.append(np.zeros_like(model.trainable_variables[i].numpy()))

        # TensorFlow requires one model to be trained with a dedicated optimizer.
        self.local_optimizers = []
        for i in range (self.num_local_workers):
            self.local_optimizers.append(SGD(momentum = 0.9))
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        if self.rank == 0:
            print ("FedLUAR solver!")

        self.num_comms = np.zeros((len(model.trainable_variables)))
        self.num_params = np.zeros((len(model.trainable_variables)))
        for i in range (len(model.trainable_variables)):
            self.num_params[i] = np.prod(model.trainable_variables[i].shape)

        self.kernels = []
        for i in range (len(model.trainable_variables)):
            if len(model.trainable_variables[i].shape) > 1:
                self.kernels.append(i)
        self.kernels = np.array(self.kernels)
        self.score = np.zeros((len(self.kernels)))

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
        with tf.GradientTape() as tape:
            prediction = model(data, training = True)
            loss = self.cross_entropy_batch(label, prediction)
            regularization_losses = model.losses
            total_loss = tf.add_n(regularization_losses + [loss])
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, grads

    def average_model (self, checkpoint, epoch_id):
        # Receive the updates of the active layers.
        offset = 0
        for i in range (len(checkpoint.models[0].trainable_variables)):
            if len(checkpoint.models[0].trainable_variables[i].shape) > 1:
                if i not in self.recycling_layers: # update
                    local_params = []
                    for j in range (self.num_local_workers):
                        local_params.append(checkpoint.models[j].trainable_variables[i])
                    local_params_sum = tf.math.add_n(local_params)
                    new_param = np.array(self.comm.allreduce(local_params_sum, op = MPI.SUM) / self.num_workers)

                    self.prev_updates[i] = np.subtract(new_param, self.prev_params[i])
                    self.score[offset] = np.linalg.norm(self.prev_updates[i].flatten()) / (np.linalg.norm(self.prev_params[i].flatten()) + 1e-6)
                else: # recycle
                    new_param = np.add(self.prev_params[i], self.prev_updates[i])

                for j in range (self.num_local_workers):
                    checkpoint.models[j].trainable_variables[i].assign(new_param)
                self.prev_params[i] = new_param
                offset += 1
            else:
                local_params = []
                for j in range (self.num_local_workers):
                    local_params.append(checkpoint.models[j].trainable_variables[i])
                local_params_sum = tf.math.add_n(local_params)
                global_param = self.comm.allreduce(local_params_sum, op = MPI.SUM) / self.num_workers
                for j in range (self.num_local_workers):
                    checkpoint.models[j].trainable_variables[i].assign(global_param)

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
            if self.rank == 0:
                f = open("weight.txt", "w")
                for i in range (len(weight)):
                    f.write("%2d: %f %s\n" %(i, weight[i], str(checkpoint.models[0].trainable_variables[self.kernels[i]].shape)))
                f.close()

                total = 0
                max_cost = 0
                for i in range (len(checkpoint.models[0].trainable_variables)):
                    max_cost += ((epoch_id + 1) * self.num_params[i])
                    total += (self.num_comms[i] * self.num_params[i])
                cost = total * 100 / (max_cost + 1e-6)
                print ("comm: %f percent\n" %(cost))