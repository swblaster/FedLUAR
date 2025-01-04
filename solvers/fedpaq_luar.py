from mpi4py import MPI
import tensorflow as tf
import math
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean
import config as cfg

class FedPAQ_LUAR:
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

        # Recycle score.
        self.num_recycling_layers = cfg.reuse_layer
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
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        if self.rank == 0:
            print ("FedPAQ + LUAR is the local solver!")

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

    def average_model (self, checkpoint, epoch_id, num_comms):
        if self.rank == 0:
            print(f"recycling layers: {self.recycling_layers}")
        # trainable parameters
        offset = 0
        for i in range (self.num_t_params):
            if len(checkpoint.models[0].trainable_variables[i].shape) > 1:
                if i not in self.recycling_layers: # update
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
                    
                    np_last_t_param = np.array(self.last_t_param[i])
                    self.prev_updates[i] = np.array(update)
                    self.score[offset] = np.linalg.norm(self.prev_updates[i].flatten()) / (np.linalg.norm(np_last_t_param.flatten()) + 1e-6)
                    num_comms[i] += 1

                else: # recycle
                    prev_updates_tensor = tf.convert_to_tensor(self.prev_updates[i])
                    average_param = tf.math.add(self.last_t_param[i], prev_updates_tensor)

                for j in range (self.num_local_workers):
                    checkpoint.models[j].trainable_variables[i].assign(average_param)
                self.last_t_param[i] = average_param
                offset += 1

            else:
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
            f = open(f"num_comms({cfg.optimizer} {cfg.dataset} reuse {cfg.reuse_layer} layers {cfg.quantizer_level}).txt", "a")
            for i in range (len(num_comms)):
                if len(params[i].shape) > 1:
                    f.write("%3d: %d\n" %(i, num_comms[i]))
            f.close()

            f = open(f"comm_cost({cfg.optimizer} {cfg.dataset} reuse {cfg.reuse_layer} layers {cfg.quantizer_level}).txt", "a")
            f.write("actual: %f total: %f cost: %f\n" %(actual_size, total_size, cost))
            f.close()
            print("record complete")