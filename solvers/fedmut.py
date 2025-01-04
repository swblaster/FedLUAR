from mpi4py import MPI
import numpy as np
import math
import random
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean

class FedMut:
    def __init__(self, model, num_classes, num_workers, average_interval):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.num_classes = num_classes
        self.num_workers = num_workers

        self.num_local_workers = int(num_workers / self.size)
        self.average_interval = average_interval
        
        # 각 local에 대한 optimizer
        self.local_optimizers = []
        
        
        self.num_t_params = len(model.trainable_variables)
        self.num_nt_params = len(model.non_trainable_variables)

        self.last_param = []

        # HyperParameter for FedMut
        '''
        1.  alpha decides the distance of the mutated model with the global model.
            a higher value alph indicates a greater distance between the mutated model and global model.

            a greater distance can guide the global model to converge into a flater area.
            However, a too-large distance can make the global model difficult to converge.

            In paper, 
            when alpha = 1 -> FedMut == FedAvg
            when alpha < 5 -> the inference accuracy of FedMut  increases with increasing value of alpha
            when alpha == 5 -> FedMut cannot train a usable model.

        2.  Dynamic Preference Mutation Strategy
            beta is also a hyper parameter.
            the higher value of beta leads training faster 
            the higher value of beta  make training converge faster, but if it is too large, it can interfere with training.
        
            In paper,
            when initial_beta = 0.1, 0.3, 0.5 -> converge faster
            when initial_beta = 0.7 -> training fail
        
        These values are not the result of mathematical proofs, but rather the results obtained through experiments in the paper.
        '''
        self.alpha = 0.5

        '''
        FEMNIST: 0.5
        <CIFAR10>
        1.05 x
        1.1 x
        1.0 x (0.49)


        '''
        #self.initial_beta = 0.5
        #self.beta = 0
        
        for i in range(self.num_local_workers):
            self.local_optimizers.append(SGD(momentum = 0.9))
        
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            #self.loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        if self.rank == 0:
            print("FedMut is the local solver")


    # Generate Mutated Model
    def mutation(self, weights_delta, global_param, active_clients, num_layer):

        mutated_model = [[] for _ in range(active_clients)]

        if active_clients % 2 == 1:
            mutated_model = [[tf.Variable(gp) for gp in global_param] for _ in range(active_clients)]

        # Generate random vector
        random_vector = [-1] * (math.floor(active_clients//2)) + [1] * (math.floor(active_clients//2))
        random_vectors = [random_vector[:] for _ in range(num_layer)]

        # Shuffle random vector
        for i in range(num_layer):
            random.shuffle(random_vectors[i])
        
        for i in range(2 * math.floor(active_clients//2)):
            mutated_weights = global_param
            for j in range(num_layer):
                mutation_value = self.alpha * random_vectors[j][i] * weights_delta[j]
                mutated_weights[j] = tf.math.add(global_param[j], mutation_value)
                mutated_model[i].append(mutated_weights[j])

        return mutated_model
    
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
            regularization_losses = model.losses
            total_loss = tf.add_n(regularization_losses + [loss])
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, grads
    
    def average_model(self, checkpoint, epoch_id):
        
        # Current param - last param
        weight_deltas = []
        # List for global model parameter
        global_weights = []
        average_param_store = []

        for i in range(self.num_t_params):
            params = []
            for j in range(self.num_local_workers):
                param = checkpoint.models[j].trainable_variables[i]
                last_param = self.last_param[i]
                delta = tf.math.subtract(param, last_param)
                params.append(delta)
            localsum_param = tf.math.add_n(params)
            globalsum_param = self.comm.allreduce(localsum_param, op = MPI.SUM)
            update = globalsum_param / self.num_workers
            average_param = tf.math.add(update, self.last_param[i])
            average_param_store.append(average_param)

            weight_deltas.append(tf.math.subtract(average_param, self.last_param[i]))
            global_weights.append(average_param)
            
            self.last_param[i] = average_param
        
        
        '''
        In the paper, there is no explanation for value 'Tb'.
        '''

        # Mutation
        mutated_model = self.mutation(weight_deltas, global_weights, self.num_local_workers, len(weight_deltas))
        
        if self.rank == 0:
            print("Completed generating mutated models!")


        # Dispatch mutated model to clients
        for i in range(self.num_t_params):
            for j in range(self.num_local_workers):
                checkpoint.models[j].trainable_variables[i].assign(mutated_model[j][i])

            # Non-trainable variables.
        for i in range (len(checkpoint.models[0].non_trainable_variables)):
            local_params = []
            for j in range (self.num_local_workers):
                local_params.append(checkpoint.models[j].non_trainable_variables[i])
            local_params_sum = tf.math.add_n(local_params)

            global_param = self.comm.allreduce(local_params_sum, op = MPI.SUM) / self.num_workers
            for j in range (self.num_local_workers):
                checkpoint.models[j].non_trainable_variables[i].assign(global_param)
                    