import numpy as np
import os
import time
import random
import json
import tensorflow as tf
from mpi4py import MPI

class federated_emnist:
    def __init__(self, batch_size, num_workers, num_clients, num_classes, active_ratio, path=None):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.num_classes = num_classes
        self.train_batch_size = batch_size
        self.valid_batch_size = 20
        self.num_clients = num_clients
        self.num_workers = num_workers
        self.num_local_workers = int(num_workers / self.size)
        num_local_devices_per_client = 1

        # Read the training data.
        base_path = "../lmls-fl/leaf/data/femnist/data/train/"
        num_loaded_clients = 0
        x = []
        y = []
        self.num_device_samples = []
        for filename in os.listdir(base_path):
            full_path = os.path.join(base_path, filename)
            with open(full_path, "r") as jfile:
                data = json.load(jfile)
                for i in range (len(data["users"])):
                    user = data["users"][i]
                    train_data = data["user_data"][user]["x"]
                    train_label = data["user_data"][user]["y"]
                    x.append(np.reshape(np.array(train_data), (-1, 28, 28, 1)))
                    y.append(tf.keras.utils.to_categorical(train_label, self.num_classes))
                    num_loaded_clients += 1
                    if num_loaded_clients == self.num_clients:
                        break
                jfile.close()
            if num_loaded_clients == self.num_clients:
                break

        self.train_x = []
        self.train_y = []

        for i in range(self.num_clients):
            temp=np.concatenate((x[i], x[(i+1)%len(x)],x[(i+2)%len(x)],x[(i+3)%len(x)], x[(i+4)%len(x)], x[(i+5)%len(x)], x[(i+6)%len(x)], x[(i+7)%len(x)], x[(i+8)%len(x)], x[(i+9)%len(x)]) ,axis=0)
            self.train_x.append(temp)
            temp2=np.concatenate((y[i], y[(i+1)%len(y)],y[(i+2)%len(y)],y[(i+3)%len(y)], y[(i+4)%len(y)], y[(i+5)%len(y)], y[(i+6)%len(y)], y[(i+7)%len(y)], y[(i+8)%len(y)], y[(i+9)%len(y)]) ,axis=0)
            self.train_y.append(temp2)
            self.num_device_samples.append(len(self.train_x[i]))
            if self.rank == 0:
                print ("client %3d length: %d" %(i, self.num_device_samples[i]))

        self.num_train_samples = sum(self.num_device_samples)
        max_num_device_samples = max(self.num_device_samples)
        self.num_local_train_samples = self.num_local_workers * max_num_device_samples
        self.epoch_length = int(self.num_local_train_samples / self.train_batch_size)

        # Read the validation(test) data.
        base_path = "../lmls-fl/leaf/data/femnist/data/test/"
        num_loaded_clients = 0
        x = []
        y = []
        for filename in os.listdir(base_path):
            full_path = os.path.join(base_path, filename)
            with open(full_path, "r") as jfile:
                data = json.load(jfile)
                for i in range (len(data["users"])):
                    user = data["users"][i]
                    test_data = data["user_data"][user]["x"]
                    test_label = data["user_data"][user]["y"]
                    x.append(np.reshape(np.array(test_data), (-1, 28, 28, 1)))
                    y.append(tf.keras.utils.to_categorical(test_label, self.num_classes))
                    num_loaded_clients += 1
                    if num_loaded_clients == self.num_clients:
                        break
                jfile.close()
            if num_loaded_clients == self.num_clients:
                break

        self.test_x = np.concatenate(x, axis = 0)
        self.test_y = np.concatenate(y, axis = 0)
        self.num_valid_batches = len(self.test_y) // self.valid_batch_size
        self.num_valid_samples = self.num_valid_batches * self.valid_batch_size

        self.devices = np.zeros((self.num_local_workers))
        self.device_steps = np.zeros((self.num_local_workers)).astype(int)

        # Find the number of local training samples for the local clients.
        self.num_local_samples = []
        for i in range (self.num_clients):
            self.num_local_samples.append(len(self.train_x[i]))

    def shuffle(self, devices):
        self.devices = np.copy(devices)
        self.shuffled_index = []
        for i in range (self.num_local_workers):
            self.shuffled_index.append(np.arange(self.num_device_samples[devices[i]], dtype='int32'))
            random.seed(time.time())
            random.shuffle(self.shuffled_index[i])

    def samples(self):
        return self.num_local_samples
    
    def read_train_image(self, indices):
        info = indices.numpy()
        sample_id = info[0]
        device_id = info[1]
        image = self.train_x[device_id][sample_id]
        label = self.train_y[device_id][sample_id]

        return image, label

    def read_test_image(self, sample_id):
        index = sample_id.numpy()
        image = self.test_x[index]
        label = self.test_y[index]
        return image, label

    def train_dataset(self, client_id):
        # Client_id should be the global client ID.
        num_samples = self.num_local_samples[client_id]
        client_index = np.full((num_samples, 1), client_id)
        sample_index = np.reshape(np.arange(num_samples), (num_samples, 1))
        indices = np.concatenate((sample_index, client_index), axis = 1)
        dataset = tf.data.Dataset.from_tensor_slices(indices)
        dataset = dataset.shuffle(num_samples, seed = int(time.time()))
        dataset = dataset.map(lambda x: tf.py_function(self.read_train_image, inp=[x], Tout=[tf.float32, tf.float32]))
        dataset = dataset.batch(self.train_batch_size)
        dataset = dataset.repeat()
        return dataset.__iter__()

    def valid_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_valid_samples))
        dataset = dataset.map(lambda x: tf.py_function(self.read_test_image, inp=[x], Tout=[tf.float32, tf.float32]))
        dataset = dataset.batch(self.valid_batch_size)
        dataset = dataset.repeat()
        return dataset.__iter__()

    def check_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_valid_samples * 100))
        dataset = dataset.map(lambda x: tf.py_function(self.read_test_image, inp=[x], Tout=[tf.float32, tf.float32]))
        dataset = dataset.batch(self.valid_batch_size * 100)
        dataset = dataset.repeat()
        return dataset.__iter__()
