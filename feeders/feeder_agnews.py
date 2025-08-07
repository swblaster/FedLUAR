'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2025/08/07
Jisoo Kim
Sunwoo Lee, Ph.D.
<starprin3@inha.edu>
<sunwool@inha.ac.kr>
'''
import os
import random
import time
import math
import pickle
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tensorflow.python.data.experimental import AUTOTUNE
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=float)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

class agnews:
    def __init__ (self, batch_size, num_classes, num_clients, alpha):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.alpha = alpha
        self.train_batch_size = batch_size
        self.valid_batch_size = 100
        train_data = None
        train_label = None
        self.sample_length = 128
        

        # Load the data splits.
        name = "ag_news_subset"
        train_ds = tfds.as_numpy(tfds.load(name = name, split = 'train', as_supervised = True))
        train_data = []
        train_label = []
        for (text, label) in train_ds:
            train_data.append(text.decode('utf-8'))
            train_label.append(label)        
        self.train_data = np.array(train_data)
        self.train_label = np.array(train_label)

        valid_ds = tfds.as_numpy(tfds.load(name = name, split = 'test', as_supervised = True))
        valid_data = []
        valid_label = []
        for (text, label) in valid_ds:
            valid_data.append(text.decode('utf-8'))
            valid_label.append(label)        
        self.valid_data = np.array(valid_data)
        self.valid_label = np.array(valid_label)

        num_valid_samples = self.valid_data.shape[0]

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.train_data = tokenizer(
            list(self.train_data),
            truncation=True,
            padding=True,
            max_length=self.sample_length,
            return_tensors="np"
        )

        self.valid_data = tokenizer(
            list(self.valid_data),
            truncation=True,
            padding=True,
            max_length=self.sample_length,
            return_tensors="np"
        )     

        self.num_valid_batches = int(math.floor(num_valid_samples / (self.valid_batch_size)))
        self.num_valid_samples = self.num_valid_batches * self.valid_batch_size

        self.partitions = {}

        if self.rank == 0:
            for i in range(self.num_clients):
                name = "partitions/agnews" + str(num_classes) + "_" + str(i) + ".txt"
                f = open(name, "r")
                lines = f.readlines()
                partition = []
                for j in range (len(lines)):
                    line = lines[j].split('\n')
                    value = int(line[0])
                    partition.append(value)
                self.partitions[i] = partition
                f.close()
        
        self.partitions = self.comm.bcast(self.partitions, root = 0)

        if self.rank == 0:
            for i in range (len(self.partitions)):
                print ("worker " + str(i) + " has " + str(len(self.partitions[i])) + " samples")

        self.num_local_samples = []
        for i in range (self.num_clients):
            self.num_local_samples.append(len(self.partitions[i]))

        self.train_label = dense_to_one_hot(self.train_label, self.num_classes)
        self.valid_label = dense_to_one_hot(self.valid_label, self.num_classes)

    def samples(self):
        return self.num_local_samples
    
    def train_dataset (self, client_id):
        data_input_ids = []
        data_attention_mask = []
        label = []
        
        for i in range(len(self.partitions[client_id])):
            index = self.partitions[client_id][i]
            data_input_ids.append(self.train_data['input_ids'][index])
            data_attention_mask.append(self.train_data['attention_mask'][index])
            
            label.append(self.train_label[index])

        data_input_ids = np.array(data_input_ids)
        data_attention_mask = np.array(data_attention_mask)
        label = np.array(label)

        data = {'input_ids': data_input_ids, 'attention_mask': data_attention_mask}

        files_ds = tf.data.Dataset.from_tensor_slices((data, label))
        files_ds = files_ds.shuffle(self.num_local_samples[client_id])
        train_ds = files_ds.batch(self.train_batch_size).repeat()
        return train_ds.__iter__()


    def valid_dataset (self):
        valid_data_input_ids = []
        valid_data_attention_mask = []
        valid_data_input_ids = np.array(self.valid_data['input_ids'])
        valid_data_attention_mask = np.array(self.valid_data['attention_mask'])
        data = {'input_ids': valid_data_input_ids, 'attention_mask': valid_data_attention_mask}
        dataset = tf.data.Dataset.from_tensor_slices((data, self.valid_label))
        dataset = dataset.batch(self.valid_batch_size)
        dataset = dataset.repeat()
        return dataset.__iter__()
    
    def reuse_dataset (self):
        data_input_ids = []
        data_attention_mask = []
        label = []

        for i in range(len(self.partitions[0])):
            index = self.partitions[0][i]
            data_input_ids.append(self.train_data['input_ids'][index])
            data_attention_mask.append(self.train_data['attention_mask'][index])
            label.append(self.train_label[index])

        data_input_ids = np.array(data_input_ids)
        data_attention_mask = np.array(data_attention_mask)
        label = np.array(label)
        
        data = {'input_ids': data_input_ids, 'attention_mask': data_attention_mask}

        files_ds = tf.data.Dataset.from_tensor_slices((data, label))
        files_ds = files_ds.shuffle(len(data_input_ids))
        same_ds = files_ds.batch(self.train_batch_size).repeat()
        iterator = same_ds.__iter__()
        data, label = next(iterator)
        return data, label
