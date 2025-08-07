'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2025/08/07
Jisoo Kim
Sunwoo Lee, Ph.D.
<starprin3@inha.edu>
<sunwool@inha.ac.kr>
'''
'''
Dataset-specific hyper-parameters.
'''
cifar10_config = {
    "batch_size": 32,
    "min_lr": 0.2,
    "max_lr": 0.2,
    "num_classes": 10,
    "epochs": 200,
    "decay": {100, 150},
    "weight_decay": 0.0001,
}

cifar100_config = {
    "batch_size": 32,
    "min_lr": 0.1,
    "max_lr": 0.4,
    "num_classes": 100,
    "epochs": 300,
    "decay": {150, 200},
    "weight_decay": 0.0005,
}

femnist_config = {
    "batch_size": 20,
    "min_lr": 0.01,
    "max_lr": 0.01,
    "num_classes": 62,
    "epochs": 200, 
    "decay": {100, 150},
    "weight_decay": 0.0001,
}

agnews_config = {
    "batch_size": 128,
    "min_lr": 0.00001,
    "max_lr": 0.00001,
    "num_classes": 4,
    "epochs": 100,
    "decay": {60, 80},
    "weight_decay": 0.0001,
}

recycling_layers = 12

num_processes_per_node = 8
dataset = "cifar10"
average_interval = 20
phi = 1
num_workers = 32
num_candidates = 48
checkpoint = 0

'''
0: FedAvg
1: FedLUAR
'''
optimizer = 1

'''
Federated Learning settings
1. Device activation ratio (0.25, 0.5, 1)
2. Dirichlet's concentration parameter (0.1, 0.5, 1)
'''
active_ratio = 0.25
alpha = 0.1
