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

# For Fedpara
low_rank_ratio = 0.2

# For LBGM
threshold = 0.99

# For PruneFL
reconfiguration_iteration = 50
comm = 0.7

# For Feddropoutavg
dropout_rate = 0.75

# For FedPAQ
quantizer_level = 16

# For FedLUAR
reuse_layer = 10

# For FedProx
mu = 0.001

num_processes_per_node = 8
dataset = "cifar10"
average_interval = 20
phi = 1
num_workers = 32
num_candidates = 48
checkpoint = 0

'''
0: FedAvg
1: FedLAMA
2: FedLUAR
3: LBGM
4: FedDropoutavg
5: PruneFL
6: FedPara
7: FedPAQ
8: FedPAQ + FedLUAR
9: FedDyn
10: FedOpt
11: FedProx
12: FedMut
13: Moon
14: FedACG
'''
optimizer = 2

'''
Federated Learning settings
1. Device activation ratio (0.25, 0.5, 1)
2. Dirichlet's concentration parameter (0.1, 0.5, 1)
'''
active_ratio = 0.25
alpha = 0.1

# while pgrep -f "mpirun -n 2 python3 main.py" > /dev/null; do sleep 1; done && mpirun -n 2 python3 main.py
