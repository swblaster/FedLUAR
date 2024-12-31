'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2023/09/25
Sunwoo Lee, Ph.D.
<sunwool@inha.ac.kr>
'''
import numpy as np
import tensorflow as tf
import config as cfg
from train import framework
from mpi4py import MPI
from solvers.fedavg import FedAvg
from solvers.fedlama import FedLAMA
from solvers.fedluar import FedLUAR
from solvers.LBGM import LBGM
from solvers.feddropoutavg import FedDropoutAvg
from solvers.prunefl import PruneFL

from model import resnet20, wideresnet28, cnn, distilBert
from feeders.feeder_cifar import cifar
from feeders.feeder_agnews import agnews
from feeders.feeder_femnist import federated_emnist
                
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    local_rank = rank % len(gpus)

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[local_rank], 'GPU')

    num_clients = int(cfg.num_workers / cfg.active_ratio)

    if cfg.dataset == "cifar10":
        batch_size = cfg.cifar10_config["batch_size"]
        num_epochs = cfg.cifar10_config["epochs"]
        min_lr = cfg.cifar10_config["min_lr"]
        max_lr = cfg.cifar10_config["max_lr"]
        num_classes = cfg.cifar10_config["num_classes"]
        decays = list(cfg.cifar10_config["decay"])
        weight_decay = cfg.cifar10_config["weight_decay"]

        dataset = cifar(batch_size = batch_size,
                        num_workers = cfg.num_workers,
                        num_clients = num_clients,
                        num_classes = num_classes,
                        alpha = cfg.alpha)
    elif cfg.dataset == "cifar100":
        batch_size = cfg.cifar100_config["batch_size"]
        num_epochs = cfg.cifar100_config["epochs"]
        min_lr = cfg.cifar100_config["min_lr"]
        max_lr = cfg.cifar100_config["max_lr"]
        num_classes = cfg.cifar100_config["num_classes"]
        decays = list(cfg.cifar100_config["decay"])
        weight_decay = cfg.cifar100_config["weight_decay"]

        dataset = cifar(batch_size = batch_size,
                        num_workers = cfg.num_workers,
                        num_clients = num_clients,
                        num_classes = num_classes,
                        alpha = cfg.alpha)
    elif cfg.dataset == "femnist":
        batch_size = cfg.femnist_config["batch_size"]
        num_epochs = cfg.femnist_config["epochs"]
        min_lr = cfg.femnist_config["min_lr"]
        max_lr = cfg.femnist_config["max_lr"]
        num_classes = cfg.femnist_config["num_classes"]
        decays = list(cfg.femnist_config["decay"])
        weight_decay = cfg.femnist_config["weight_decay"]

        dataset = federated_emnist(batch_size = batch_size,
                                   num_workers = cfg.num_workers,
                                   num_clients = num_clients,
                                   num_classes = num_classes,
                                   active_ratio = cfg.active_ratio)
    elif cfg.dataset == "agnews":
        batch_size = cfg.agnews_config["batch_size"]
        num_epochs = cfg.agnews_config["epochs"]
        min_lr = cfg.agnews_config["min_lr"]
        max_lr = cfg.agnews_config["max_lr"]
        num_classes = cfg.agnews_config["num_classes"]
        decays = list(cfg.agnews_config["decay"])
        weight_decay = cfg.agnews_config["weight_decay"]

        dataset = agnews(batch_size = batch_size,
                             num_classes = num_classes,
                             num_clients = num_clients,
                             alpha = cfg.alpha)
    else:
        print ("config.py has a wrong dataset definition.\n")
        exit()

    if rank == 0:
        print ("---------------------------------------------------")
        print ("dataset: " + cfg.dataset)
        print ("number of workers: " + str(cfg.num_workers))
        print ("average interval: " + str(cfg.average_interval))
        print ("batch_size: " + str(batch_size))
        print ("training epochs: " + str(num_epochs))
        print ("---------------------------------------------------")

    num_local_workers = cfg.num_workers // size
    models = []
    if cfg.dataset == "cifar10":
        for i in range (num_local_workers):
            models.append(resnet20(weight_decay, num_classes).build_model())
    elif cfg.dataset == "cifar100":
        for i in range (num_local_workers):
            models.append(wideresnet28(weight_decay, num_classes).build_model())
    elif cfg.dataset == "femnist":
        for i in range (num_local_workers):
            models.append(cnn(weight_decay, num_classes).build_model())
    elif cfg.dataset == "agnews":
        for i in range (num_local_workers):
            models.append(distilBert(weight_decay, dataset.sample_length, num_classes).build_model())
    
    if cfg.optimizer == 0:
        solver = FedAvg(num_classes = num_classes,
                        num_workers = cfg.num_workers,
                        average_interval = cfg.average_interval)
    elif cfg.optimizer == 1:
        solver = FedLAMA(num_classes = num_classes,
                         num_workers = cfg.num_workers,
                         average_interval = cfg.average_interval,
                         model = models[0],
                         phi = cfg.phi)
    elif cfg.optimizer == 2:
        solver = FedLUAR(model = models[0],
                         num_classes = num_classes,
                         num_workers = cfg.num_workers,
                         average_interval = cfg.average_interval)
    elif cfg.optimizer == 3:
        solver = LBGM(num_classes = num_classes,
                        num_workers = cfg.num_workers,
                        average_interval = cfg.average_interval)
    elif cfg.optimizer == 4:
        solver = FedDropoutAvg(num_classes = num_classes,
                        num_workers = cfg.num_workers,
                        average_interval = cfg.average_interval)
    elif cfg.optimizer == 5:
        solver = FedDropoutAvg(num_classes = num_classes,
                        num_workers = cfg.num_workers,
                        average_interval = cfg.average_interval)
    else:
        print ("Invalid optimizer option!\n")
        exit()

    trainer = framework(models = models,
                        dataset = dataset,
                        solver = solver,
                        num_epochs = num_epochs,
                        min_lr = min_lr,
                        max_lr = max_lr,
                        decay_epochs = decays,
                        num_classes = num_classes,
                        num_workers = cfg.num_workers,
                        num_clients = num_clients,
                        num_candidates = cfg.num_candidates,
                        average_interval = cfg.average_interval,
                        phi = cfg.phi,
                        do_checkpoint = cfg.checkpoint)
    trainer.train()
