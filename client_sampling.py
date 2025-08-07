'''
Large-scale Machine Learning Systems Lab. (LMLS lab)
2025/08/07
Jisoo Kim, M.S.
Sunwoo Lee, Ph.D.
<starprin3@inha.edu>
<sunwool@inha.ac.kr>
'''
from mpi4py import MPI
import numpy as np
import math
import time
import tensorflow as tf

class sampling:
    def __init__ (self, num_clients, num_workers, num_candidates):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_workers = num_workers
        self.num_clients = num_clients
        self.num_candidates = num_candidates
        self.num_local_workers = int(num_workers / self.size)
        self.num_local_clients = int(num_clients / self.size)
        self.local_losses = np.full((self.num_clients), np.Inf)
        self.fixed_losses = np.full((self.num_clients), np.Inf)
        self.local_norms = np.zeros((self.num_clients))
        self.avg_norms = np.zeros((self.num_clients))
        self.num_updates = np.zeros((self.num_clients))
        self.active_devices = np.zeros((self.num_clients))
        self.rng = np.random.default_rng(int(time.time()))
        np.random.seed(int(time.time()))

    def random (self):
        self.active_devices = np.random.choice(np.arange(self.num_clients), size = self.num_workers, replace = False)
        self.active_devices = self.comm.bcast(self.active_devices, root = 0)
        return self.active_devices
