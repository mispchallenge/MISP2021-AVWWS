import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

_int_classes = int 
import random

class MyDistributedSampler(Sampler):
    def get_sq_list(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.toral_size:self.num_replicas]
        return indices

    def __init__(self,dataset,num_replicas=None,rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError('requires distributed package to be acailable')
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = len(self.dataset) // self.num_replicas
        self.total_size = len(self.dataset)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset),generator=g).tolist()
        indices = indices[self.rank:self.toral_size:self.num_replicas]
        return iter(indices)

    
    def __len__(self):
        return self.num_samples
    def set_epoch(self,epoch):
        self.epoch = epoch
