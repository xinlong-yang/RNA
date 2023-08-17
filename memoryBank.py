import torch

class MemoeyBank:
    def __init__(self, banksize, dim, device):
        self.K = banksize
        self.f_dim = dim

        self.features = -1 * torch.ones(self.K, self.f_dim).to(device) 
        self.targets = -1 * torch.ones(self.K, dtype=torch.long).to(device) 
 
        self.ptr = 0
 
    @property
    def is_full(self):
        
        return self.targets[-1].item() != -1 
 
    def get(self):
        if self.is_full:
            return self.features, self.targets
        else:
            return self.features[:self.ptr], self.targets[:self.ptr]
 
 
    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.features[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.features[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size
