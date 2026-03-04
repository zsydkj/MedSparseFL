import torch
import numpy as np

class CountSketch:
    """ CountSketch utility for compressing gradients """
    def __init__(self, num_hash=5, sketch_size=1024):
        self.num_hash = num_hash
        self.sketch_size = sketch_size
        self.hash_funcs = [lambda x, i=i: (hash((x, i)) % sketch_size) for i in range(num_hash)]
        self.sign_funcs = [lambda x, i=i: 1 if hash((x, i, 'sign')) % 2 == 0 else -1 for i in range(num_hash)]

    def sketch(self, grad_vector):
        sketch = torch.zeros(self.num_hash, self.sketch_size, device=grad_vector.device)
        for h in range(self.num_hash):
            for idx, val in enumerate(grad_vector):
                sketch[h, self.hash_funcs[h](idx)] += self.sign_funcs[h](idx) * val
        return sketch

    def recover(self, sketch, grad_size):
        grad_est = torch.zeros(grad_size, device=sketch.device)
        for i in range(grad_size):
            estimates = [sketch[h, self.hash_funcs[h](i)] * self.sign_funcs[h](i) for h in range(self.num_hash)]
            grad_est[i] = torch.median(torch.tensor(estimates, device=grad_est.device))
        return grad_est


class Masking:
    """ Simple masking for privacy """
    @staticmethod
    def apply_mask(grad_vector, mask_ratio=0.1):
        mask = (torch.rand_like(grad_vector) > mask_ratio).float()
        return grad_vector * mask

class HomomorphicEncryption:
    """ Placeholder for HE operations """
    @staticmethod
    def encrypt(tensor):
        # In practice use a proper HE library
        return tensor + 0.01  # dummy noise

    @staticmethod
    def decrypt(tensor):
        return tensor - 0.01  # remove dummy noise