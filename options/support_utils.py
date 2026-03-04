import torch

class SupportConsistency:
    """ Maintain support consistency S* and φ_i,t for each client """
    def __init__(self, num_params, window_size=5):
        self.num_params = num_params
        self.window_size = window_size
        self.history = []  # list of previous supports

    def update_support(self, grad_vector, threshold=1e-3):
        support = (grad_vector.abs() > threshold).float()
        self.history.append(support)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        # compute consistent support φ_i,t (average over window)
        phi = torch.stack(self.history, dim=0).mean(dim=0)
        return support, phi