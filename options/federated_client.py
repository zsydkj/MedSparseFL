import torch
import torch.nn as nn
from options.privacy_utils import CountSketch, Masking, HomomorphicEncryption
from options.support_utils import SupportConsistency

class FederatedClient:
    def __init__(self, client_id, data, model, lr=0.01, device='cpu', mask_ratio=0.1, gsn_threshold=0.5):
        self.client_id = client_id
        self.data = data
        self.model = model.to(device)
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.support_tracker = SupportConsistency(sum(p.numel() for p in self.model.parameters()))
        self.cs = CountSketch()
        self.mask_ratio = mask_ratio
        self.gsn_threshold = gsn_threshold
        self.residual = None

    def local_train(self, gsn=None, epochs=1):
        self.model.train()
        for _ in range(epochs):
            for x, y in self.data:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = nn.CrossEntropyLoss()(output, y)
                loss.backward()

                # Flatten gradients
                grad_vector = torch.cat([p.grad.view(-1) for p in self.model.parameters()])

                # Add residual from previous round
                if self.residual is not None:
                    grad_vector += self.residual

                # Compute GSN scores if provided
                if gsn is not None:
                    scores = gsn(grad_vector)
                    mask = (scores > self.gsn_threshold).float()
                    sparse_grad = grad_vector * mask
                else:
                    sparse_grad = grad_vector

                # Update residual
                self.residual = grad_vector - sparse_grad

                # Support consistency
                support, phi = self.support_tracker.update_support(sparse_grad)

                # Privacy: masking + homomorphic encryption
                sparse_grad = Masking.apply_mask(sparse_grad, self.mask_ratio)
                sparse_grad = HomomorphicEncryption.encrypt(sparse_grad)

                # Store for server aggregation
                self.last_grad = sparse_grad
                self.last_phi = phi