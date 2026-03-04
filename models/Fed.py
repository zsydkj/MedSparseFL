import torch
import copy
from models.Nets import ResNet18, GradientScoreNetwork

# --------------------------
# Client class for federated learning
# --------------------------
class Client:
    def __init__(self, client_id, model, dataloader, device='cuda', lr=0.01):
        self.id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.dataloader = dataloader
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.residual = None  # residual feedback for sparse gradients

    def compute_sparse_update(self, gsn, support_tracker, mask_ratio=0.1, threshold=0.5):
        """
        Compute sparse gradient update using GSN, support consistency, and residual feedback
        """
        self.model.train()
        # Accumulate gradients over one epoch
        for x, y in self.dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = torch.nn.CrossEntropyLoss()(output, y)
            loss.backward()

            # Flatten gradients into a single vector
            grad_vector = torch.cat([p.grad.view(-1) for p in self.model.parameters()])

            # Add residual from previous round
            if self.residual is not None:
                grad_vector += self.residual

            # Compute GSN scores
            scores = gsn(grad_vector)
            mask = (scores > threshold).float()

            # Apply support-aware mask
            sparse_grad = grad_vector * mask

            # Update residual for next round
            self.residual = grad_vector - sparse_grad

            # Update support consistency
            support, phi = support_tracker.update_support(sparse_grad)

            # Store for server aggregation
            self.last_sparse_grad = sparse_grad
            self.last_phi = phi

        return self.last_sparse_grad, self.last_phi

# --------------------------
# Server class for federated aggregation
# --------------------------
class Server:
    def __init__(self, global_model, lr=0.01):
        self.global_model = global_model
        self.lr = lr

    def aggregate_updates(self, client_updates, client_phis, anomaly_threshold=3.0):
        """
        Aggregate sparse updates from clients using dynamic weighting and anomaly suppression
        """
        grads = torch.stack(client_updates, dim=0)  # [num_clients, num_params]
        phis = torch.stack(client_phis, dim=0)      # [num_clients, num_params]

        # z-score anomaly detection
        mean_grad = grads.mean(dim=0)
        std_grad = grads.std(dim=0) + 1e-8
        z_scores = ((grads - mean_grad) / std_grad).abs()
        anomaly_mask = (z_scores > anomaly_threshold).any(dim=1).float()  # 1 for anomalous client

        # compute dynamic weights using support consistency
        client_weights = phis.mean(dim=1) * (1 - anomaly_mask)

        # normalize weights
        client_weights = client_weights / (client_weights.sum() + 1e-8)

        # weighted aggregation
        aggregated_grad = (grads.T @ client_weights).T

        # update global model
        pointer = 0
        for p in self.global_model.parameters():
            numel = p.numel()
            p.data -= self.lr * aggregated_grad[pointer:pointer + numel].view_as(p)
            pointer += numel

    def train_round(self, clients, gsn, support_trackers):
        """
        Perform one round of federated learning
        """
        client_updates, client_phis = [], []
        for c, tracker in zip(clients, support_trackers):
            grad, phi = c.compute_sparse_update(gsn, tracker)
            client_updates.append(grad)
            client_phis.append(phi)
        self.aggregate_updates(client_updates, client_phis)