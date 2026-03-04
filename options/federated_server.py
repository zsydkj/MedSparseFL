import torch
from options.aggregation_utils import Aggregator
from options.privacy_utils import HomomorphicEncryption

class FederatedServer:
    def __init__(self, clients, model, args):
        self.clients = clients
        self.model = model
        self.args = args
        self.aggregator = Aggregator(sum(p.numel() for p in self.model.parameters()))

    def broadcast_model(self):
        """Send current global model to all clients"""
        global_state = [p.data.clone() for p in self.model.parameters()]
        for client in self.clients:
            for p_client, p_global in zip(client.model.parameters(), global_state):
                p_client.data.copy_(p_global)

    def aggregate(self):
        """Collect encrypted gradients, decrypt, aggregate, and update model"""
        client_grads = []
        client_weights = []
        for client in self.clients:
            grad = HomomorphicEncryption.decrypt(client.last_grad)
            client_grads.append(grad)
            client_weights.append(client.last_phi.mean())

        # Use aggregator to combine client updates
        aggregated_grad = self.aggregator.aggregate(client_grads, client_weights)

        # Update global model
        pointer = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data -= self.args.lr * aggregated_grad[pointer:pointer + numel].view_as(p)
            pointer += numel