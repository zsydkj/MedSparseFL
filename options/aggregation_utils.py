import torch

class Aggregator:
    """ Dynamic weighted aggregation with anomaly suppression """
    def __init__(self, model_size, anomaly_threshold=3.0):
        self.model_size = model_size
        self.anomaly_threshold = anomaly_threshold

    def aggregate(self, client_grads, client_weights=None):
        """
        client_grads: list of gradient vectors from clients
        client_weights: optional list of dynamic weights
        """
        grads = torch.stack(client_grads, dim=0)  # [num_clients, num_params]
        if client_weights is None:
            client_weights = torch.ones(len(client_grads), device=grads.device)

        # detect anomalous clients using z-score
        mean_grad = grads.mean(dim=0)
        std_grad = grads.std(dim=0) + 1e-8
        z_scores = ((grads - mean_grad) / std_grad).abs()
        anomaly_mask = (z_scores > self.anomaly_threshold).any(dim=1).float()
        weights = client_weights * (1 - anomaly_mask)  # zero weight for anomalies

        # normalize weights
        weights = weights / (weights.sum() + 1e-8)
        aggregated_grad = (grads.T @ weights).T
        return aggregated_grad