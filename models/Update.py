import torch
import torch.nn.functional as F

# Local model update at client
def local_update(model, dataloader, optimizer, epochs=1, device='cuda'):
    model.train()
    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
    return model.state_dict()