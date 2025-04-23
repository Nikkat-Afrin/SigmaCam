import torch

class SigmaCamModel:
    """
    Wrap a PyTorch model for boundary computation.
    """
    def __init__(self, model, input_shape, T, centroid, device='cpu'):
        self.model = model.to(device)
        self.input_shape = input_shape
        self.T = T.to(device)
        self.centroid = centroid.to(device)
        self.device = device
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return raw outputs or probabilities."""
        self.model.eval()
        with torch.no_grad():
            return self.model(x.to(self.device))


def wrapper(model, input_shape, T, centroid, device='cpu'):
    """Factory to create SigmaCamModel instance."""
    return SigmaCamModel(model, input_shape, T, centroid, device)
