import torch.nn.functional


# Compute CE (Cross Entropy) loss with soft probabilities.
class CESoftLoss:
    @staticmethod
    def forward(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(dtype=torch.float32)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
