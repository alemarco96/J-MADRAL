import torch.nn.functional

# Compute MLM (Masked Language Modeling) loss.
class MLMLoss:
    @staticmethod
    def forward(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits.contiguous().view(-1, logits.shape[-1]),
                                                 labels.contiguous().view(-1),
                                                 reduction="mean")
