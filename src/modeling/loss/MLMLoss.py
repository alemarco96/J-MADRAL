import torch.nn.functional

# Compute MLM (Masked Language Modeling) loss.
class MLMLoss:
    @staticmethod
    def forward(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not torch.any(labels != -100):
            return torch.zeros([], dtype=torch.float32, device=labels.device)
        
        valid_idxs = labels != -100
        logits = logits[valid_idxs]
        labels = labels[valid_idxs]
        del valid_idxs
        
        return torch.nn.functional.cross_entropy(logits.contiguous().view(-1, logits.shape[-1]),
                                                 labels.contiguous().view(-1),
                                                 reduction="mean")
