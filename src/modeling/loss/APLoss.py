import torch.nn.functional


# Compute AP (Aspect Prediction) loss.
class APLoss:
    @staticmethod
    def forward(logits: tuple[torch.Tensor, ...], labels: torch.Tensor) -> torch.Tensor:
        if (labels != -100).sum().item() <= 0:
            return torch.zeros([], dtype=torch.float32, device=labels.device)

        ap_loss = torch.stack([torch.nn.functional.cross_entropy(
            logits[i].contiguous().view(-1, logits[i].shape[-1]),
            labels[:, i].contiguous().view(-1),
            ignore_index=-100,
            reduction="none") for i in range(labels.shape[1])], dim=1)

        # Compute the mean, considering only the non-invalid labels.
        return ap_loss.sum() / (labels != -100).sum()
