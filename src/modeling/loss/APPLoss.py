import torch.nn.functional


# Compute APP (Aspect Presence Prediction) loss.
class APPLoss:
    @staticmethod
    def forward(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Append a 1 at the end for the aspect related to the [CLS] token, which is always present.
        presence_labels = (labels != -100).long()
        presence_labels = torch.cat([presence_labels,
                                     torch.ones((presence_labels.shape[0], 1),
                                                dtype=presence_labels.dtype,
                                                device=presence_labels.device)], dim=1)

        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits.contiguous(),
            presence_labels.contiguous().to(logits.dtype),
            reduction="mean")
