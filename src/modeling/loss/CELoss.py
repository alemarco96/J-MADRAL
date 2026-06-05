import torch.nn.functional


# Compute the CE (Cross Entropy) loss.
class CELoss:
    @staticmethod
    def forward(q_embedding: torch.Tensor, d_embedding: torch.Tensor) -> torch.Tensor:
        similarity = torch.matmul(q_embedding, d_embedding.transpose(0, 1))
        labels = torch.arange(q_embedding.shape[0], dtype=torch.int64, device=similarity.device)

        return torch.nn.functional.cross_entropy(similarity, labels)
