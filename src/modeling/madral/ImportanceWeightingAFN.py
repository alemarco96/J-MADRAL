from modeling.generic.AFNModule import AFNModule
import torch.nn


class ImportanceWeightingAFN(AFNModule):
    def __init__(self, embedding_size: int, num_aspects: int):
        super(ImportanceWeightingAFN, self).__init__(embedding_size, num_aspects)

        # Linear layer predicting the importance of the i-th aspect. NOTE: do not use a single linear layer,
        # as each aspect embedding must influence only the corresponding importance logit.
        self.afn_importance_linear = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_size, 1, bias=True) for _ in range(self.num_aspects)])

    def forward(self, aspects_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute the per-aspect importance logits.
        importance_logits = torch.cat([self.afn_importance_linear[i](aspects_embedding[:, i, :])
                                       for i in range(self.num_aspects)], dim=1)

        # Use the importance logits to compute the per-aspect weights and the pooled embedding.
        importance_weights = torch.softmax(importance_logits, dim=1)
        pooled_embedding = torch.bmm(importance_weights.unsqueeze(1), aspects_embedding).squeeze(1)

        return pooled_embedding, importance_logits, importance_weights
