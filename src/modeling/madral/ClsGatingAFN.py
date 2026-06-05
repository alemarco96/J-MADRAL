from modeling.generic.AFNModule import AFNModule
import torch.nn


class ClsGatingAFN(AFNModule):
    def __init__(self, embedding_size: int, num_aspects: int):
        super(ClsGatingAFN, self).__init__(embedding_size, num_aspects)

        # Linear layer predicting from the CLS token the importance of the i-th aspect.
        self.afn_importance_linear = torch.nn.Linear(embedding_size, num_aspects, bias=True)

    def forward(self, aspects_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute the per-aspect importance logits.
        importance_logits = self.afn_importance_linear(aspects_embedding[:, -1, :])

        # Use the presence logits to compute the per-aspect weights and the pooled embedding.
        importance_weights = torch.softmax(importance_logits, dim=1)
        pooled_embedding = torch.bmm(importance_weights.unsqueeze(1), aspects_embedding).squeeze(1)

        return pooled_embedding, importance_logits, importance_weights
