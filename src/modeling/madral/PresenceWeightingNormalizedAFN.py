from modeling.generic.AFNModule import AFNModule
import torch.nn


class PresenceWeightingNormalizedAFN(AFNModule):
    def __init__(self, embedding_size: int, num_aspects: int):
        super(PresenceWeightingNormalizedAFN, self).__init__(embedding_size, num_aspects)

        # Linear layer predicting if the i-th aspect is present or not. NOTE: do not use a single linear layer,
        # as each aspect embedding must influence only the corresponding presence logit.
        self.afn_presence_linear = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_size, 1, bias=True) for _ in range(self.num_aspects)])

        # Learnable parameters for the importance weight of each aspect.
        self.afn_aspect_gamma = torch.nn.Parameter(torch.ones([self.num_aspects], dtype=torch.float32))


    def forward(self, aspects_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute the per-aspect presence logits.
        presence_logits = torch.cat([self.afn_presence_linear[i](aspects_embedding[:, i, :])
                                     for i in range(self.num_aspects)], dim=1)

        # Use the presence logits to compute the per-aspect weights and the pooled embedding.
        presence_weights = torch.sigmoid(presence_logits) * self.afn_aspect_gamma.unsqueeze(0)
        presence_weights = torch.nn.functional.normalize(presence_weights, p=1, dim=1)
        pooled_embedding = torch.bmm(presence_weights.unsqueeze(1), aspects_embedding).squeeze(1)

        return pooled_embedding, presence_logits, presence_weights
