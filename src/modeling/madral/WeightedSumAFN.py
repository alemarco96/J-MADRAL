from modeling.generic.AFNModule import AFNModule
import torch.nn


class WeightedSumAFN(AFNModule):
    def __init__(self, embedding_size: int, num_aspects: int):
        super(WeightedSumAFN, self).__init__(embedding_size, num_aspects)

        # Learnable parameters for the importance weight of each aspect.
        self.afn_aspect_gamma = torch.nn.Parameter(torch.zeros([self.num_aspects], dtype=torch.float32))

    def forward(self, aspects_embedding: torch.Tensor) -> tuple[torch.Tensor, None, torch.Tensor]:
        # Compute the per-aspect weights and the pooled embedding.
        presence_weights = torch.softmax(self.afn_aspect_gamma, dim=0).unsqueeze(0).\
            expand(aspects_embedding.shape[0], -1)
        pooled_embedding = torch.bmm(presence_weights.unsqueeze(1), aspects_embedding).squeeze(1)

        return pooled_embedding, None, presence_weights
