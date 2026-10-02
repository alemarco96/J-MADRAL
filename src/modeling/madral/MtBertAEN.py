from modeling.generic.AENModule import AENModule
import torch.nn


class MtBertAEN(AENModule):
    def __init__(self, embedding_size: int, num_aspects: int):
        super(MtBertAEN, self).__init__(embedding_size, num_aspects)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        # Consider exclusively the [CLS] token.
        aspects_embedding = last_hidden_state[:, 0, :].unsqueeze(1)
        aspects_embedding = aspects_embedding.expand(-1, self.num_aspects, -1)
        return aspects_embedding
