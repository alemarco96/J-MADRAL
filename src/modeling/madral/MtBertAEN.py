from modeling.generic.AENModule import AENModule
import torch.nn


class MtBertAEN(AENModule):
    def __init__(self, embedding_size: int, num_aspects: int):
        super(MtBertAEN, self).__init__(embedding_size, num_aspects)

        # Define the "Q" 2D tensor used in the eq. 1 at page 4 of the MADRAL paper.
        self.aen_query_embedding = torch.nn.Parameter(torch.empty([self.num_aspects, embedding_size]))
        torch.nn.init.normal_(self.aen_query_embedding, mean=0.0, std=0.02)

        # Define the attention used in eq. 2 at page 4 of the MADRAL paper. The Wq, Wk, Wv matrices applied to
        # the input q, k, v tensors are already included in this object.
        self.aen_attention = torch.nn.MultiheadAttention(embedding_size,
                                                         num_heads=1,
                                                         bias=False,
                                                         add_bias_kv=False,
                                                         batch_first=True)

        # Disable the output linear layer, by setting and freezing it equal to the identity.
        with torch.no_grad():
            self.aen_attention.out_proj.weight.copy_(torch.eye(embedding_size))
            self.aen_attention.out_proj.weight.requires_grad = False
            self.aen_attention.out_proj.bias = None

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Consider exclusively the [CLS] token.
        aen_tokens_embedding = last_hidden_state[:, 0, :].unsqueeze(1)

        # Copy the "Q" learnable parameter for every input in the current batch.
        batch_size = aen_tokens_embedding.shape[0]
        aen_query_embedding = self.aen_query_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        # Perform the AEN attention.
        aspects_embedding = self.aen_attention(aen_query_embedding,
                                               aen_tokens_embedding,
                                               aen_tokens_embedding,
                                               need_weights=False)[0]

        return aspects_embedding
