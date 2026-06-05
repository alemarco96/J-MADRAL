from modeling.generic.AENModule import AENModule
import torch.nn


class CrossMadralAEN(AENModule):
    def __init__(self, embedding_size: int, num_aspects: int):
        super(CrossMadralAEN, self).__init__(embedding_size, num_aspects)

        # Define the "Q" 2D tensor used in the eq. 1 at page 4 of the MADRAL paper.
        self.query_embedding = torch.nn.Parameter(torch.empty([self.num_aspects, embedding_size]))
        torch.nn.init.normal_(self.query_embedding, mean=0.0, std=0.02)

        # Define the attention used in eq. 2 at page 4 of the MADRAL paper. The Wq, Wk, Wv matrices applied to
        # the input q, k, v tensors are already included in this object.
        self.attention = torch.nn.MultiheadAttention(embedding_size,
                                                     num_heads=1,
                                                     bias=False,
                                                     add_bias_kv=False,
                                                     batch_first=True)

        # # Disable the output linear layer, by setting and freezing it equal to the identity.
        # with torch.no_grad():
        #     torch.nn.init.eye_(self.attention.out_proj.weight)
        # self.attention.out_proj.weight.requires_grad = False
        # self.attention.out_proj.bias = None

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        # Obtain the masks used to separate tokens belonging to query and documents.
        q_attention_mask = attention_mask[0]
        d_attention_mask = attention_mask[1]

        # Copy the "Q" learnable parameter for every input in the current batch.
        batch_size = last_hidden_state.shape[0]
        aen_query_embedding = self.query_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        # Perform the AEN attention.
        q_aspects_embedding = self.attention(aen_query_embedding,
                                             last_hidden_state,
                                             last_hidden_state,
                                             key_padding_mask=q_attention_mask == 0,
                                             need_weights=False)[0]
        d_aspects_embedding = self.attention(aen_query_embedding,
                                             last_hidden_state,
                                             last_hidden_state,
                                             key_padding_mask=d_attention_mask == 0,
                                             need_weights=False)[0]
        del q_attention_mask, d_attention_mask

        return torch.stack([q_aspects_embedding, d_aspects_embedding], dim=0)
