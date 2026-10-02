import torch.nn


class MLMHeadModule(torch.nn.Module):
    tokens_embedding_weight_name: str | None = None

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
