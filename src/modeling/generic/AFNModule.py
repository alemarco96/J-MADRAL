import torch.nn


class AFNModule(torch.nn.Module):
    def __init__(self, embedding_size: int, num_aspects: int):
        assert embedding_size is not None and isinstance(embedding_size, int) and embedding_size > 0
        assert num_aspects is not None and isinstance(num_aspects, int) and num_aspects > 0
        super(AFNModule, self).__init__()

        self.embedding_size = embedding_size
        self.num_aspects = num_aspects

    def forward(self, aspects_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        raise NotImplementedError
