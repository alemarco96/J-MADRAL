from modeling.generic.ModelInput import ModelInput
from modeling.generic.ModelOutput import ModelOutput
import torch.nn


class EncoderModule(torch.nn.Module):
    tokens_embedding_weight_name: str | None = None

    def forward(self, encoder_input: ModelInput) -> ModelOutput:
        raise NotImplementedError
