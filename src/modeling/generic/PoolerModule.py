from modeling.generic.ModelInput import ModelInput
from modeling.generic.ModelOutput import ModelOutput
import torch.nn


class PoolerModule(torch.nn.Module):
    def forward(self, encoder_input: ModelInput, encoder_output: ModelOutput) -> ModelOutput:
        raise NotImplementedError
