from modeling.generic.ModelInput import ModelInput
from modeling.generic.ModelOutput import ModelOutput
from modeling.generic.PoolerModule import PoolerModule
import torch.nn


class LinearPooler(PoolerModule):
    def __init__(self, config):
        super().__init__()

        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.activation = torch.nn.Tanh()

    def forward(self, encoder_input: ModelInput, encoder_output: ModelOutput) -> ModelOutput:
        # Isolate the representation of the [CLS] tokens.
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]

        # Process the cls embedding through the linear pooler.
        pooled_embedding = self.activation(self.linear(cls_embedding))

        # # Normalize the pooled embedding.
        # pooled_embedding = torch.nn.functional.normalize(pooled_embedding, dim=1)

        # Return the output data.
        # noinspection PyTypeChecker
        return ModelOutput(aspects_embedding=None,
                           aspects_logits=None,
                           attentions=encoder_output.attentions,
                           hidden_states=encoder_output.hidden_states,
                           last_hidden_state=encoder_output.last_hidden_state,
                           mlm_logits=None,
                           pooler_output=pooled_embedding,
                           presence_logits=None,
                           presence_weights=None)
