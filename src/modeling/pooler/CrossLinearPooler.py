from modeling.generic.CrossEncoderModelOutput import CrossEncoderModelOutput
from modeling.generic.ModelInput import ModelInput
from modeling.generic.ModelOutput import ModelOutput
from modeling.generic.PoolerModule import PoolerModule
import torch.nn


class CrossLinearPooler(PoolerModule):
    def __init__(self, config):
        super().__init__()

        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.activation = torch.nn.Tanh()
        self.out_linear = torch.nn.Linear(config.hidden_size, 1, bias=True)

    def forward(self, encoder_input: ModelInput, encoder_output: ModelOutput) -> CrossEncoderModelOutput:
        # Isolate the representation of the [CLS] tokens.
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]

        # Process the cls embedding through the linear pooler.
        pooled_embedding = self.activation(self.linear(cls_embedding))

        # Obtain relevance prediction projecting the [CLS] token embeddings to a classification layer.
        relevance_logits = self.out_linear(pooled_embedding).view(-1)

        # Return the output data.
        # noinspection PyTypeChecker
        # return ModelOutput(aspects_embedding=None,
        #                    aspects_logits=None,
        #                    attentions=encoder_output.attentions,
        #                    hidden_states=encoder_output.hidden_states,
        #                    last_hidden_state=encoder_output.last_hidden_state,
        #                    mlm_logits=None,
        #                    pooler_output=pooled_embedding,
        #                    presence_logits=None,
        #                    presence_weights=None)

        # Return the output data.
        return CrossEncoderModelOutput(attentions=encoder_output.attentions,
                                       d_aspects_embedding=None,
                                       d_aspects_logits=None,
                                       hidden_states=encoder_output.hidden_states,
                                       last_hidden_state=encoder_output.last_hidden_state,
                                       mlm_logits=None,
                                       q_aspects_embedding=None,
                                       q_aspects_logits=None,
                                       relevance_logits=relevance_logits)
