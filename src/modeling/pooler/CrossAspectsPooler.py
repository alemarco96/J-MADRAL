from modeling.generic.CrossEncoderModelOutput import CrossEncoderModelOutput
from modeling.generic.ModelInput import ModelInput
from modeling.generic.ModelOutput import ModelOutput
from modeling.generic.PoolerModule import PoolerModule
from modeling.madral.CrossMadralAEN import CrossMadralAEN
import torch.nn


class CrossAspectsPooler(PoolerModule):
    def __init__(self, config):
        super(CrossAspectsPooler, self).__init__()

        if config.aen_module == "CrossMadralAEN":
            self.aen = CrossMadralAEN(config.hidden_size, config.num_aspects)
        else:
            raise ValueError(f"Invalid config.aen_module: found {config.aen_module}.")

        self.out_linear = torch.nn.Linear((2 * config.num_aspects + 1) * config.hidden_size, 2, bias=True)

    def forward(self, encoder_input: ModelInput, encoder_output: ModelOutput) -> CrossEncoderModelOutput:
        assert encoder_input.query_mask is not None
        assert encoder_input.document_mask is not None
        assert encoder_output.last_hidden_state is not None

        # Isolate the representation of the [CLS] tokens.
        cls_embedding = encoder_output.last_hidden_state[:, 0, :].unsqueeze(1)

        # Run the AEN module, to extract aspects embedding for both the query and the document.
        aspects_embedding = self.aen(encoder_output.last_hidden_state,
                                     torch.stack([encoder_input.query_mask,
                                                  encoder_input.document_mask], dim=0))
        q_aspects_embedding = aspects_embedding[0]
        d_aspects_embedding = aspects_embedding[1]
        del aspects_embedding

        # Obtain relevance prediction projecting the [CLS] token embeddings and both query and document
        # aspects embedding to a classification layer.
        input_embeddings = torch.cat([cls_embedding, q_aspects_embedding, d_aspects_embedding],
                                     dim=1).view(cls_embedding.shape[0], -1)
        relevance_logits = self.out_linear(input_embeddings)

        # Return the output data.
        return CrossEncoderModelOutput(attentions=encoder_output.attentions,
                                       d_aspects_embedding=d_aspects_embedding,
                                       d_aspects_logits=None,
                                       hidden_states=encoder_output.hidden_states,
                                       last_hidden_state=encoder_output.last_hidden_state,
                                       mlm_logits=None,
                                       q_aspects_embedding=q_aspects_embedding,
                                       q_aspects_logits=None,
                                       relevance_logits=relevance_logits)

        # # noinspection PyTypeChecker
        # return ModelOutput(aspects_embedding=torch.stack([q_aspects_embedding, d_aspects_embedding], dim=0),
        #                    aspects_logits=None,
        #                    attentions=encoder_output.attentions,
        #                    hidden_states=encoder_output.hidden_states,
        #                    last_hidden_state=encoder_output.last_hidden_state,
        #                    mlm_logits=None,
        #                    pooler_output=None,
        #                    presence_logits=relevance_logits,
        #                    presence_weights=None)
