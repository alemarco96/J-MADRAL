from modeling.generic.ModelInput import ModelInput
from modeling.generic.ModelOutput import ModelOutput
from modeling.generic.PoolerModule import PoolerModule
from modeling.madral.AspectsGatingAFN import AspectsGatingAFN
from modeling.madral.ClsGatingAFN import ClsGatingAFN
from modeling.madral.ImportanceWeightingAFN import ImportanceWeightingAFN
from modeling.madral.MadralAEN import MadralAEN
from modeling.madral.MtBertAEN import MtBertAEN
from modeling.madral.PresenceWeightingAFN import PresenceWeightingAFN
from modeling.madral.WeightedSumAFN import WeightedSumAFN
import torch.nn


class AspectsPooler(PoolerModule):
    def __init__(self, config):
        super(AspectsPooler, self).__init__()

        if config.aen_module == "MadralAEN":
            self.aen = MadralAEN(config.hidden_size, config.num_aspects)
        elif config.aen_module == "MtBertAEN":
            self.aen = MtBertAEN(config.hidden_size, config.num_aspects)
        else:
            raise ValueError(f"Invalid config.aen_module: found {config.aen_module}.")

        if config.afn_module == "AspectsGatingAFN":
            self.afn = AspectsGatingAFN(config.hidden_size, config.num_aspects + 1)
        elif config.afn_module == "ClsGatingAFN":
            self.afn = ClsGatingAFN(config.hidden_size, config.num_aspects + 1)
        elif config.afn_module == "ImportanceWeightingAFN":
            self.afn = ImportanceWeightingAFN(config.hidden_size, config.num_aspects + 1)
        elif config.afn_module == "PresenceWeightingAFN":
            self.afn = PresenceWeightingAFN(config.hidden_size, config.num_aspects + 1)
        elif config.afn_module == "WeightedSumAFN":
            self.afn = WeightedSumAFN(config.hidden_size, config.num_aspects + 1)
        else:
            raise ValueError(f"Invalid config.afn_module: found {config.afn_module}.")

    def forward(self, encoder_input: ModelInput, encoder_output: ModelOutput) -> ModelOutput:
        # Isolate the representation of the [CLS] tokens.
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]

        # Run the AEN and AFN modules of MADRAL. Append the [CLS] embedding as the last aspect.
        aspects_embedding = self.aen(encoder_output.last_hidden_state, encoder_input.attention_mask)
        aspects_embedding = torch.cat([aspects_embedding, cls_embedding.unsqueeze(1)], dim=1)
        pooled_embedding, presence_logits, presence_weights = self.afn(aspects_embedding)

        # # Normalize the pooled embedding.
        # pooled_embedding = torch.nn.functional.normalize(pooled_embedding, dim=1)

        if encoder_input.output_logits is None or not encoder_input.output_logits:
            presence_logits = None

        # Return the output data.
        # noinspection PyTypeChecker
        return ModelOutput(aspects_embedding=aspects_embedding,
                           aspects_logits=None,
                           attentions=encoder_output.attentions,
                           hidden_states=encoder_output.hidden_states,
                           last_hidden_state=encoder_output.last_hidden_state,
                           mlm_logits=None,
                           pooler_output=pooled_embedding,
                           presence_logits=presence_logits,
                           presence_weights=presence_weights)
