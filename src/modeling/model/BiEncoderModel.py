from modeling.config.BiEncoderConfig import BiEncoderConfig
from modeling.encoder.BertEncoder import BertEncoder
from modeling.encoder.ModernBertEncoder import ModernBertEncoder
from modeling.generic.ModelInput import ModelInput
from modeling.generic.ModelOutput import ModelOutput
from modeling.madral.MadralAEN import MadralAEN
from modeling.pooler.AspectsPooler import AspectsPooler
from modeling.pooler.LinearPooler import LinearPooler
import torch.nn
import torch.utils.checkpoint
import transformers
from typing import Callable


class BiEncoderModel(transformers.PreTrainedModel):
    config_class = BiEncoderConfig

    def __init__(self, config, *args, **kwargs):
        super(BiEncoderModel, self).__init__(config)

        if config.encoder_module == "BERT":
            self.encoder = BertEncoder(config)
        elif config.encoder_module == "ModernBERT":
            self.encoder = ModernBertEncoder(config)
        else:
            raise ValueError(f"Invalid config.encoder_module: found {config.encoder_module}.")

        if config.pooler_module == "Linear":
            self.pooler = LinearPooler(config)
        elif config.pooler_module == "Aspects":
            self.pooler = AspectsPooler(config)
        else:
            raise ValueError(f"Invalid config.pooler_module: found {config.pooler_module}.")

        # Initialize weights and apply final processing.
        # if isinstance(self.pooler, AspectsPooler) and isinstance(self.pooler.aen, MadralAEN):
        #     self.pooler.aen.aen_attention.out_proj.weight.requires_grad = False

        self.supports_gradient_checkpointing = True
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

        self.post_init()

    def post_init(self):
        super().post_init()
        # if isinstance(self.pooler, AspectsPooler) and isinstance(self.pooler.aen, MadralAEN):
        #     self.pooler.aen.aen_attention.out_proj.weight.requires_grad = False

    def _set_gradient_checkpointing(self, enable: bool = True,
                                    gradient_checkpointing_func: Callable = torch.utils.checkpoint.checkpoint):
        self._gradient_checkpointing_func = gradient_checkpointing_func
        self.gradient_checkpointing = enable
        super()._set_gradient_checkpointing(enable, gradient_checkpointing_func)

    def forward(self,
                input_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                token_type_ids: torch.Tensor | None = None,
                output_attentions: bool | None = None,
                output_hidden_states: bool | None = None,
                output_logits: bool | None = None,
                return_dict: bool | None = None,
                *args, **kwargs)\
            -> ModelOutput | tuple[torch.FloatTensor | tuple[torch.FloatTensor, ...] | None, ...]:
        encoder_input = ModelInput(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   output_logits=output_logits)

        result = self.pooler(encoder_input, self.encoder(encoder_input))

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if return_dict:
            return result
        else:
            return result.as_tuple()
