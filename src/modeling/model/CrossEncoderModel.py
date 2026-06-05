from modeling.config.CrossEncoderConfig import CrossEncoderConfig
from modeling.encoder.BertEncoder import BertEncoder
from modeling.encoder.ModernBertEncoder import ModernBertEncoder
from modeling.generic.ModelInput import ModelInput
from modeling.generic.CrossEncoderModelOutput import CrossEncoderModelOutput
from modeling.pooler.CrossAspectsPooler import CrossAspectsPooler
from modeling.pooler.CrossLinearPooler import CrossLinearPooler
import torch.nn
import torch.utils.checkpoint
import transformers
from typing import Callable


class CrossEncoderModel(transformers.PreTrainedModel):
    config_class = CrossEncoderConfig

    def __init__(self, config, *args, **kwargs):
        super(CrossEncoderModel, self).__init__(config)

        if config.encoder_module == "BERT":
            self.encoder = BertEncoder(config)
        elif config.encoder_module == "ModernBERT":
            self.encoder = ModernBertEncoder(config)
        else:
            raise ValueError(f"Invalid config.encoder_module: found {config.encoder_module}.")

        if config.pooler_module == "CrossAspects":
            self.pooler = CrossAspectsPooler(config)
        elif config.pooler_module == "CrossLinear":
            self.pooler = CrossLinearPooler(config)
        else:
            raise ValueError(f"Invalid config.pooler_module: found {config.pooler_module}.")

        self.sep_token_id = config.sep_token_id
        # assert isinstance(self.sep_token_id, int)

        # Initialize weights and apply final processing.
        self.supports_gradient_checkpointing = True
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
        self.post_init()

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
                return_dict: bool | None = None,
                *args, **kwargs)\
            -> CrossEncoderModelOutput | tuple[torch.FloatTensor | tuple[torch.FloatTensor, ...] | None, ...]:
        assert input_ids is not None
        assert isinstance(input_ids, torch.Tensor)
        assert attention_mask is not None
        assert isinstance(attention_mask, torch.Tensor)

        # Move the input tensor to the same device as the encoder model.
        input_ids = input_ids.to(self.encoder.encoder.device)
        attention_mask = attention_mask.to(self.encoder.encoder.device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.encoder.encoder.device)

        # Compute the query and document masks.
        if isinstance(self.pooler, CrossAspectsPooler):
            sep_index = torch.nonzero(input_ids == self.sep_token_id)[:, 1][::2]
            positions = torch.arange(input_ids.shape[1], dtype=torch.int64, device=self.encoder.encoder.device) \
                .unsqueeze(0).expand(input_ids.shape[0], -1)

            query_mask = ((positions > 0) & (positions <= sep_index.unsqueeze(1))).to(dtype=torch.int64) * attention_mask
            document_mask = (positions > sep_index.unsqueeze(1)).to(dtype=torch.int64) * attention_mask
            del sep_index, positions
        else:
            query_mask = None
            document_mask = None

        encoder_input = ModelInput(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   query_mask=query_mask,
                                   document_mask=document_mask,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)
        result = self.pooler(encoder_input, self.encoder(encoder_input))

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if return_dict:
            return result
        else:
            return result.as_tuple()
