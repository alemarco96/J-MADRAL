from modeling.generic.EncoderModule import EncoderModule
from modeling.generic.ModelInput import ModelInput
from modeling.generic.ModelOutput import ModelOutput
import transformers
import torch.utils.checkpoint
from typing import Callable


class BertEncoder(EncoderModule):
    tokens_embedding_weight_name: str | None = "encoder.embeddings.word_embeddings.weight"

    def __init__(self, config, *args, **kwargs):
        super(BertEncoder, self).__init__()

        self.encoder = transformers.BertModel(config, add_pooling_layer=False)

        self.supports_gradient_checkpointing = True
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    def _set_gradient_checkpointing(self, enable: bool = True,
                                    gradient_checkpointing_func: Callable = torch.utils.checkpoint.checkpoint):
        self._gradient_checkpointing_func = gradient_checkpointing_func
        self.gradient_checkpointing = enable

    def forward(self, encoder_input: ModelInput) -> ModelOutput:
        # Do inference with the encoder model.
        if self.gradient_checkpointing and self.training:
            def _encoder_forward(input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
                return self.encoder(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    output_attentions=encoder_input.output_attentions,
                                    output_hidden_states=encoder_input.output_hidden_states)

            encoder_output = self._gradient_checkpointing_func(_encoder_forward,
                                                               encoder_input.input_ids,
                                                               encoder_input.attention_mask,
                                                               encoder_input.token_type_ids)
        else:
            encoder_output = self.encoder(input_ids=encoder_input.input_ids,
                                          token_type_ids=encoder_input.token_type_ids,
                                          attention_mask=encoder_input.attention_mask,
                                          output_attentions=encoder_input.output_attentions,
                                          output_hidden_states=encoder_input.output_hidden_states)

        # Return the encoder output.
        return ModelOutput.from_base_output(encoder_output)
