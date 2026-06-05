from modeling.config.CrossEncoderConfig import CrossEncoderConfig
from modeling.generic.CrossEncoderModelOutput import CrossEncoderModelOutput
from modeling.mlm_head.BertMLMHead import BertMLMHead
from modeling.mlm_head.ModernBertMLMHead import ModernBertMLMHead
from modeling.model.CrossEncoderModel import CrossEncoderModel
import torch.nn
import torch.utils.checkpoint
import transformers
from typing import Callable


class TrainCrossEncoderModel(transformers.PreTrainedModel):
    config_class = CrossEncoderConfig

    def __init__(self, config):
        super(TrainCrossEncoderModel, self).__init__(config)

        self.model = CrossEncoderModel(config)

        # Define the MLM head required to compute the MLM loss.
        if config.encoder_module == "BERT":
            self.mlm_head = BertMLMHead(config)
        elif config.encoder_module == "ModernBERT":
            self.mlm_head = ModernBertMLMHead(config)
        else:
            raise ValueError(f"Invalid config.encoder_module: found {config.encoder_module}.")

        # Define the linear layers required to classify each aspect into the correct class.
        if config.aspects_size is not None:
            self.aspects_linear = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size, aspect_out_size, bias=True)
                for aspect_out_size in config.aspects_size])
        else:
            self.aspects_linear = None

        self.supports_gradient_checkpointing = True
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
        super().post_init()

    def _set_gradient_checkpointing(self, enable: bool = True,
                                    gradient_checkpointing_func: Callable = torch.utils.checkpoint.checkpoint):
        self._gradient_checkpointing_func = gradient_checkpointing_func
        self.gradient_checkpointing = enable
        super()._set_gradient_checkpointing(enable, gradient_checkpointing_func)

    def untie_all(self):
        # Obtain the state dictionary of the entire model.
        sd = self.state_dict()

        # Find all tied weights in the entire model.
        sd_map = {}
        for k1, v1 in sd.items():
            assert isinstance(k1, str)
            if v1 is not None:
                assert isinstance(v1, torch.Tensor)
                v1_location = v1.data_ptr()

                inner = sd_map.get(v1_location, [])
                inner.append(k1)
                sd_map[v1_location] = inner
                del inner, v1_location
        try:
            del k1, v1
        except UnboundLocalError:
            pass
        sd_map = {k: v for k, v in sd_map.items() if len(v) > 1}

        # Continue further processing only if tied weights are found.
        if len(sd_map) > 0:
            # Untie all tied weights, by cloning the tensor associated with the first tied key found.
            sd_updates = [{k2: sd[k1[0]].clone() for k2 in k1[1:]} for k1 in sd_map.values()]
            sd |= {k2: v2 for v1 in sd_updates for k2, v2 in v1.items()}
            del sd_map, sd_updates

            # Apply the modified state dictionary to the entire model.
            self.load_state_dict(sd, strict=True)

            self.untie_tokens_embedding_weights()


    def tie_tokens_embedding_weights(self):
        if self.config.encoder_module == "BERT":
            self.mlm_head.mlm_head.predictions.decoder.weight = \
                self.model.encoder.encoder.embeddings.word_embeddings.weight

            self.mlm_head.mlm_head.predictions.decoder.bias = self.mlm_head.mlm_head.predictions.bias
        elif self.config.encoder_module == "ModernBERT":
            self.mlm_head.decoder.weight = self.model.encoder.encoder.embeddings.tok_embeddings.weight
        else:
            raise ValueError(f"Invalid config.encoder_module: found {self.config.encoder_module}.")

    def untie_tokens_embedding_weights(self):
        if self.config.encoder_module == "BERT":
            if self.mlm_head.mlm_head.predictions.decoder.weight is \
                    self.model.encoder.encoder.embeddings.word_embeddings.weight:
                self.mlm_head.mlm_head.predictions.decoder.weight = \
                    torch.nn.Parameter(self.model.encoder.encoder.embeddings.word_embeddings.weight.clone())

            if self.mlm_head.mlm_head.predictions.decoder.bias is self.mlm_head.mlm_head.predictions.bias:
                self.mlm_head.mlm_head.predictions.decoder.bias = \
                    torch.nn.Parameter(self.mlm_head.mlm_head.predictions.bias.clone())
        elif self.config.encoder_module == "ModernBERT":
            if self.mlm_head.decoder.weight is self.model.encoder.encoder.embeddings.tok_embeddings.weight:
                self.mlm_head.decoder.weight = \
                    torch.nn.Parameter(self.model.encoder.encoder.embeddings.tok_embeddings.weight.clone())
        else:
            raise ValueError(f"Invalid config.encoder_module: found {self.config.encoder_module}.")

    @torch.no_grad()
    def encode(self,
               input_ids: torch.Tensor | None = None,
               attention_mask: torch.Tensor | None = None,
               token_type_ids: torch.Tensor | None = None,
               **kwargs) -> torch.Tensor:
        # noinspection PyTypeChecker
        return self.forward(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            query_mask=kwargs.get("query_mask", None),
                            document_mask=kwargs.get("document_mask", None),
                            output_attentions=False,
                            output_hidden_states=False,
                            return_dict=True,
                            **kwargs).relevance_logits

    def forward(self,
                input_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                token_type_ids: torch.Tensor | None = None,
                output_attentions: bool | None = None,
                output_hidden_states: bool | None = None,
                return_dict: bool | None = None,
                **kwargs) -> CrossEncoderModelOutput | tuple[torch.FloatTensor | tuple[torch.FloatTensor, ...] | None, ...]:
        if input_ids is None or attention_mask is None:
            raise ValueError("The input_ids and attention_mask must not be null.")

        # Verify that the shape of the input parameters matches.
        assert input_ids.shape == attention_mask.shape
        if token_type_ids is not None:
            assert input_ids.shape == token_type_ids.shape

        # Perform inference using the underlying model.
        output = self.model.forward(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=True)
        assert isinstance(output, CrossEncoderModelOutput)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if return_dict:
            return output
        else:
            return output.to_tuple()

    def train(self, mode: bool = True):
        super().train(mode)

        self.model.train(mode)
        self.mlm_head.train(mode)

        if self.aspects_linear is not None:
            self.aspects_linear.train(mode)

    def eval(self):
        super().eval()

        self.model.eval()
        self.mlm_head.eval()

        if self.aspects_linear is not None:
            self.aspects_linear.eval()
