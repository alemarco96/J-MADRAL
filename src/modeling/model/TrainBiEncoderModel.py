from modeling.config.TrainBiEncoderConfig import TrainBiEncoderConfig
from modeling.generic.ModelOutput import ModelOutput
from modeling.generic.TrainDoubleOutput import TrainOutput
from modeling.mlm_head.BertMLMHead import BertMLMHead
from modeling.mlm_head.ModernBertMLMHead import ModernBertMLMHead
from modeling.model.BiEncoderModel import BiEncoderModel
import torch.nn
import torch.utils.checkpoint
import transformers
from typing import Callable


class TrainBiEncoderModel(transformers.PreTrainedModel):
    config_class = TrainBiEncoderConfig

    def __init__(self, config):
        super(TrainBiEncoderModel, self).__init__(config)

        if config.q_config.hidden_size != config.d_config.hidden_size:
            raise ValueError(f"Invalid embedding size: found {config.q_config.hidden_size} "
                             f"and {config.d_config.hidden_size}.")

        self.q_model = BiEncoderModel(config.q_config)
        self.d_model = BiEncoderModel(config.d_config)

        # Define the MLM head required to compute the MLM loss.
        if config.d_config.encoder_module == "BERT":
            self.d_mlm_head = BertMLMHead(config.d_config)
        elif config.d_config.encoder_module == "ModernBERT":
            self.d_mlm_head = ModernBertMLMHead(config.d_config)
        else:
            raise ValueError(f"Invalid config.d_config.encoder_module: found {config.d_config.encoder_module}.")

        # Define the linear layers required to classify each aspect into the correct class.
        if config.d_config.aspects_size is not None:
            self.aspects_linear = torch.nn.ModuleList(
                [torch.nn.Linear(config.d_config.hidden_size, aspect_out_size, bias=True)
                 for aspect_out_size in config.d_config.aspects_size])
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

            self.untie_document_tokens_embedding_weights()

    def clone_qd_models(self):
        self.q_model = type(self.d_model)(self.d_model.config)
        self.q_model.load_state_dict({k: v.clone() for k, v in self.d_model.state_dict().items()})

    def tie_qd_models(self):
        self.q_model = type(self.d_model)(self.d_model.config)
        self.q_model.load_state_dict(self.d_model.state_dict())

    def clone_qd_encoders(self):
        self.q_model.encoder = type(self.d_model.encoder)(self.d_model.encoder.config)
        self.q_model.encoder.load_state_dict({k: v.clone() for k, v in self.d_model.encoder.state_dict().items()})

    def tie_qd_encoders(self):
        self.q_model.encoder = type(self.d_model.encoder)(self.d_model.encoder.config)
        self.q_model.encoder.load_state_dict(self.d_model.encoder.state_dict())

    def clone_qd_tokens_embeddings(self):
        if type(self.q_model.encoder) != type(self.d_model.encoder):
            raise ValueError(f"Mismatch on query and document encoder: found instances of "
                             f"{type(self.q_model.encoder)} and {type(self.d_model.encoder)}.")

        weight = self.d_model.encoder.state_dict()[type(self.d_model.encoder).tokens_embedding_weight_name].clone()
        sd = self.q_model.encoder.state_dict() | {type(self.d_model.encoder).tokens_embedding_weight_name: weight}
        self.q_model.encoder.load_state_dict(sd, strict=True)

    def tie_qd_tokens_embeddings(self):
        if type(self.q_model.encoder) != type(self.d_model.encoder):
            raise ValueError(f"Mismatch on query and document encoder: found instances of "
                             f"{type(self.q_model.encoder)} and {type(self.d_model.encoder)}.")

        weight = self.d_model.encoder.state_dict()[type(self.d_model.encoder).tokens_embedding_weight_name]
        sd = self.q_model.encoder.state_dict() | {type(self.d_model.encoder).tokens_embedding_weight_name: weight}
        self.q_model.encoder.load_state_dict(sd, strict=True)

    def tie_mlm_head(self):
        weight = self.d_model.encoder.state_dict()[type(self.d_model.encoder).tokens_embedding_weight_name]
        sd = self.d_mlm_head.state_dict() | {type(self.d_mlm_head).tokens_embedding_weight_name: weight}
        self.d_mlm_head.load_state_dict(sd, strict=True)


    def tie_document_tokens_embedding_weights(self):
        if self.d_model.config.encoder_module == "BERT":
            self.d_mlm_head.mlm_head.predictions.decoder.weight = \
                self.d_model.encoder.encoder.embeddings.word_embeddings.weight

            self.d_mlm_head.mlm_head.predictions.decoder.bias = self.d_mlm_head.mlm_head.predictions.bias
        elif self.d_model.config.encoder_module == "ModernBERT":
            self.d_mlm_head.decoder.weight = self.d_model.encoder.encoder.embeddings.tok_embeddings.weight
        else:
            raise ValueError(f"Invalid config.d_config.encoder_module: found {self.d_model.config.encoder_module}.")

    def untie_document_tokens_embedding_weights(self):
        if self.d_model.config.encoder_module == "BERT":
            if self.d_mlm_head.mlm_head.predictions.decoder.weight is \
                    self.d_model.encoder.encoder.embeddings.word_embeddings.weight:
                self.d_mlm_head.mlm_head.predictions.decoder.weight = \
                    torch.nn.Parameter(self.d_model.encoder.encoder.embeddings.word_embeddings.weight.clone())

            if self.d_mlm_head.mlm_head.predictions.decoder.bias is self.d_mlm_head.mlm_head.predictions.bias:
                self.d_mlm_head.mlm_head.predictions.decoder.bias = \
                    torch.nn.Parameter(self.d_mlm_head.mlm_head.predictions.bias.clone())
        elif self.d_model.config.encoder_module == "ModernBERT":
            if self.d_mlm_head.decoder.weight is self.d_model.encoder.encoder.embeddings.tok_embeddings.weight:
                self.d_mlm_head.decoder.weight = \
                    torch.nn.Parameter(self.d_model.encoder.encoder.embeddings.tok_embeddings.weight.clone())
        else:
            raise ValueError(f"Invalid config.d_config.encoder_module: found {self.d_model.config.encoder_module}.")

    def tie_query_and_document_tokens_embedding_weights(self):
        if self.q_model.config.encoder_module == "BERT" and self.d_model.config.encoder_module == "BERT":
            self.q_model.encoder.encoder.embeddings.word_embeddings.weight = \
                self.d_model.encoder.encoder.embeddings.word_embeddings.weight
        elif self.q_model.config.encoder_module == "ModernBERT" and self.d_model.config.encoder_module == "ModernBERT":
            self.q_model.encoder.encoder.embeddings.tok_embeddings.weight = \
                self.d_model.encoder.encoder.embeddings.tok_embeddings.weight
        else:
            raise ValueError(f"Invalid config.q_config.encoder_module and/or config.d_config.encoder_module: found "
                             f"{self.q_model.config.encoder_module} and "
                             f"{self.d_model.config.encoder_module}.")

    def untie_query_and_document_tokens_embedding_weights(self):
        if self.q_model.config.encoder_module == "BERT" and self.d_model.config.encoder_module == "BERT":
            if self.q_model.encoder.encoder.embeddings.word_embeddings.weight is \
                    self.d_model.encoder.encoder.embeddings.word_embeddings.weight:
                self.q_model.encoder.encoder.embeddings.word_embeddings.weight = \
                    torch.nn.Parameter(self.d_model.encoder.encoder.embeddings.word_embeddings.weight.clone())
        elif self.q_model.config.encoder_module == "ModernBERT" and self.d_model.config.encoder_module == "ModernBERT":
            if self.q_model.encoder.encoder.embeddings.tok_embeddings.weight is \
                    self.d_model.encoder.encoder.embeddings.tok_embeddings.weight:
                self.q_model.encoder.encoder.embeddings.tok_embeddings.weight = \
                    torch.nn.Parameter(self.d_model.encoder.encoder.embeddings.tok_embeddings.weight.clone())
        else:
            raise ValueError(f"Invalid config.q_config.encoder_module and/or config.d_config.encoder_module: found "
                             f"{self.q_model.config.encoder_module} and "
                             f"{self.d_model.config.encoder_module}.")

    @torch.no_grad()
    def encode_query(self,
                     input_ids: torch.Tensor | None = None,
                     attention_mask: torch.Tensor | None = None,
                     token_type_ids: torch.Tensor | None = None,
                     **kwargs) -> torch.Tensor:
        # noinspection PyTypeChecker
        return self.forward(q_input_ids=input_ids,
                            q_attention_mask=attention_mask,
                            q_token_type_ids=token_type_ids,
                            d_input_ids=None,
                            d_attention_mask=None,
                            d_token_type_ids=None,
                            return_dict=True,
                            **kwargs).q_output.pooler_output

    @torch.no_grad()
    def encode_document(self,
                        input_ids: torch.Tensor | None = None,
                        attention_mask: torch.Tensor | None = None,
                        token_type_ids: torch.Tensor | None = None,
                        **kwargs) -> torch.Tensor:
        # noinspection PyTypeChecker
        return self.forward(q_input_ids=None,
                            q_attention_mask=None,
                            q_token_type_ids=None,
                            d_input_ids=input_ids,
                            d_attention_mask=attention_mask,
                            d_token_type_ids=token_type_ids,
                            return_dict=True,
                            **kwargs).d_output.pooler_output

    def forward(self,
                q_input_ids: torch.Tensor | None = None,
                q_attention_mask: torch.Tensor | None = None,
                q_token_type_ids: torch.Tensor | None = None,
                d_input_ids: torch.Tensor | None = None,
                d_attention_mask: torch.Tensor | None = None,
                d_token_type_ids: torch.Tensor | None = None,
                output_attentions: bool | None = None,
                output_hidden_states: bool | None = None,
                output_logits: bool | None = None,
                return_dict: bool | None = None,
                **kwargs) -> TrainOutput | tuple[ModelOutput, ModelOutput]:
        if q_input_ids is not None and q_attention_mask is not None:
            # Verify that the shape of the input parameters matches.
            assert q_input_ids.shape == q_attention_mask.shape
            if q_token_type_ids is not None:
                assert q_input_ids.shape == q_token_type_ids.shape

            # Perform inference using the query encoder.
            q_output = self.q_model.forward(input_ids=q_input_ids,
                                            attention_mask=q_attention_mask,
                                            token_type_ids=q_token_type_ids,
                                            output_attentions=output_attentions,
                                            output_hidden_states=output_hidden_states,
                                            return_dict=True)
            assert isinstance(q_output, ModelOutput)
        else:
            q_output = None

        if d_input_ids is not None and d_attention_mask is not None:
            # Verify that the shape of the input parameters matches.
            assert d_input_ids.shape == d_attention_mask.shape
            if d_token_type_ids is not None:
                assert d_input_ids.shape == d_token_type_ids.shape

            # Perform inference using the document encoder.
            d_output = self.d_model.forward(input_ids=d_input_ids,
                                            attention_mask=d_attention_mask,
                                            token_type_ids=d_token_type_ids,
                                            output_attentions=output_attentions,
                                            output_hidden_states=output_hidden_states,
                                            output_logits=output_logits,
                                            return_dict=True)
            assert isinstance(d_output, ModelOutput)
        else:
            d_output = None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if return_dict:
            return TrainOutput(q_output=q_output, d_output=d_output)
        else:
            return q_output, d_output

    def train(self, mode: bool = True):
        super().train(mode)

        self.q_model.train(mode)
        self.d_model.train(mode)
        self.d_mlm_head.train(mode)

        if self.aspects_linear is not None:
            self.aspects_linear.train(mode)

    def eval(self):
        super().eval()

        self.q_model.eval()
        self.d_model.eval()
        self.d_mlm_head.eval()

        if self.aspects_linear is not None:
            self.aspects_linear.eval()
