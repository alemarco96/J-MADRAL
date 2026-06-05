from modeling.generic.MLMHeadModule import MLMHeadModule
import torch.nn
import transformers.models.bert.modeling_bert


class BertMLMHead(MLMHeadModule):
    tokens_embedding_weight_name: str | None = "mlm_head.cls.predictions.decoder.weight"

    def __init__(self, config):
        super(BertMLMHead, self).__init__()
        self.mlm_head = transformers.models.bert.modeling_bert.BertOnlyMLMHead(config)

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        return self.mlm_head(last_hidden_state)
