from modeling.generic.MLMHeadModule import MLMHeadModule
import torch.nn
import transformers.models.modernbert.modeling_modernbert


class ModernBertMLMHead(MLMHeadModule):
    tokens_embedding_weight_name: str | None = "decoder.weight"

    def __init__(self, config):
        super(ModernBertMLMHead, self).__init__()
        self.head = transformers.models.modernbert.modeling_modernbert.ModernBertPredictionHead(config)
        self.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(last_hidden_state))
