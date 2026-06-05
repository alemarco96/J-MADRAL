import dataclasses
import torch


@dataclasses.dataclass
class ModelInput:
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    query_mask: torch.Tensor | None = None
    document_mask: torch.Tensor | None = None
    token_type_ids: torch.Tensor | None = None
    output_attentions: bool | None = None
    output_hidden_states: bool | None = None
    output_logits: bool | None = None
    return_dict: bool | None = None
