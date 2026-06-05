import dataclasses
import torch
import transformers.modeling_outputs
import transformers.utils


@dataclasses.dataclass
class ModelOutput(transformers.utils.ModelOutput):
    attentions: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    mlm_logits: torch.FloatTensor | None = None
    aspects_embedding: torch.FloatTensor | None = None
    aspects_logits: tuple[torch.FloatTensor, ...] | None = None
    presence_logits: torch.FloatTensor | None = None
    presence_weights: torch.FloatTensor | None = None

    def as_tuple(self) -> tuple[torch.FloatTensor | tuple[torch.FloatTensor, ...] | None, ...]:
        # Put "pooler_output" in first position, then all the other fields ordered by name.
        return ((self.pooler_output, self.aspects_embedding, self.aspects_logits, self.attentions, self.hidden_states,
                 self.last_hidden_state, self.mlm_logits, self.presence_logits, self.presence_weights))

    @staticmethod
    def from_base_output(output: transformers.modeling_outputs.ModelOutput):
        return ModelOutput(attentions=output.get("attentions", None),
                           hidden_states=output.get("hidden_states", None),
                           last_hidden_state=output.get("last_hidden_state", None),
                           pooler_output=output.get("pooler_output", None),
                           mlm_logits=None,
                           aspects_embedding=None,
                           aspects_logits=None,
                           presence_logits=None,
                           presence_weights=None)
