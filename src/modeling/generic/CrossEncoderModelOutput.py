import dataclasses
import torch
import transformers.modeling_outputs
import transformers.utils


@dataclasses.dataclass
class CrossEncoderModelOutput(transformers.utils.ModelOutput):
    attentions: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    last_hidden_state: torch.FloatTensor | None = None
    mlm_logits: torch.FloatTensor | None = None
    q_aspects_embedding: torch.FloatTensor | None = None
    d_aspects_embedding: torch.FloatTensor | None = None
    q_aspects_logits: tuple[torch.FloatTensor, ...] | None = None
    d_aspects_logits: tuple[torch.FloatTensor, ...] | None = None
    relevance_logits: torch.FloatTensor | None = None

    def as_tuple(self) -> tuple[torch.FloatTensor | tuple[torch.FloatTensor, ...] | None, ...]:
        # Put "relevance_logit" in first position, then all the other fields ordered by name.
        return ((self.relevance_logits, self.attentions, self.d_aspects_embedding, self.d_aspects_logits,
                 self.hidden_states, self.last_hidden_state, self.mlm_logits, self.q_aspects_embedding,
                 self.q_aspects_logits))

    @staticmethod
    def from_base_output(output: transformers.modeling_outputs.ModelOutput):
        return CrossEncoderModelOutput(attentions=output.get("attentions", None),
                                       hidden_states=output.get("hidden_states", None),
                                       last_hidden_state=output.get("last_hidden_state", None),
                                       mlm_logits=None,
                                       q_aspects_embedding=None,
                                       q_aspects_logits=None,
                                       d_aspects_embedding=None,
                                       d_aspects_logits=None,
                                       relevance_logits=None)
