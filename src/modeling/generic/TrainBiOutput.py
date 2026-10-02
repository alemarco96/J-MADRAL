import dataclasses
from modeling.generic.ModelOutput import ModelOutput


@dataclasses.dataclass
class TrainBiOutput:
    q_output: ModelOutput | None = None
    d_output: ModelOutput | None = None

    def as_tuple(self) -> tuple[ModelOutput | None, ModelOutput | None]:
        return self.q_output, self.d_output
