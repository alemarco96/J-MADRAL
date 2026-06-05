from modeling.generic.TrainDoubleOutput import TrainDoubleOutput
from modeling.loss.CELoss import CELoss
from modeling.model.TrainDoubleBiEncoderModel import TrainDoubleBiEncoderModel
import torch.nn
import transformers
from typing import Any


class AttemptFinetuningTrainer(transformers.Trainer):
    def __init__(self, *args, logging_filename: str, **kwargs):
        super(AttemptFinetuningTrainer, self).__init__(*args, **kwargs)

        self.logging_filename = logging_filename
        self.tot_steps = 0
        self.curr_steps = 0
        self.curr_losses = [0.0, 0.0]

        # Write the header to the logging file.
        with open(self.logging_filename, "wt", encoding="utf-8") as fo:
            print("Steps\tCE\tLoss", file=fo, flush=True)
        del fo

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        # Compute the model output on the given input data.
        assert isinstance(model, TrainDoubleBiEncoderModel)
        if model.aspects_linear is not None:
            assert isinstance(model.aspects_linear, torch.nn.ModuleList)

        encoder_output = model.forward(q_input_ids=inputs["q_input_ids"],
                                       q_attention_mask=inputs["q_attention_mask"],
                                       q_token_type_ids=inputs.get("q_token_type_ids", None),
                                       d_input_ids=inputs["d_input_ids"],
                                       d_attention_mask=inputs["d_attention_mask"],
                                       d_token_type_ids=inputs.get("d_token_type_ids", None),
                                       output_attentions=False,
                                       output_hidden_states=False,
                                       output_logits=True,
                                       return_dict=True)
        assert isinstance(encoder_output, TrainDoubleOutput)

        ce_loss = CELoss.forward(encoder_output.q_output.pooler_output, encoder_output.d_output.pooler_output)
        loss = ce_loss

        self.curr_losses = [v1 + v2 for v1, v2 in zip(self.curr_losses, [float(torch.sum(ce_loss.detach())),
                                                                         float(torch.sum(loss.detach()))])]

        self.curr_steps += 1
        if self.curr_steps >= 1000:
            self.tot_steps += self.curr_steps

            with open(self.logging_filename, "at", encoding="utf-8") as fo:
                str_losses = "\t".join([f"{v / self.curr_steps:.6f}" for v in self.curr_losses])
                print(f"{self.tot_steps}\t{str_losses}", file=fo, flush=True)
                del str_losses
            del fo

            self.curr_steps = 0
            self.curr_losses = [0.0, 0.0]

        return loss
