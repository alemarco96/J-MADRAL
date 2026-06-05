from modeling.generic.MLMHeadModule import MLMHeadModule
from modeling.generic.TrainDoubleOutput import TrainDoubleOutput
from modeling.loss.MLMLoss import MLMLoss
from modeling.model.TrainDoubleBiEncoderModel import TrainDoubleBiEncoderModel
import torch.nn
import transformers
from typing import Any


class AttemptPretrainingTrainer(transformers.Trainer):
    def __init__(self, *args, pretrain_alpha: float, logging_filename: str, **kwargs):
        super(AttemptPretrainingTrainer, self).__init__(*args, **kwargs)
        self.pretrain_alpha = pretrain_alpha

        self.logging_filename = logging_filename
        self.tot_steps = 0
        self.curr_steps = 0
        self.curr_losses = [0.0, 0.0, 0.0, 0.0]

        # Write the header to the logging file.
        with open(self.logging_filename, "wt", encoding="utf-8") as fo:
            print("Steps\tC-MLM\tC2A-MLM\tA2C-MLM\tLoss", file=fo, flush=True)
        del fo

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        assert isinstance(model, TrainDoubleBiEncoderModel)
        assert isinstance(model.d_mlm_head, MLMHeadModule)

        d_mlm_labels = inputs["d_mlm_labels"]
        aspects_mlm_labels = inputs["aspects_mlm_labels"]
        content_mlm_labels = inputs["content_mlm_labels"]

        # Compute the model output on the given input data: content only.
        encoder_output = model.forward(q_input_ids=None,
                                       q_attention_mask=None,
                                       q_token_type_ids=None,
                                       d_input_ids=inputs["d_input_ids"],
                                       d_attention_mask=inputs["d_attention_mask"],
                                       d_token_type_ids=inputs.get("d_token_type_ids", None),
                                       output_attentions=False,
                                       output_hidden_states=False,
                                       output_logits=True,
                                       return_dict=True)
        assert isinstance(encoder_output, TrainDoubleOutput)

        # ** We need to compute the d-MLM loss here! **
        encoder_output.d_output.mlm_logits = model.d_mlm_head(encoder_output.d_output.last_hidden_state)
        d_mlm_loss = MLMLoss.forward(encoder_output.d_output.mlm_logits, d_mlm_labels)
        del encoder_output

        # Compute the model output on the given input data: aspects masked on full text.
        encoder_output = model.forward(q_input_ids=None,
                                       q_attention_mask=None,
                                       q_token_type_ids=None,
                                       d_input_ids=inputs["aspects_input_ids"],
                                       d_attention_mask=inputs["full_attention_mask"],
                                       d_token_type_ids=inputs.get("full_token_type_ids", None),
                                       output_attentions=False,
                                       output_hidden_states=False,
                                       output_logits=True,
                                       return_dict=True)
        assert isinstance(encoder_output, TrainDoubleOutput)

        # ** We need to compute the c2a-MLM logits here! **
        encoder_output.d_output.mlm_logits = model.d_mlm_head(encoder_output.d_output.last_hidden_state)
        c2a_mlm_loss = MLMLoss.forward(encoder_output.d_output.mlm_logits, aspects_mlm_labels)
        del encoder_output

        # Compute the model output on the given input data: aspects masked on full text.
        encoder_output = model.forward(q_input_ids=None,
                                       q_attention_mask=None,
                                       q_token_type_ids=None,
                                       d_input_ids=inputs["content_input_ids"],
                                       d_attention_mask=inputs["full_attention_mask"],
                                       d_token_type_ids=inputs.get("full_token_type_ids", None),
                                       output_attentions=False,
                                       output_hidden_states=False,
                                       output_logits=True,
                                       return_dict=True)
        assert isinstance(encoder_output, TrainDoubleOutput)

        # ** We need to compute the c2a-MLM logits here! **
        encoder_output.d_output.mlm_logits = model.d_mlm_head(encoder_output.d_output.last_hidden_state)
        a2c_mlm_loss = MLMLoss.forward(encoder_output.d_output.mlm_logits, content_mlm_labels)
        del encoder_output

        loss = d_mlm_loss + self.pretrain_alpha * (c2a_mlm_loss + a2c_mlm_loss)

        self.curr_losses = [v1 + v2 for v1, v2 in zip(self.curr_losses, [float(torch.sum(d_mlm_loss.detach())),
                                                                         float(torch.sum(c2a_mlm_loss.detach())),
                                                                         float(torch.sum(a2c_mlm_loss.detach())),
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
            self.curr_losses = [0.0, 0.0, 0.0, 0.0]

        return loss
