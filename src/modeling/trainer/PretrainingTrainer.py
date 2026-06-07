from modeling.generic.MLMHeadModule import MLMHeadModule
from modeling.generic.TrainDoubleOutput import TrainOutput
from modeling.loss.APLoss import APLoss
from modeling.loss.MLMLoss import MLMLoss
from modeling.model.TrainBiEncoderModel import TrainBiEncoderModel
import torch.nn
import transformers
import typing


class PretrainingTrainer(transformers.Trainer):
    def __init__(self, *args, pretrain_alpha: float, logging_filename: str, **kwargs):
        super(PretrainingTrainer, self).__init__(*args, **kwargs)
        self.pretrain_alpha = pretrain_alpha

        self.logging_filename = logging_filename
        self.tot_steps = 0
        self.curr_steps = 0
        self.curr_losses = [0.0, 0.0, 0.0]

        # Write the header to the logging file.
        if self.logging_filename is not None:
            with open(self.logging_filename, "wt", encoding="utf-8") as fo:
                print("Steps\tMLM\tAP\tLoss", file=fo, flush=True)
            del fo

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | typing.Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        mlm_labels = inputs["mlm_labels"]
        aspects_labels = inputs["aspects_labels"]

        # Compute the model output on the given input data.
        assert isinstance(model, TrainBiEncoderModel)
        assert isinstance(model.d_mlm_head, MLMHeadModule)
        if model.aspects_linear is not None:
            assert isinstance(model.aspects_linear, torch.nn.ModuleList)

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
        assert isinstance(encoder_output, TrainOutput)

        # ** We need to compute the MLM and AP logits here! **
        # Use the MLM head to perform prediction.
        encoder_output.d_output.mlm_logits = model.d_mlm_head(encoder_output.d_output.last_hidden_state)

        # Compute the class of each aspect. Do not compute for the aspect related to the [CLS] token.
        if model.aspects_linear is not None:
            encoder_output.d_output.aspects_logits = tuple(model.aspects_linear[i](
                encoder_output.d_output.aspects_embedding[:, i]) for i in range(aspects_labels.shape[1]))

        mlm_loss = MLMLoss.forward(encoder_output.d_output.mlm_logits, mlm_labels)
        if encoder_output.d_output.aspects_logits is not None and aspects_labels is not None:
            ap_loss = APLoss.forward(encoder_output.d_output.aspects_logits, aspects_labels)
        else:
            ap_loss = torch.zeros([], dtype=torch.float32, device=encoder_output.d_output.last_hidden_state.device)

        loss = mlm_loss + self.pretrain_alpha * ap_loss

        self.curr_losses = [v1 + v2 for v1, v2 in zip(self.curr_losses, [float(torch.sum(mlm_loss.detach())),
                                                                         float(torch.sum(ap_loss.detach())),
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
            self.curr_losses = [0.0, 0.0, 0.0]

        return loss
