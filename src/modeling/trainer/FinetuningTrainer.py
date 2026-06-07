from modeling.generic.TrainDoubleOutput import TrainOutput
from modeling.loss.APLoss import APLoss
from modeling.loss.CELoss import CELoss
from modeling.model.TrainBiEncoderModel import TrainBiEncoderModel
import torch.nn
import transformers
import typing


class FinetuningTrainer(transformers.Trainer):
    def __init__(self, *args, finetune_alpha: float, logging_filename: str, **kwargs):
        super(FinetuningTrainer, self).__init__(*args, **kwargs)
        self.finetune_alpha = finetune_alpha

        self.logging_filename = logging_filename
        self.tot_steps = 0
        self.curr_steps = 0
        self.curr_losses = [0.0, 0.0, 0.0, 0.0]

        # Write the header to the logging file.
        if self.logging_filename is not None:
            with open(self.logging_filename, "wt", encoding="utf-8") as fo:
                print("Steps\tCE\tq-AP\td-AP\tLoss", file=fo, flush=True)
            del fo

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | typing.Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        # Compute the model output on the given input data.
        assert isinstance(model, TrainBiEncoderModel)
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
        assert isinstance(encoder_output, TrainOutput)

        # Compute the CE loss.
        ce_loss = CELoss.forward(encoder_output.q_output.pooler_output, encoder_output.d_output.pooler_output)

        # Extract the aspects labels for both queries and documents.
        q_aspects_labels = inputs.get("q_aspects_labels", None)
        d_aspects_labels = inputs.get("d_aspects_labels", None)

        # ** We need to compute the AP logits here! **
        # Compute the class of each aspect. Do not compute for the aspect related to the [CLS] token.
        if model.aspects_linear is not None and q_aspects_labels is not None:
            encoder_output.q_output.aspects_logits = tuple(model.aspects_linear[i](
                encoder_output.q_output.aspects_embedding[:, i]) for i in range(q_aspects_labels.shape[1]))
        if model.aspects_linear is not None and d_aspects_labels is not None:
            encoder_output.d_output.aspects_logits = tuple(model.aspects_linear[i](
                encoder_output.d_output.aspects_embedding[:, i]) for i in range(d_aspects_labels.shape[1]))

        if encoder_output.q_output.aspects_logits is not None and q_aspects_labels is not None:
            q_ap_loss = APLoss.forward(encoder_output.q_output.aspects_logits, q_aspects_labels)
        else:
            q_ap_loss = torch.zeros([], dtype=torch.float32, device=encoder_output.d_output.pooler_output.device)
        if encoder_output.d_output.aspects_logits is not None and d_aspects_labels is not None:
            d_ap_loss = APLoss.forward(encoder_output.d_output.aspects_logits, d_aspects_labels)
        else:
            d_ap_loss = torch.zeros([], dtype=torch.float32, device=encoder_output.d_output.pooler_output.device)

        # Add AP loss to the CE loss, using self.finetune_alpha as scaling factor.
        if self.finetune_alpha != 0.0:
            loss = ce_loss + self.finetune_alpha * (q_ap_loss + d_ap_loss)
        else:
            loss = ce_loss

        self.curr_losses = [v1 + v2 for v1, v2 in zip(self.curr_losses, [float(torch.sum(ce_loss.detach())),
                                                                         float(torch.sum(q_ap_loss.detach())),
                                                                         float(torch.sum(d_ap_loss.detach())),
                                                                         float(torch.sum(loss.detach()))])]

        self.curr_steps += 1
        if self.curr_steps >= 1000:
            self.tot_steps += self.curr_steps

            if self.logging_filename is not None:
                with open(self.logging_filename, "at", encoding="utf-8") as fo:
                    str_losses = "\t".join([f"{v / self.curr_steps:.6f}" for v in self.curr_losses])
                    print(f"{self.tot_steps}\t{str_losses}", file=fo, flush=True)
                    del str_losses
                del fo

            self.curr_steps = 0
            self.curr_losses = [0.0, 0.0, 0.0, 0.0]

        return loss
