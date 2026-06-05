import torch
import transformers
from typing import Any


class MadralPretrainingCollator(transformers.DataCollator):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 mlm_probability: float = 0.15,
                 mask_replace_probability: float = 0.80,
                 random_replace_probability: float = 0.10):
        self.tokenizer = tokenizer
        self.mlm_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                                         mlm_probability=mlm_probability,
                                                                         mask_replace_prob=mask_replace_probability,
                                                                         random_replace_prob=random_replace_probability)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Build the BatchEncoding for the current batch.
        if "token_type_ids" in batch[0].keys():
            batch_tokens =  self.tokenizer.pad({
                "input_ids": [x["input_ids"] for x in batch],
                "attention_mask": [x["attention_mask"] for x in batch],
                "token_type_ids": [x["token_type_ids"] for x in batch],
                "special_tokens_mask": [x["special_tokens_mask"] for x in batch]
            }, return_tensors="pt")
        else:
            batch_tokens = self.tokenizer.pad({
                "input_ids": [x["input_ids"] for x in batch],
                "attention_mask": [x["attention_mask"] for x in batch],
                "special_tokens_mask": [x["special_tokens_mask"] for x in batch]
            }, return_tensors="pt")

        # Apply MLM masking.
        input_ids, mlm_labels = self.mlm_collator.torch_mask_tokens(batch_tokens.input_ids,
                                                                    batch_tokens.special_tokens_mask)

        # Extract the aspects labels.
        aspects_labels = torch.tensor([x["categories"] for x in batch], dtype=torch.int64)

        if "token_type_ids" in batch[0].keys():
            return {
                "d_input_ids": input_ids,
                "d_attention_mask": batch_tokens.attention_mask,
                "d_token_type_ids": batch_tokens.token_type_ids,
                "mlm_labels": mlm_labels,
                "aspects_labels": aspects_labels
            }
        else:
            return {
                "d_input_ids": input_ids,
                "d_attention_mask": batch_tokens.attention_mask,
                "mlm_labels": mlm_labels,
                "aspects_labels": aspects_labels
            }
