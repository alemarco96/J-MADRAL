import torch
import transformers
from typing import Any


class PretrainingCollator(transformers.DataCollator):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 max_num_tokens: int | None = None,
                 mlm_probability: float = 0.15,
                 mask_replace_probability: float = 0.80,
                 random_replace_probability: float = 0.10):
        self.tokenizer = tokenizer
        self.max_num_tokens = max_num_tokens
        self.mlm_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                                         mlm_probability=mlm_probability,
                                                                         mask_replace_prob=mask_replace_probability,
                                                                         random_replace_prob=random_replace_probability)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Verify the given parameters are valid.
        assert batch is not None
        assert isinstance(batch, list)
        assert all(k in batch[0].keys() for k in ["text", "aspects"])

        # Tokenize the current batch data.
        batch_tokens = self.tokenizer([v["text"] for v in batch],
                                      add_special_tokens=True,
                                      truncation=True,
                                      padding=True,
                                      max_length=self.max_num_tokens,
                                      return_special_tokens_mask=True,
                                      return_tensors="pt")

        # Apply MLM masking.
        input_ids, mlm_labels = self.mlm_collator.torch_mask_tokens(batch_tokens.input_ids,
                                                                    batch_tokens.special_tokens_mask)

        # Extract the aspects labels.
        aspects_labels = torch.tensor([x["aspects"] for x in batch], dtype=torch.int64)

        if "token_type_ids" in batch_tokens.keys():
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
