import torch
import transformers
from typing import Any


class AttemptPretrainingCollator(transformers.DataCollator):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 aspects_mlm_probability: float = 0.60,
                 aspects_mask_replace_probability: float = 0.80,
                 aspects_random_replace_probability: float = 0.10,
                 content_mlm_probability: float = 0.15,
                 content_mask_replace_probability: float = 0.80,
                 content_random_replace_probability: float = 0.10):
        self.tokenizer = tokenizer
        self.aspects_mlm_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=aspects_mlm_probability,
            mask_replace_prob=aspects_mask_replace_probability,
            random_replace_prob=aspects_random_replace_probability)
        self.content_mlm_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=content_mlm_probability,
            mask_replace_prob=content_mask_replace_probability,
            random_replace_prob=content_random_replace_probability)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Build the BatchEncoding for the current batch.
        if "full_token_type_ids" in batch[0].keys():
            batch_full_tokens = self.tokenizer.pad({
                "input_ids": [x["full_input_ids"] for x in batch],
                "attention_mask": [x["full_attention_mask"] for x in batch],
                "token_type_ids": [x["full_token_type_ids"] for x in batch],
                "special_tokens_mask": [x["full_special_tokens_mask"] for x in batch]
            }, return_tensors="pt")

            batch_content_tokens = self.tokenizer.pad({
                "input_ids": [x["content_input_ids"] for x in batch],
                "attention_mask": [x["content_attention_mask"] for x in batch],
                "token_type_ids": [x["content_token_type_ids"] for x in batch],
                "special_tokens_mask": [x["content_special_tokens_mask"] for x in batch]
            }, return_tensors="pt")
        else:
            batch_full_tokens = self.tokenizer.pad({
                "input_ids": [x["full_input_ids"] for x in batch],
                "attention_mask": [x["full_attention_mask"] for x in batch],
                "special_tokens_mask": [x["full_special_tokens_mask"] for x in batch]
            }, return_tensors="pt")

            batch_content_tokens = self.tokenizer.pad({
                "input_ids": [x["content_input_ids"] for x in batch],
                "attention_mask": [x["content_attention_mask"] for x in batch],
                "special_tokens_mask": [x["content_special_tokens_mask"] for x in batch]
            }, return_tensors="pt")

        # Find the location of the first [SEP] token.
        sep1_index = torch.where(batch_full_tokens.input_ids == self.tokenizer.sep_token_id)[1][0::2].unsqueeze(1)
        assert sep1_index.shape[0] == batch_full_tokens.input_ids.shape[0]

        # Create two different masks signaling the tokens for the aspects and content, respectively.
        mask = torch.arange(batch_full_tokens.input_ids.shape[1]).unsqueeze(0).expand(sep1_index.shape[0], -1)
        aspects_mask = batch_full_tokens.special_tokens_mask.masked_fill(mask > sep1_index, 1)
        content_mask = batch_full_tokens.special_tokens_mask.masked_fill(mask < sep1_index, 1)
        del mask, sep1_index

        # Apply MLM masking separately to the aspects and content tokens.
        aspects_full_input_ids, aspects_full_mlm_labels = self.aspects_mlm_collator.torch_mask_tokens(
            batch_full_tokens.input_ids.clone(), aspects_mask)
        content_full_input_ids, content_full_mlm_labels = self.content_mlm_collator.torch_mask_tokens(
            batch_full_tokens.input_ids.clone(), content_mask)
        del aspects_mask, content_mask

        # Apply MLM masking to the content only tokens.
        d_input_ids, d_mlm_labels = self.content_mlm_collator.torch_mask_tokens(
            batch_content_tokens.input_ids.clone(), batch_content_tokens.special_tokens_mask)

        return {
            "d_input_ids": d_input_ids,
            "d_attention_mask": batch_content_tokens.attention_mask,
            "d_token_type_ids": batch_content_tokens.get("token_type_ids", None),
            "d_mlm_labels": d_mlm_labels,
            "aspects_input_ids": aspects_full_input_ids,
            "content_input_ids": content_full_input_ids,
            "full_attention_mask": batch_full_tokens.attention_mask,
            "full_token_type_ids": batch_full_tokens.get("token_type_ids", None),
            "aspects_mlm_labels": aspects_full_mlm_labels,
            "content_mlm_labels": content_full_mlm_labels
        }
