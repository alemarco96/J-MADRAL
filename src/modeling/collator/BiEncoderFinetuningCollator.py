import torch
import transformers
from typing import Any


class BiEncoderFinetuningCollator(transformers.DataCollator):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 max_num_tokens: int | None = 128):
        self.tokenizer = tokenizer
        self.max_num_tokens = max_num_tokens

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Extract all the keys of the batch data dictionary.
        batch_keys = set.union(*[set(x.keys()) for x in batch])

        # Determine which are the positives and negatives in the current batch.
        num_positives = sorted({int(k[8:-5]) for k in batch_keys
                                if k.startswith("positive") and k.endswith("_text")})
        num_negatives = sorted({int(k[8:-5]) for k in batch_keys
                                if k.startswith("negative") and k.endswith("_text")})

        # Determine the sequence of fields to tokenize, when considering documents.
        fields_to_tokenize = [f"positive{i}" for i in num_positives] + \
                             [f"negative{i}" for i in num_negatives]
        del num_positives, num_negatives

        # Tokenize the queries.
        q_tokens_id = self.tokenizer([b["query_text"] for b in batch],
                                     add_special_tokens=True,
                                     truncation=True,
                                     padding=True,
                                     max_length=self.max_num_tokens,
                                     return_tensors="pt")

        # Tokenize the documents.
        d_tokens_id = self.tokenizer([b[f"{f}_text"] for f in fields_to_tokenize for b in batch],
                                     add_special_tokens=True,
                                     truncation=True,
                                     padding=True,
                                     max_length=self.max_num_tokens,
                                     return_tensors="pt")

        # Prepare the aspects labels.
        q_aspects_labels = torch.tensor([b["query_aspects"] for b in batch],
                                        dtype=torch.int64, device=q_tokens_id.input_ids.device)
        d_aspects_labels = torch.tensor([b[f"{f}_aspects"] for f in fields_to_tokenize for b in batch],
                                        dtype=torch.int64, device=d_tokens_id.input_ids.device)
        del fields_to_tokenize

        return {
            "task": [b["task"] for b in batch],
            "q_input_ids": q_tokens_id.input_ids,
            "q_attention_mask": q_tokens_id.attention_mask,
            "q_token_type_ids": q_tokens_id.get("token_type_ids", None),
            "q_aspects_labels": q_aspects_labels,
            "d_input_ids": d_tokens_id.input_ids,
            "d_attention_mask": d_tokens_id.attention_mask,
            "d_token_type_ids": d_tokens_id.get("token_type_ids", None),
            "d_aspects_labels": d_aspects_labels
        }
