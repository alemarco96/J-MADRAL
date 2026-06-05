import torch
import transformers
from typing import Any


class MadralFinetuningCollator(transformers.DataCollator):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Extract all the keys of the batch data dictionary.
        batch_keys = set.union(*[set(x.keys()) for x in batch])

        # Determine which are the positives and negatives in the current batch.
        num_positives = sorted({int(k[8:-10]) for k in batch_keys
                                if k.startswith("positive") and k.endswith("_input_ids")})
        num_negatives = sorted({int(k[8:-10]) for k in batch_keys
                                if k.startswith("negative") and k.endswith("_input_ids")})
        fields_to_tokenize = [f"positive{i}" for i in num_positives] + [f"negative{i}" for i in num_negatives]
        del num_positives, num_negatives


        if "query_token_type_ids" in batch[0].keys():
            # Prepare the input for the queries.
            q_tokens_id = self.tokenizer.pad({
                "input_ids": [b["query_input_ids"] for b in batch],
                "attention_mask": [b["query_attention_mask"] for b in batch],
                "token_type_ids": [b["query_token_type_ids"] for b in batch]
            }, return_tensors="pt")

            # Prepare the input for the documents.
            d_tokens_id = self.tokenizer.pad({
                "input_ids": [b[f"{f}_input_ids"] for f in fields_to_tokenize for b in batch],
                "attention_mask": [b[f"{f}_attention_mask"] for f in fields_to_tokenize for b in batch],
                "token_type_ids": [b[f"{f}_token_type_ids"] for f in fields_to_tokenize for b in batch]
            }, return_tensors="pt")
        else:
            # Prepare the input for the queries.
            q_tokens_id = self.tokenizer.pad({
                "input_ids": [b["query_input_ids"] for b in batch],
                "attention_mask": [b["query_attention_mask"] for b in batch]
            }, return_tensors="pt")

            # Prepare the input for the documents.
            d_tokens_id = self.tokenizer.pad({
                "input_ids": [b[f"{f}_input_ids"] for f in fields_to_tokenize for b in batch],
                "attention_mask": [b[f"{f}_attention_mask"] for f in fields_to_tokenize for b in batch]
            }, return_tensors="pt")

        # Prepare the aspects labels.
        q_aspects_labels = torch.tensor([b["query_aspects"] for b in batch],
                                        dtype=torch.int64, device=q_tokens_id.input_ids.device)
        d_aspects_labels = torch.tensor([b[f"{f}_aspects"] for f in fields_to_tokenize for b in batch],
                                        dtype=torch.int64, device=d_tokens_id.input_ids.device)

        return {
            "q_input_ids": q_tokens_id.input_ids,
            "q_attention_mask": q_tokens_id.attention_mask,
            "q_token_type_ids": q_tokens_id.get("token_type_ids", None),
            "q_aspects_labels": q_aspects_labels,
            "d_input_ids": d_tokens_id.input_ids,
            "d_attention_mask": d_tokens_id.attention_mask,
            "d_token_type_ids": d_tokens_id.get("token_type_ids", None),
            "d_aspects_labels": d_aspects_labels
        }
