import torch
import transformers
import typing


class CrossFinetuningCollator(transformers.DataCollator):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 max_num_tokens: int | None = 128):
        self.tokenizer = tokenizer
        self.max_num_tokens = max_num_tokens

    def __call__(self, batch: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
        # Tokenize the text.
        tokens_id = self.tokenizer([b["query_text"] for b in batch],
                                   text_pair=[b["document_text"] for b in batch],
                                   add_special_tokens=True,
                                   truncation=True,
                                   padding=True,
                                   max_length=self.max_num_tokens,
                                   return_tensors="pt")

        # Extract the labels to use for training.
        q_aspects_labels = torch.tensor([b[f"query_aspects"] for b in batch], dtype=torch.int64)
        d_aspects_labels = torch.tensor([b[f"document_aspects"] for b in batch], dtype=torch.int64)
        qd_labels = torch.tensor([b[f"qd_label"] for b in batch], dtype=torch.int64)

        return {
            "input_ids": tokens_id.input_ids,
            "attention_mask": tokens_id.attention_mask,
            "token_type_ids": tokens_id.get("token_type_ids", None),
            "q_aspects_labels": q_aspects_labels,
            "d_aspects_labels": d_aspects_labels,
            "qd_labels": qd_labels
        }
