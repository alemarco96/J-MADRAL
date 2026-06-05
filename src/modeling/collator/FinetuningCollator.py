import datasets
import torch
import transformers
from typing import Any


class FinetuningCollator(transformers.DataCollator):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizerBase,
                 # document_dataset: datasets.Dataset,
                 document_dataset: dict[str, dict[str, str | list[int]]],
                 # query_dataset: datasets.Dataset):
                 query_dataset: dict[int, dict[str, str | list[int]]]):
        self.tokenizer = tokenizer
        self.d_dataset = document_dataset
        self.q_dataset = query_dataset
        # self.q_dataset = {x["id"]: {"text": x["query"], "aspects": x["categories"]} for x in query_dataset}
        # self.d_dataset = {x["id"]: {"text": x["text"], "aspects": x["categories"]} for x in document_dataset}
        # self.d_dataset = document_dataset
        # self.q_dataset = query_dataset

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # # Extract all the keys of the batch data dictionary.
        # batch_keys = set.union(*[set(x.keys()) for x in batch])
        # assert all(x in batch_keys for x in ["query_id", "positives_id", "negatives_id"])
        # del batch_keys

        # Extract a map between queries and its set of positive documents.
        positives_map = {x["query_id"]: set(x["positives_id"]) for x in batch}

        # Extract the id of all queries and documents of the current batch.
        queries_id = sorted(x["query_id"] for x in batch)
        # documents_id = sorted(set.union(*[set(x["positives_id"]) for x in batch]) | \
        #                       set.union(*[set(x["negatives_id"]) for x in batch]))
        documents_id = sorted({v2 for v1 in batch for v2 in v1["positives_id"] + v1["negatives_id"]})

        # Extract the rows from the dataset corresponding to the current queries.
        # queries_row = self.q_dataset.filter(lambda x: x["id"] in queries_id, keep_in_memory=True)
        # queries_id = list(queries_row["id"])

        # Tokenize the text and extract the aspects of the current queries.
        # queries_input = self.tokenizer(list(queries_row["query"]),
        queries_input = self.tokenizer([self.q_dataset[k1]["text"] for k1 in queries_id],
                                       add_special_tokens=True,
                                       truncation=True,
                                       padding=True,
                                       return_tensors="pt")
        queries_aspects = torch.tensor([self.q_dataset[k]["aspects"] for k in queries_id], dtype=torch.int64)
        # queries_aspects = torch.tensor(list(queries_row["categories"]), dtype=torch.int64)
        # del queries_row

        # Extract the rows from the dataset corresponding to the current documents.
        # documents_row = self.d_dataset.filter(lambda x: x["id"] in documents_id, keep_in_memory=True)
        # documents_id = list(documents_row["id"])

        # Tokenize the text and extract the aspects of the current documents.
        # documents_input = self.tokenizer(list(documents_row["text"]),
        documents_input = self.tokenizer([self.d_dataset[k1]["text"] for k1 in documents_id],
                                         add_special_tokens=True,
                                         truncation=True,
                                         padding=True,
                                         return_tensors="pt")
        documents_aspects = torch.tensor([self.d_dataset[k]["aspects"] for k in documents_id], dtype=torch.int64)
        # documents_aspects = torch.tensor(list(documents_row["categories"]), dtype=torch.int64)
        # del documents_row

        # Create the labels tensor (BxN, [i][j]==1 if document j is relevant to query i, otherwise ==0).
        relevance_labels = torch.tensor([[1 if d in positives_map[q] else 0 for d in documents_id]
                                         for q in queries_id], dtype=torch.int64)
        del positives_map, queries_id, documents_id

        return {
            "q_input_ids": queries_input.input_ids,
            "q_attention_mask": queries_input.attention_mask,
            "q_token_type_ids": queries_input.get("token_type_ids", None),
            "q_aspects_labels": queries_aspects,
            "d_input_ids": documents_input.input_ids,
            "d_attention_mask": documents_input.attention_mask,
            "d_token_type_ids": documents_input.get("token_type_ids", None),
            "d_aspects_labels": documents_aspects,
            "qd_relevance_labels": relevance_labels
        }
