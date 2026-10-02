import datasets
import numpy
import operator
import random
import torch
import transformers
import typing


class FinetuningDataset(datasets.Dataset, transformers.TrainerCallback):
    # noinspection PyMissingConstructor
    def __init__(self,
                 p_corpus_data: dict[int | str, dict[str, typing.Any]],
                 r_corpus_data: dict[int | str, dict[str, typing.Any]],
                 p_queries_data: dict[int | str, dict[str, typing.Any]],
                 r_queries_data: dict[int | str, dict[str, typing.Any]],
                 p_train_data: dict[int | str, dict[str, typing.Any]],
                 r_train_data: dict[int | str, dict[str, typing.Any]],
                 batch_size: int = 64,
                 num_positives: int = 1,
                 num_negatives: int = 1):
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.epoch = 0

        self.p_corpus_data = p_corpus_data
        self.r_corpus_data = r_corpus_data

        self.p_queries_data = p_queries_data
        self.r_queries_data = r_queries_data

        self.p_train_data = {k1: {
            "p_id": [(k, r1, r2) for k, r1, r2 in v1["p_rank"]],
            "n_id": [(k, r1, r2) for k, r1, r2 in v1["n_rank"]],
            "hn_id": [(k, r1, r2) for k, r1, r2 in v1["hn_rank"]]
        } for k1, v1 in p_train_data.items()}
        self.p_train_data = {k: v for k, v in self.p_train_data.items() if len(v["p_id"]) > 0}
        self.p_keys = list(self.p_train_data.keys())

        self.r_train_data = {k1: {
            "p_id": [(k, r1, r2) for k, r1, r2 in v1["p_rank"]],
            "n_id": [(k, r1, r2) for k, r1, r2 in v1["n_rank"]],
            "hn_id": [(k, r1, r2) for k, r1, r2 in v1["hn_rank"]]
        } for k1, v1 in r_train_data.items()}
        self.r_train_data = {k: v for k, v in self.r_train_data.items() if len(v["p_id"]) > 0}
        self.r_keys = list(self.r_train_data.keys())

        self.train_keys = self.interleave_train_keys()


    # Ensure to have batches full of queries/documents for the same task.
    # If all the in-batch negatives are of the same type (products/reviews), the learning should be more effective.
    def interleave_train_keys(self) -> list[int | str]:
        # Interleave P and R keys (query_id) in blocks equal to the batch size.
        train_keys = []

        len_pr = max(len(self.p_keys), len(self.r_keys))
        for idx in range(0, len_pr, self.batch_size):
            for keys, task in zip((self.p_keys, self.r_keys), ("p", "r")):
                data = keys[idx: idx + self.batch_size]
                if len(data) == self.batch_size:
                    train_keys.append([(task, k) for k in data])
                elif len(data) <= 0:
                    pass
                else:
                    # Try to expand the last batch with additional elements selected from previous data.
                    # If there is not enough data, just discard the batch.
                    if idx >= self.batch_size - len(data):
                        data += random.sample(keys[:idx], self.batch_size - len(data))
                        train_keys.append([(task, k) for k in data])
                del data
            try:
                del keys, task
            except UnboundLocalError:
                pass
        try:
            del idx
        except UnboundLocalError:
            pass
        del len_pr

        # Return the final list obtained.
        return [v2 for v1 in train_keys for v2 in v1]


    def on_epoch_begin(self,
                       args: transformers.TrainingArguments,
                       state: transformers.TrainerState,
                       control: transformers.TrainerControl,
                       **kwargs):
        # Shuffle the data to be used for training in the current epoch.
        self.shuffle()


    def on_epoch_end(self,
                     args: transformers.TrainingArguments,
                     state: transformers.TrainerState,
                     control: transformers.TrainerControl,
                     **kwargs):
        self.epoch += 1


    def shuffle(self,
                seed: typing.Optional[int] = None,
                generator: typing.Optional[numpy.random.Generator] = None,
                keep_in_memory: bool = False,
                load_from_cache_file: typing.Optional[bool] = None,
                indices_cache_file_name: typing.Optional[str] = None,
                writer_batch_size: typing.Optional[int] = 1000,
                new_fingerprint: typing.Optional[str] = None
                ) -> datasets.Dataset:
        random.shuffle(self.p_keys)
        random.shuffle(self.r_keys)
        self.train_keys = self.interleave_train_keys()
        return self


    def take(self, n: int) -> datasets.Dataset:
        self.p_keys = self.p_keys[:n]
        self.r_keys = self.r_keys[:n]

        self.train_keys = self.interleave_train_keys()
        return self


    def __len__(self):
        return len(self.train_keys)


    def __getitem__(self, idx):
        # Convert arguments to list of ints.
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
            if len(idx) <= 0:
                raise ValueError("The given tensor is empty.")
            if not isinstance(idx[0], int):
                raise ValueError("The given tensor doesn't have integer dtype.")
        elif isinstance(idx, list):
            if len(idx) <= 0:
                raise ValueError("The given list is empty.")
        elif isinstance(idx, int):
            idx = [idx]
        else:
            raise NotImplementedError

        # Obtain the training triple(s).
        result = []
        for i1 in idx:
            task, query_id = self.train_keys[i1]

            if task == "p":
                q_data = self.p_queries_data[query_id]
                c_data = self.p_corpus_data
                t_data = self.p_train_data[query_id]
            elif task == "r":
                q_data = self.r_queries_data[query_id]
                c_data = self.r_corpus_data
                t_data = self.r_train_data[query_id]
            else:
                raise ValueError(f"The given task is not supported: {task}.")

            # Select the positive document(s).
            positives_id = t_data["p_id"]
            positives_id = [positives_id[(self.epoch * self.num_positives + i) % len(positives_id)][0]
                            for i in range(len(positives_id))][:self.num_positives]
            assert len(positives_id) == self.num_positives

            # Select the negative document(s).
            negatives_id = sorted(t_data["n_id"] + t_data["hn_id"],
                                  key=operator.itemgetter(1, 2), reverse=False)
            negatives_id = [negatives_id[(self.epoch * self.num_negatives + i) % len(negatives_id)][0]
                            for i in range(len(negatives_id))][:self.num_negatives]
            assert len(negatives_id) == self.num_negatives

            # Add the current output data to the result.
            result.append(
                {
                    "task": task,
                    "query_text": q_data["text"]
                } |
                {
                    f"positive{i}_text": c_data[doc_id]["text"] for i, doc_id in enumerate(positives_id)
                } |
                {
                    f"negative{i}_text": c_data[doc_id]["text"] for i, doc_id in enumerate(negatives_id)
                } |
                {
                    "query_aspects": q_data["aspects"]
                } |
                {
                    f"positive{i}_aspects": c_data[doc_id]["aspects"] for i, doc_id in enumerate(positives_id)
                } |
                {
                    f"negative{i}_aspects": c_data[doc_id]["aspects"] for i, doc_id in enumerate(negatives_id)
                }
            )
            del query_id, task, q_data, c_data, t_data, positives_id, negatives_id
        try:
            del i1
        except UnboundLocalError:
            pass

        batch_keys = set.union(*[set(b.keys()) for b in result])
        pos_keys = sorted({int(k[8:-5]) for k in batch_keys if k.startswith("positive") and k.endswith("_text")})
        neg_keys = sorted({int(k[8:-5]) for k in batch_keys if k.startswith("negative") and k.endswith("_text")})
        del batch_keys

        return \
            {
                "task": [b["task"] for b in result],
                "query_text": [b["query_text"] for b in result]
            } | {
                f"positive{i}_text": [b[f"positive{i}_text"] for b in result] for i in pos_keys
            } | {
                f"negative{i}_text": [b[f"negative{i}_text"] for b in result] for i in neg_keys
            } | {
                "query_aspects": [b["query_aspects"] for b in result],
            } | {
                f"positive{i}_aspects": [b[f"positive{i}_aspects"] for b in result] for i in pos_keys
            } | {
                f"negative{i}_aspects": [b[f"negative{i}_aspects"] for b in result] for i in neg_keys
            }
