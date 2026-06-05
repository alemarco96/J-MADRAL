import datasets
import numpy
import operator
import random
import torch
import transformers
import typing


class JointFinetuningDataset(datasets.Dataset, transformers.TrainerCallback):
    # noinspection PyMissingConstructor
    def __init__(self,
                 r_corpus_data: dict[int | str, dict[str, typing.Any]],
                 s_corpus_data: dict[int | str, dict[str, typing.Any]],
                 r_train_data: dict[int | str, dict[str, typing.Any]],
                 s_train_data: dict[int | str, dict[str, typing.Any]],
                 rs_train_data: dict[int | str, dict[str, typing.Any]],
                 sr_train_data: dict[int | str, dict[str, typing.Any]],
                 batch_size: int = 64,
                 num_warmup_epochs: int = 4,
                 every_k_epochs: int = 4,
                 min_rank_for_hardnegatives: int = 50,
                 max_rank_for_hardnegatives: int = 100,
                 num_positives: int = 1,
                 num_negatives: int = 1,
                 num_retain_hardnegatives: int = 100):
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.min_rank_for_hardnegatives = min_rank_for_hardnegatives
        self.max_rank_for_hardnegatives = max_rank_for_hardnegatives
        self.num_retain_hardnegatives = num_retain_hardnegatives
        self.batch_size = batch_size

        self.epoch = 0
        self.num_warmup_epochs = num_warmup_epochs
        self.every_k_epochs = every_k_epochs

        self.r_corpus_data = r_corpus_data
        self.s_corpus_data = s_corpus_data

        self.r_queries_data = {k: {"text": v["query_text"], "aspects": v["query_aspects"]}
                               for k, v in r_train_data.items()}
        self.s_queries_data = {k: {"text": v["query_text"], "aspects": v["query_aspects"]}
                               for k, v in s_train_data.items()}
        self.rs_queries_data = {k: {"text": v["query_text"], "aspects": v["query_aspects"]}
                               for k, v in rs_train_data.items()}
        self.sr_queries_data = {k: {"text": v["query_text"], "aspects": v["query_aspects"]}
                               for k, v in sr_train_data.items()}

        self.r_exclude_data = {k: set(v["exclude_id"]) for k, v in r_train_data.items()}
        self.s_exclude_data = {k: set(v["exclude_id"]) for k, v in s_train_data.items()}
        self.rs_exclude_data = {k: set(v["exclude_id"]) for k, v in rs_train_data.items()}
        self.sr_exclude_data = {k: set(v["exclude_id"]) for k, v in sr_train_data.items()}

        self.r_train_data = {k1: {
            "p_id": [(k, r1, r2) for k, r1, r2 in v1["p_rank"]],
            "n_id": [(k, r1, r2) for k, r1, r2 in v1["n_rank"]],
            "hn_id": [(k, r1, r2) for k, r1, r2 in v1["hn_rank"][:num_retain_hardnegatives]]
        } for k1, v1 in r_train_data.items()}
        self.r_train_data = {k: v for k, v in self.r_train_data.items() if len(v["p_id"]) > 0}
        self.r_keys = list(self.r_train_data.keys())

        self.s_train_data = {k1: {
            "p_id": [(k, r1, r2) for k, r1, r2 in v1["p_rank"]],
            "n_id": [(k, r1, r2) for k, r1, r2 in v1["n_rank"]],
            "hn_id": [(k, r1, r2) for k, r1, r2 in v1["hn_rank"][:num_retain_hardnegatives]]
        } for k1, v1 in s_train_data.items()}
        self.s_train_data = {k: v for k, v in self.s_train_data.items() if len(v["p_id"]) > 0}
        self.s_keys = list(self.s_train_data.keys())

        self.rs_train_data = {k1: {
            "p_id": [(k, r1, r2) for k, r1, r2 in v1["p_rank"]],
            "n_id": [(k, r1, r2) for k, r1, r2 in v1["n_rank"]],
            "hn_id": [(k, r1, r2) for k, r1, r2 in v1["hn_rank"][:num_retain_hardnegatives]]
        } for k1, v1 in rs_train_data.items()}
        self.rs_train_data = {k: v for k, v in self.rs_train_data.items() if len(v["p_id"]) > 0}
        self.rs_keys = list(self.rs_train_data.keys())

        self.sr_train_data = {k1: {
            "p_id": [(k, r1, r2) for k, r1, r2 in v1["p_rank"]],
            "n_id": [(k, r1, r2) for k, r1, r2 in v1["n_rank"]],
            "hn_id": [(k, r1, r2) for k, r1, r2 in v1["hn_rank"][:num_retain_hardnegatives]]
        } for k1, v1 in sr_train_data.items()}
        self.sr_train_data = {k: v for k, v in self.sr_train_data.items() if len(v["p_id"]) > 0}
        self.sr_keys = list(self.sr_train_data.keys())

        self.train_keys = self.interleave_train_keys()


    # Ensure to have batches full of queries/documents for the same task.
    # If all the in-batch negatives are of the same type (products/reviews), the learning should be more effective.
    def interleave_train_keys(self) -> list[int | str]:
        # Interleave R and S keys (query_id) in blocks equal to the batch size.
        train_keys = []

        len_rs = max(len(self.r_keys), len(self.s_keys), len(self.rs_keys), len(self.sr_keys))
        for idx in range(0, len_rs, self.batch_size):
            for keys, task in zip((self.r_keys, self.s_keys, self.rs_keys, self.sr_keys),
                                  ("r", "s", "rs", "sr")):
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
        del len_rs

        # Return the final list obtained.
        # random.shuffle(train_keys)
        return [v2 for v1 in train_keys for v2 in v1]


    def update_train_data(self, task: str, run_data: dict[int | str, list[int | str]]) -> None:
        if task is None or not isinstance(task, str) or task not in {"r", "s", "rs", "sr"}:
            raise ValueError(f"The given task is not supported: {task}.")

        # --------------------------------------------------------------------------------------------------------------
        # Update the hard negatives, using the run data.
        # --------------------------------------------------------------------------------------------------------------
        not_found = 1000

        for query_id in run_data.keys():
            if task == "r":
                train_data = self.r_train_data
                exclude_data = self.r_exclude_data
            elif task == "s":
                train_data = self.s_train_data
                exclude_data = self.s_exclude_data
            elif task == "rs":
                train_data = self.rs_train_data
                exclude_data = self.rs_exclude_data
            elif task == "sr":
                train_data = self.sr_train_data
                exclude_data = self.sr_exclude_data
            else:
                raise ValueError(f"The given task is not supported: {task}.")

            # Do not consider queries not included in the training data.
            if query_id not in train_data.keys():
                del train_data, exclude_data
                continue

            # Extract the relevant data for the given query.
            t_data = train_data[query_id]
            run = {k: i for i, k in enumerate(run_data[query_id])}
            exclude_id = exclude_data.get(query_id, set())

            # Compute the new positives, negatives, and hard negatives.
            p_id = sorted([(k, run.get(k, not_found), r1) for k, r1, _ in t_data["p_id"]],
                          key=operator.itemgetter(1, 2), reverse=False)
            n_id = sorted([(k, run.get(k, not_found), r1) for k, r1, _ in t_data["n_id"]],
                          key=operator.itemgetter(1, 2), reverse=False)
            hn_id = set.union(*[{k for k, i in run.items() if k not in exclude_id and \
                                 self.min_rank_for_hardnegatives <= i < self.max_rank_for_hardnegatives},
                                {k for k, _ in t_data["hn_id"]}])
            old_hn_id = {k: r1 for k, r1, _ in t_data["hn_id"]}
            hn_id = sorted([(k, run.get(k, not_found), old_hn_id.get(k, not_found)) for k in hn_id],
                           key=operator.itemgetter(1, 2), reverse=False)[:self.num_retain_hardnegatives]
            del old_hn_id, t_data, run, exclude_id

            value_data = {"p_id": p_id, "n_id": n_id, "hn_id": hn_id}
            del p_id, n_id, hn_id

            # Save the new documents into the training data.
            if task == "r":
                self.r_train_data[query_id] = value_data
            elif task == "s":
                self.s_train_data[query_id] = value_data
            elif task == "rs":
                self.rs_train_data[query_id] = value_data
            elif task == "sr":
                self.sr_train_data[query_id] = value_data
            else:
                raise ValueError(f"The given task is not supported: {task}.")
            del value_data
        try:
            del query_id
        except UnboundLocalError:
            pass
        del not_found


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
        random.shuffle(self.r_keys)
        random.shuffle(self.s_keys)
        random.shuffle(self.rs_keys)
        random.shuffle(self.sr_keys)
        self.train_keys = self.interleave_train_keys()
        return self


    def take(self, n: int) -> datasets.Dataset:
        self.r_keys = self.r_keys[:n]
        self.s_keys = self.s_keys[:n]
        self.rs_keys = self.rs_keys[:n]
        self.sr_keys = self.sr_keys[:n]

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

            if task == "r":
                q_data = self.r_queries_data[query_id]
                c_data = self.r_corpus_data
                t_data = self.r_train_data[query_id]
            elif task == "s":
                q_data = self.s_queries_data[query_id]
                c_data = self.s_corpus_data
                t_data = self.s_train_data[query_id]
            elif task == "rs":
                q_data = self.rs_queries_data[query_id]
                c_data = self.s_corpus_data
                t_data = self.rs_train_data[query_id]
            elif task == "sr":
                q_data = self.sr_queries_data[query_id]
                c_data = self.r_corpus_data
                t_data = self.sr_train_data[query_id]
            else:
                raise ValueError(f"The given task is not supported: {task}.")

            # Select the positive document(s).
            positives_id = t_data["p_id"]
            if self.epoch < self.num_warmup_epochs:  # + self.every_k_epochs:
                positives_id = [positives_id[(self.epoch * self.num_positives + i) % len(positives_id)][0]
                                for i in range(len(positives_id))][:self.num_positives]
            else:
                epoch_idx = (self.epoch - self.num_warmup_epochs) % self.every_k_epochs
                positives_id = [positives_id[len(positives_id) - 1 - \
                                             ((epoch_idx * self.num_positives + i) % len(positives_id))][0]
                                for i in range(len(positives_id))][:self.num_positives]
                del epoch_idx
            assert len(positives_id) == self.num_positives

            # Select the negative document(s).
            negatives_id = sorted(t_data["n_id"] + t_data["hn_id"],
                                  key=operator.itemgetter(1, 2), reverse=False)
            if self.epoch < self.num_warmup_epochs:
                negatives_id = [negatives_id[(self.epoch * self.num_negatives + i) % len(negatives_id)][0]
                                for i in range(len(negatives_id))][:self.num_negatives]
            else:
                epoch_idx = (self.epoch - self.num_warmup_epochs) % self.every_k_epochs
                negatives_id = [negatives_id[(epoch_idx * self.num_negatives + i) % len(negatives_id)][0]
                                for i in range(len(negatives_id))][:self.num_negatives]
                del epoch_idx
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
