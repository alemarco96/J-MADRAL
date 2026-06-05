import datasets
import numpy
import operator
import random
import torch
import transformers
import typing


def parse_cat_id(cat_id: str) -> tuple[int, ...]:
    if cat_id == "":
        return ()
    else:
        return tuple(int(v) for v in cat_id.split("-"))


class JointFinetuningDataset(datasets.Dataset, transformers.TrainerCallback):
    # noinspection PyMissingConstructor
    def __init__(self,
                 r_corpus_data: dict[int | str, dict[str, typing.Any]],
                 s_corpus_data: dict[int | str, dict[str, typing.Any]],
                 r2s_mapping_data: dict[int | str, list[int | str]],
                 cat2r_mapping_data: dict[str, list[int | str]],
                 r_train_data: dict[int | str, dict[str, typing.Any]],
                 s_train_data: dict[int | str, dict[str, typing.Any]],
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

        self.r2s_mapping_data = {k: v for k, v in r2s_mapping_data.items() if len(v) > 0}

        self.cat2r_mapping_data = {str(k): v for k, v in cat2r_mapping_data.items()}
        self.r2cat_mapping_data = {v2: str(k1) for k1, v1 in self.cat2r_mapping_data.items() for v2 in v1}

        self.r_queries_data = {k: {"text": v["query_text"], "aspects": v["query_aspects"]}
                               for k, v in r_train_data.items()}
        self.s_queries_data = {k: {"text": v["query_text"], "aspects": v["query_aspects"]}
                               for k, v in s_train_data.items()}

        self.r_exclude_data = {k: set(v["exclude_id"]) for k, v in r_train_data.items()}
        self.s_exclude_data = {k: set(v["exclude_id"]) for k, v in s_train_data.items()}

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

        self.rs_train_data, self.sr_train_data = self.build_rs_sr_train_data(len(self.r_keys), len(self.s_keys))
        self.rs_keys = list(self.rs_train_data.keys())
        self.sr_keys = list(self.sr_train_data.keys())

        self.train_keys = self.interleave_train_keys()


    def build_rs_sr_train_data(self, num_rs: int, num_sr: int) -> tuple[dict, dict]:
        # --------------------------------------------------------------------------------------------------------------
        # Shuffle the lists in the mapping data.
        # --------------------------------------------------------------------------------------------------------------
        # Shuffle the list of reviews associated with each product.
        for v in self.r2s_mapping_data.values():
            random.shuffle(v)

        # Shuffle the list of products associated with each category.
        for v in self.cat2r_mapping_data.values():
            random.shuffle(v)

        # --------------------------------------------------------------------------------------------------------------
        # Build the "product (R) => review (S)" training data.
        # --------------------------------------------------------------------------------------------------------------
        # Select random products which will be used for "product (R) => review (S)" training.
        products = random.sample(list(self.r2s_mapping_data.keys()), num_rs)

        # Select a random review associated with the product as positive document (d+).
        p_id = [random.sample(self.r2s_mapping_data[v], self.num_positives) for v in products]

        # Find the categories related to the chosen product.
        p_category = [self.r2cat_mapping_data[v] for v in products]

        # Find self.num_negatives different products sharing the same categories. The associated reviews will serve
        # as negative documents (d-). Use close categories if not enough elements can be found.
        n_products = [[v2 for cat in [x2 for x1 in [(v1 - i, v1 + i) for i in range(1, 100)] for x2 in x1]
                       for v2 in self.cat2r_mapping_data.get(cat, []) if v2 != k1]
                      for k1, v1 in zip(products, p_category)]
        del p_category
        n_products = [[v2 for v2 in v1 if v2 in self.r2s_mapping_data.keys()][:self.num_negatives]
                      for v1 in n_products]

        # Select a random review associated with every product selected as negative (d-).
        n_id = [[random.choice(self.r2s_mapping_data[v2]) for v2 in v1] for v1 in n_products]
        del n_products

        rs_train_data = {k1: {
            "p_id": [(k2, i2, i2) for i2, k2 in enumerate(p_id[i1])],
            "n_id": [(k2, i2, i2) for i2, k2 in enumerate(n_id[i1])],
            "hn_id": []
        } for i1, k1 in enumerate(products)}
        del products, p_id, n_id

        print(f"Start length of RS train data: {len(rs_train_data)} / {num_rs}.", flush=True)

        rs_train_data = {k: v for k, v in rs_train_data.items()
                         if len(v["p_id"]) == self.num_positives and len(v["n_id"]) == self.num_negatives}

        print(f"Length of RS train data: {len(rs_train_data)} / {num_rs}.", flush=True)

        # --------------------------------------------------------------------------------------------------------------
        # Shuffle the lists in the mapping data.
        # --------------------------------------------------------------------------------------------------------------
        # Shuffle the list of reviews associated with each product.
        for v in self.r2s_mapping_data.values():
            random.shuffle(v)

        # Shuffle the list of products associated with each category.
        for v in self.cat2r_mapping_data.values():
            random.shuffle(v)

        # --------------------------------------------------------------------------------------------------------------
        # Build the "review (S) => product (R)" training data.
        # --------------------------------------------------------------------------------------------------------------
        # Select random products which will be used for "review (S) => product (R)" training.
        p_id = random.sample(list(self.r2s_mapping_data.keys()), num_sr)

        # Select a random review associated with the product as positive document (d+).
        reviews = [random.choice(self.r2s_mapping_data[v]) for v in p_id]

        # Find the categories related to the chosen product.
        p_category = [self.r2cat_mapping_data[v] for v in p_id]

        # Find self.num_negatives different products sharing the same categories. The associated reviews will serve
        # as negative documents (d-). Use close categories if not enough elements can be found.
        n_id = [[v2 for cat in [x2 for x1 in [(v1 - i, v1 + i) for i in range(1, 100)] for x2 in x1]
                 for v2 in self.cat2r_mapping_data.get(cat, []) if v2 != k1][:self.num_negatives]
                for k1, v1 in zip(p_id, p_category)]
        del p_category

        sr_train_data = {k1: {
            "p_id": [(p_id[i1], 0, 0)],
            "n_id": [(k2, i2, i2) for i2, k2 in enumerate(n_id[i1])],
            "hn_id": []
        } for i1, k1 in enumerate(reviews)}
        del reviews, p_id, n_id

        print(f"Start length of SR train data: {len(sr_train_data)} / {num_sr}.", flush=True)

        # NOTE: this code will not work in self.num_positives > 1!
        # TODO: allow for batches with a single positive, for this case.
        sr_train_data = {k: v for k, v in sr_train_data.items()
                         if len(v["p_id"]) == self.num_positives and len(v["n_id"]) == self.num_negatives}

        print(f"Length of SR train data: {len(sr_train_data)} / {num_sr}.", flush=True)

        # --------------------------------------------------------------------------------------------------------------
        # Return the final training data.
        # --------------------------------------------------------------------------------------------------------------
        return rs_train_data, sr_train_data


    # Ensure to have batches full of queries/documents for the same task.
    # If all the in-batch negatives are of the same type (products/reviews), the learning should be more effective.
    def interleave_train_keys(self) -> list[int | str]:
        # Interleave R and S keys (query_id) in blocks equal to the batch size.
        rs_keys = []
        rs_remaining = []

        len_rs = max(len(self.r_keys), len(self.s_keys), len(self.rs_keys), len(self.sr_keys))
        for idx in range(0, len_rs, self.batch_size):
            for keys, task in zip((self.r_keys, self.s_keys, self.rs_keys, self.sr_keys),
                                  ("r", "s", "rs", "sr")):
                data = keys[idx: idx + self.batch_size]
                if len(data) == self.batch_size:
                    rs_keys.extend([(task, k) for k in data])
                elif len(data) <= 0:
                    pass
                else:
                    rs_remaining.extend([(task, k) for k in data])
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

        # Return the final list obtained. Append the remaining queries at the end of the train keys.
        return rs_keys + rs_remaining


    def update_train_data(self, task: str, run_data: dict[int | str, list[int | str]]) -> None:
        if task is None or not isinstance(task, str) or task not in {"r", "s"}:
            raise ValueError(f"The given task is not supported: {task}.")

        # --------------------------------------------------------------------------------------------------------------
        # Update the hard negatives, using the run data.
        # --------------------------------------------------------------------------------------------------------------
        not_found = max(len(self.r_corpus_data), len(self.s_corpus_data))

        for query_id in run_data.keys():
            if task == "r":
                train_data = self.r_train_data
                exclude_data = self.r_exclude_data
            elif task == "s":
                train_data = self.s_train_data
                exclude_data = self.s_exclude_data
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
        # Select the "products (R) => reviews (S)" and "reviews (S) => products (R)" training data.
        self.rs_train_data, self.sr_train_data = self.build_rs_sr_train_data(len(self.rs_keys), len(self.sr_keys))
        self.rs_keys = list(self.rs_train_data.keys())
        self.sr_keys = list(self.sr_train_data.keys())

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
                q_data = self.r_corpus_data[query_id]
                c_data = self.s_corpus_data
                t_data = self.rs_train_data[query_id]
            elif task == "sr":
                q_data = self.s_corpus_data[query_id]
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
                if task == "r":
                    epoch_idx = (self.epoch - self.num_warmup_epochs) % self.every_k_epochs
                    positives_id = [positives_id[len(positives_id) - 1 - \
                                                 ((epoch_idx * self.num_positives + i) % len(positives_id))][0]
                                    for i in range(len(positives_id))][:self.num_positives]
                    del epoch_idx
                else:
                    positives_id = random.sample(positives_id, self.num_positives)
                    positives_id = [v[0] for v in positives_id]
            assert len(positives_id) == self.num_positives

            # Select the negative document(s).
            negatives_id = sorted(t_data["n_id"] + t_data["hn_id"],
                                  key=operator.itemgetter(1, 2), reverse=False)
            if self.epoch < self.num_warmup_epochs:
                negatives_id = [negatives_id[(self.epoch * self.num_negatives + i) % len(negatives_id)][0]
                                for i in range(len(negatives_id))][:self.num_negatives]
            else:
                if task == "r":
                    epoch_idx = (self.epoch - self.num_warmup_epochs) % self.every_k_epochs
                    negatives_id = [negatives_id[(epoch_idx * self.num_negatives + i) % len(negatives_id)][0]
                                    for i in range(len(negatives_id))][:self.num_negatives]
                    del epoch_idx
                else:
                    negatives_id = random.sample(negatives_id, self.num_negatives)
                    negatives_id = [v[0] for v in negatives_id]
            assert len(negatives_id) == self.num_negatives

            # Add the current output data to the result.
            result.append(
                {
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

        return \
            {
                "query_text": [b["query_text"] for b in result]
            } | {
                f"positive{i}_text": [b[f"positive{i}_text"] for b in result]
                for i in range(self.num_positives)
            } | {
                f"negative{i}_text": [b[f"negative{i}_text"] for b in result]
                for i in range(self.num_negatives)
            } | {
                "query_aspects": [b["query_aspects"] for b in result],
            } | {
                f"positive{i}_aspects": [b[f"positive{i}_aspects"] for b in result]
                for i in range(self.num_positives)
            } | {
                f"negative{i}_aspects": [b[f"negative{i}_aspects"] for b in result]
                for i in range(self.num_negatives)
            }
