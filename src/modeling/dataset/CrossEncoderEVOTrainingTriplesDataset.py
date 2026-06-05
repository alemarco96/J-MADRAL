import datasets
import numpy
import operator
import random
import torch
import transformers
import typing


class CrossEncoderEVOTrainingTriplesDataset(datasets.Dataset, transformers.TrainerCallback):
    def __init__(self,
                 corpus_data: dict[int | str, dict[str, str | list[int]]],
                 triples_data: dict[int | str, dict[str, str | list[int] | tuple[int | str, int, int]]],
                 num_warmup_epochs: int = 4,
                 every_k_epochs: int = 4,
                 exact_label: float = 1.0,
                 substitute_label: float = 0.2,
                 complementary_label: float = 0.05,
                 irrelevant_label: float = 0.01,
                 softpositive_label: float = 1.0,
                 hardnegative_label: float = 0.01,
                 random_label: float = 0.0,
                 num_positives: int = 1,
                 num_judgements: int = 2,
                 num_hard_negatives: int = 4,
                 num_random_negatives: int = 1
                ):
        # Memorize the corpus and queries dataset.
        self.corpus_data = corpus_data
        self.corpus_keys = sorted(self.corpus_data.keys())

        self.queries_data = {k: {"text": v["query_text"], "aspects": v["query_aspects"]}
                             for k, v in triples_data.items()}

        # Select the training data.
        # noinspection PyTypeChecker
        self.train_data = {k1: {
            "exclude_id": set(v1["exclude_id"]),
            "e_id": [k2 for k2, _, _ in v1["positives_rank"] if k2 in set(v1["e_id"])],
            "s_id": [k2 for k2, _, _ in v1["negatives_rank"] if k2 in set(v1["s_id"])],
            "c_id": [k2 for k2, _, _ in v1["negatives_rank"] if k2 in set(v1["c_id"])],
            "i_id": [k2 for k2, _, _ in v1["negatives_rank"] if k2 in set(v1["i_id"])],
            "sp_id": [k2 for k2, _, _ in v1["softpositives_rank"]],
            "hn_id": [k2 for k2, _, _ in v1["hardnegatives_rank"]]
        } for k1, v1 in triples_data.items()}
        self.train_data = {k: v for k, v in self.train_data.items() if len(v["e_id"]) + len(v["sp_id"]) >= 1 and
                           (len(v["s_id"]) + len(v["c_id"])+ len(v["i_id"]) + len(v["hn_id"])) >= 1}
        self.train_keys = sorted(self.train_data.keys())

        self.num_warmup_epochs = num_warmup_epochs
        self.every_k_epochs = every_k_epochs

        self.exact_label = exact_label
        self.substitute_label = substitute_label
        self.complementary_label = complementary_label
        self.irrelevant_label = irrelevant_label
        self.softpositive_label = softpositive_label
        self.hardnegative_label = hardnegative_label
        self.random_label = random_label

        self.num_positives = num_positives
        self.num_judgements = num_judgements
        self.num_hard_negatives = num_hard_negatives
        self.num_random_negatives = num_random_negatives

        self.epoch = 0

    def update_triples(self,
                       candidates_data: dict[int | str, list[tuple[int | str, float]]]
                       # candidates_data: dict[int | str, dict[int | str, float]]
                       ) -> None:
        self.train_data.update({k1: {
            "exclude_id": set(self.train_data[k1]["exclude_id"]),
            "e_id": [k2 for k2, _ in v1 if k2 in set(self.train_data[k1]["e_id"])],
            "s_id": [k2 for k2, _ in v1 if k2 in set(self.train_data[k1]["s_id"])],
            "c_id": [k2 for k2, _ in v1 if k2 in set(self.train_data[k1]["c_id"])],
            "i_id": [k2 for k2, _ in v1 if k2 in set(self.train_data[k1]["i_id"])],
            "sp_id": [k2 for k2, _ in v1 if k2 in set(self.train_data[k1]["sp_id"])],
            "hn_id": [k2 for k2, _ in v1 if k2 in set(self.train_data[k1]["hn_id"])]
            # "e_id": [k3 for k3, _ in sorted([(k2, v1[k2]) for k2 in self.train_data[k1]["e_id"]],
            #                                 key=operator.itemgetter(1), reverse=True)],
            # "s_id": [k3 for k3, _ in sorted([(k2, v1[k2]) for k2 in self.train_data[k1]["s_id"]],
            #                                 key=operator.itemgetter(1), reverse=True)],
            # "c_id": [k3 for k3, _ in sorted([(k2, v1[k2]) for k2 in self.train_data[k1]["c_id"]],
            #                                 key=operator.itemgetter(1), reverse=True)],
            # "i_id": [k3 for k3, _ in sorted([(k2, v1[k2]) for k2 in self.train_data[k1]["i_id"]],
            #                                 key=operator.itemgetter(1), reverse=True)],
            # "sp_id": [k3 for k3, _ in sorted([(k2, v1[k2]) for k2 in self.train_data[k1]["sp_id"]],
            #                                  key=operator.itemgetter(1), reverse=True)],
            # "hn_id": [k3 for k3, _ in sorted([(k2, v1[k2]) for k2 in self.train_data[k1]["hn_id"]],
            #                                  key=operator.itemgetter(1), reverse=True)]
        } for k1, v1 in candidates_data.items()})

    def on_epoch_begin(self,
                       args: transformers.TrainingArguments,
                       state: transformers.TrainerState,
                       control: transformers.TrainerControl,
                       **kwargs) -> None:
        assert state.epoch is not None
        epoch = int(state.epoch + 1e-4)
        print(f"** EPOCH BEGIN: {epoch} **", flush=True)

        random.shuffle(self.train_keys)

    def on_epoch_end(self,
                     args: transformers.TrainingArguments,
                     state: transformers.TrainerState,
                     control: transformers.TrainerControl,
                     **kwargs) -> None:
        assert state.epoch is not None
        epoch = int(state.epoch + 1e-4)
        print(f"** EPOCH END: {epoch} **", flush=True)

        self.epoch += 1

    def take(self, n: int) -> datasets.Dataset:
        self.train_keys = sorted(self.train_data.keys())[:n]
        return self

    def shuffle(self,
                seed: typing.Optional[int] = None,
                generator: typing.Optional[numpy.random.Generator] = None,
                keep_in_memory: bool = False,
                load_from_cache_file: typing.Optional[bool] = None,
                indices_cache_file_name: typing.Optional[str] = None,
                writer_batch_size: typing.Optional[int] = 1000,
                new_fingerprint: typing.Optional[str] = None
                ) -> datasets.Dataset:
        random.shuffle(self.train_keys)
        return self

    def __len__(self) -> int:
        return len(self.train_keys)

    def __getitem__(self, idx) -> dict[str, typing.Any]:
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
            query_id = self.train_keys[i1]
            query_triple = self.train_data[query_id]

            # Select the positive document(s).
            poss_id = [(k, self.exact_label) for k in query_triple["e_id"]] + \
                      [(k, self.softpositive_label) for k in query_triple["sp_id"]]
            if len(poss_id) > 0:
                poss_id = random.sample(poss_id, min(len(poss_id), self.num_positives))
                # if self.epoch < self.num_warmup_epochs:  # + self.every_k_epochs:
                #     poss_id = [poss_id[(self.epoch * self.num_positives + i) % len(poss_id)]
                #                for i in range(len(poss_id))][:self.num_positives]
                # else:
                #     epoch_idx = (self.epoch - self.num_warmup_epochs) % self.every_k_epochs
                #     poss_id = [poss_id[len(poss_id) - 1 - ((epoch_idx * self.num_positives + i) % len(poss_id))]
                #                for i in range(len(poss_id))][:self.num_positives]
                #     del epoch_idx
            else:
                poss_id = []
            num_judg = self.num_judgements + (self.num_positives - len(poss_id))

            # Select the judged document(s).
            judg_id = [(k, self.substitute_label) for k in query_triple["s_id"]] + \
                      [(k, self.complementary_label) for k in query_triple["c_id"]] + \
                      [(k, self.irrelevant_label) for k in query_triple["i_id"]]
            if len(judg_id) > 0:
                judg_id = random.sample(judg_id, min(len(judg_id), num_judg))
                # if self.epoch < self.num_warmup_epochs:
                #     judg_id = [judg_id[(self.epoch * num_judg + i) % len(judg_id)]
                #                for i in range(len(judg_id))][:num_judg]
                # else:
                #     epoch_idx = (self.epoch - self.num_warmup_epochs) % self.every_k_epochs
                #     judg_id = [judg_id[(epoch_idx * num_judg + i) % len(judg_id)]
                #                for i in range(len(judg_id))][:num_judg]
                #     del epoch_idx
            else:
                judg_id = []
            num_hard = self.num_hard_negatives + (self.num_judgements - len(judg_id)) + \
                       (self.num_positives - len(poss_id))

            # Select the hard negative document(s).
            hard_id = [(k, self.hardnegative_label) for k in query_triple["hn_id"]]
            if len(hard_id) > 0:
                hard_id = random.sample(hard_id, min(len(hard_id), num_hard))
                # if self.epoch < self.num_warmup_epochs:
                #     hard_id = [hard_id[(self.epoch * num_hard + i) % len(hard_id)]
                #                for i in range(len(hard_id))][:num_hard]
                # else:
                #     epoch_idx = (self.epoch - self.num_warmup_epochs) % self.every_k_epochs
                #     hard_id = [hard_id[(epoch_idx * num_hard + i) % len(hard_id)]
                #                for i in range(len(hard_id))][:num_hard]
                #     del epoch_idx
            else:
                hard_id = []
            num_rand = self.num_random_negatives + (self.num_hard_negatives - len(hard_id)) + \
                       (self.num_judgements - len(judg_id)) + (self.num_positives - len(poss_id))

            # Sample the random negative document(s).
            exclude_id = query_triple["exclude_id"]
            rand_id = random.sample(self.corpus_keys, num_rand + len(exclude_id))
            rand_id = [(k, self.random_label) for k in rand_id if k not in exclude_id][:num_rand]
            del query_triple, exclude_id, num_judg, num_hard, num_rand

            # Aggregate all documents id and their associated relevance label.
            train_data = poss_id + judg_id + hard_id + rand_id
            random.shuffle(train_data)
            del poss_id, judg_id, hard_id, rand_id

            # Add the current output data to the result.
            result.extend([{
                "query_text": self.queries_data[query_id]["text"],
                "document_text": self.corpus_data[doc_id]["text"],
                "query_aspects": self.queries_data[query_id]["aspects"],
                "document_aspects": self.corpus_data[doc_id]["aspects"],
                "qd_label": doc_label
            } for doc_id, doc_label in train_data])
            del train_data
        try:
            del i1
        except UnboundLocalError:
            pass

        random.shuffle(result)
        return \
            {
                "query_text": [b["query_text"] for b in result],
                "document_text": [b["document_text"] for b in result],
                "query_aspects": [b["query_aspects"] for b in result],
                "document_aspects": [b["document_aspects"] for b in result],
                "qd_label": [b["qd_label"] for b in result]
            }
