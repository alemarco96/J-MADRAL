import datasets
import io
import numpy
import operator
import random
import torch
import transformers
import typing


class BiEncoderEVOTrainingTriplesDataset(datasets.Dataset, transformers.TrainerCallback):
    def __init__(self,
                 corpus_data: dict[int | str, dict[str, str | list[int]]],
                 triples_data: dict[int | str, dict[str, str | list[int] | tuple[int | str, int]]],
                 num_warmup_epochs: int = 4,
                 every_k_epochs: int = 4,
                 max_rank_for_exclude: int = 50,
                 num_positives: int = 1,
                 num_negatives: int = 1):
        self.corpus_data = corpus_data
        self.queries_data = {k: {"text": v["query_text"], "aspects": v["query_aspects"]}
                                for k, v in triples_data.items()}

        # Select the training data.
        # noinspection PyTypeChecker
        self.train_data = {k1: {
            "exclude_id": {k2 for k2 in v1["exclude_id"]},
            "p_id": [(k2, v2) for k2, v2 in v1["positives_rank"]],
            "n_id": [(k2, v2) for k2, v2 in v1["negatives_rank"]],
            "hn_id": [(k2, v2) for k2, v2 in v1["hardnegatives_rank"][:(num_warmup_epochs * num_negatives)]]
        } for k1, v1 in triples_data.items()}
        self.train_data = {k: v for k, v in self.train_data.items()
                           if len(v["p_id"]) >= num_positives and len(v["n_id"]) + len(v["hn_id"]) >= num_negatives}
        self.train_keys = sorted(self.train_data.keys())

        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.num_retain_hardnegatives = every_k_epochs * num_negatives

        self.max_rank_for_exclude = max_rank_for_exclude
        self.sub_k_epochs = 5
        self.min_max_rank_for_exclude = 30

        self.epoch = 0
        self.num_warmup_epochs = num_warmup_epochs
        self.every_k_epochs = every_k_epochs

    def update_triples(self,
                       run_data: dict[int | str, list[int | str]],
                       scores_data: dict[int | str, dict[int | str, float]],
                       queries_keys: dict[int | str, int],
                       queries_embedding: torch.Tensor,
                       corpus_keys: dict[int | str, int],
                       index_filename: str,
                       index_header_size: int = 45
                       ) -> None:
        # --------------------------------------------------------------------------------------------------------------
        # Update the hard negatives, using the run data.
        # --------------------------------------------------------------------------------------------------------------
        num_k_epochs = (self.epoch - self.num_warmup_epochs) // self.every_k_epochs
        max_rank_exclude = max(self.max_rank_for_exclude - self.sub_k_epochs * num_k_epochs,
                               self.min_max_rank_for_exclude)
        # max_rank_exclude = self.max_rank_for_exclude
        del num_k_epochs

        # noinspection PyTypeChecker
        exclude_id = {k1: set.union(*[
            set(v1["exclude_id"]),
            {k2 for k2, _ in sorted(scores_data.get(k1, {}).items(),
                                    key=operator.itemgetter(1), reverse=True)[:max_rank_exclude]}
        ]) for k1, v1 in self.train_data.items()}
        del max_rank_exclude

        self.train_data.update({query_id: {
            "exclude_id": self.train_data[query_id]["exclude_id"],
            "p_id": self.train_data[query_id]["p_id"],
            "n_id": self.train_data[query_id]["n_id"],
            "hn_id": [(k, 0) for k in run_data[query_id]
                      if k not in exclude_id[query_id]][:self.num_retain_hardnegatives]
        } for query_id in run_data.keys() & self.train_data.keys()})
        del exclude_id

        # --------------------------------------------------------------------------------------------------------------
        # Update the positives and negatives, using the queries and documents embeddings.
        # --------------------------------------------------------------------------------------------------------------
        def _get_doc_embedding(_file, _idx: int, _hidden_size: int) -> torch.Tensor:
            _file.seek(_idx, io.SEEK_SET)
            _embedding = _file.read(_hidden_size * 4)
            _embedding = numpy.frombuffer(_embedding, dtype=numpy.float32)
            _embedding = torch.from_numpy(_embedding).to(dtype=torch.float32).view(-1)
            return _embedding
        emb_bytes = queries_embedding.shape[-1] * 4

        with open(index_filename, "rb") as fi:
            for query_id, query_triple in self.train_data.items():
                query_embedding = queries_embedding[queries_keys[query_id]].view(1, -1).to(dtype=torch.float32)

                # Compute the score for each positive document.
                p_id = [(k, corpus_keys[k]) for k, _ in query_triple["p_id"]]
                if len(p_id) > 0:
                    p_score = torch.stack([_get_doc_embedding(fi,
                                                              index_header_size + v * emb_bytes,
                                                              query_embedding.shape[-1])
                                           for _, v in p_id], dim=0)
                    p_score = torch.matmul(query_embedding, p_score.transpose(0, 1)).view(-1)
                    p_score = sorted([(k1[0], -v1) for k1, v1 in zip(p_id, p_score.tolist())],
                                     key=operator.itemgetter(1), reverse=True)
                else:
                    p_score = []
                del p_id

                # Compute the score for each negative document.
                n_id = [(k, corpus_keys[k]) for k, _ in query_triple["n_id"]]
                if len(n_id) > 0:
                    n_score = torch.stack([_get_doc_embedding(fi,
                                                              index_header_size + v * emb_bytes,
                                                              query_embedding.shape[-1])
                                           for _, v in n_id], dim=0)
                    n_score = torch.matmul(query_embedding, n_score.transpose(0, 1)).view(-1)
                    n_score = sorted([(k1[0], -v1) for k1, v1 in zip(n_id, n_score.tolist())],
                                     key=operator.itemgetter(1), reverse=True)
                else:
                    n_score = []
                del n_id

                # Compute the score for each hard negative document.
                hn_id = [(k, corpus_keys[k]) for k, _ in query_triple["hn_id"]]
                if len(hn_id) > 0:
                    hn_score = torch.stack([_get_doc_embedding(fi,
                                                               index_header_size + v * emb_bytes,
                                                               query_embedding.shape[-1])
                                            for _, v in hn_id], dim=0)
                    hn_score = torch.matmul(query_embedding, hn_score.transpose(0, 1)).view(-1)
                    hn_score = sorted([(k1[0], -v1) for k1, v1 in zip(hn_id, hn_score.tolist())],
                                      key=operator.itemgetter(1), reverse=True)
                else:
                    hn_score = []
                del hn_id

                self.train_data[query_id] = {
                    "exclude_id": self.train_data[query_id]["exclude_id"],
                    "p_id": p_score,
                    "n_id": n_score,
                    "hn_id": hn_score
                }
                del p_score, n_score, hn_score, query_embedding
            try:
                del query_id, query_triple
            except UnboundLocalError:
                pass
        del fi, _get_doc_embedding, emb_bytes

    def on_epoch_begin(self,
                       args: transformers.TrainingArguments,
                       state: transformers.TrainerState,
                       control: transformers.TrainerControl,
                       **kwargs):
        # assert state.epoch is not None
        # epoch = int(state.epoch + 1e-4)
        # print(f"** EPOCH BEGIN: {epoch} **", flush=True)

        random.shuffle(self.train_keys)

    def on_epoch_end(self,
                     args: transformers.TrainingArguments,
                     state: transformers.TrainerState,
                     control: transformers.TrainerControl,
                     **kwargs):
        # assert state.epoch is not None
        # epoch = int(state.epoch + 1e-4)
        # print(f"** EPOCH END: {epoch} **", flush=True)

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
        random.shuffle(self.train_keys)
        return self

    def take(self, n: int) -> datasets.Dataset:
        self.train_keys = self.train_keys[:n]
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
            query_id = self.train_keys[i1]
            query_triple = self.train_data[query_id]

            # Select the positive document(s).
            positives_id = [k for k, _ in query_triple["p_id"]]
            if self.epoch < self.num_warmup_epochs:  # + self.every_k_epochs:
                positives_id = [positives_id[(self.epoch * self.num_positives + i) % len(positives_id)]
                                for i in range(len(positives_id))][:self.num_positives]
                # positives_id = [k for k, _ in positives_id[:self.num_positives]]
            else:
                epoch_idx = (self.epoch - self.num_warmup_epochs) % self.every_k_epochs
                positives_id = [positives_id[len(positives_id) - 1 - \
                                             ((epoch_idx * self.num_positives + i) % len(positives_id))]
                                for i in range(len(positives_id))][:self.num_positives]
                # positives_id = [k for k, _ in positives_id[::-1][:self.num_positives]]
                del epoch_idx
            assert len(positives_id) == self.num_positives

            # Select the negative document(s).
            negatives_id = sorted(query_triple["n_id"] + query_triple["hn_id"], key=operator.itemgetter(1))
            negatives_id = [k for k, _ in negatives_id]
            if self.epoch < self.num_warmup_epochs:
                negatives_id = [negatives_id[(self.epoch * self.num_negatives + i) % len(negatives_id)]
                                for i in range(len(negatives_id))][:self.num_negatives]
                # negatives_id = [k for k, _ in negatives_id[:self.num_negatives]]
            else:
                epoch_idx = (self.epoch - self.num_warmup_epochs) % self.every_k_epochs
                negatives_id = [negatives_id[(epoch_idx * self.num_negatives + i) % len(negatives_id)]
                                for i in range(len(negatives_id))][:self.num_negatives]
                # negatives_id = [k for k, _ in negatives_id[:self.num_negatives]]
                del epoch_idx
            assert len(negatives_id) == self.num_negatives

            # Add the current output data to the result.
            result.append(
                {
                    "query_text": self.queries_data[query_id]["text"]
                } |
                {
                    f"positive{i}_text": self.corpus_data[doc_id]["text"]
                    for i, doc_id in enumerate(positives_id)
                } |
                {
                    f"negative{i}_text": self.corpus_data[doc_id]["text"]
                    for i, doc_id in enumerate(negatives_id)
                } |
                {
                    "query_aspects": self.queries_data[query_id]["aspects"]
                } |
                {
                    f"positive{i}_aspects": self.corpus_data[doc_id]["aspects"]
                    for i, doc_id in enumerate(positives_id)
                } |
                {
                    f"negative{i}_aspects": self.corpus_data[doc_id]["aspects"]
                    for i, doc_id in enumerate(negatives_id)
                }
            )
            del query_triple, positives_id, negatives_id
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
