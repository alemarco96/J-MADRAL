import datasets
import io
import numpy
import operator
import random
import torch
import transformers
import typing


class BiEncoderTrainingTriplesDataset(datasets.Dataset, transformers.TrainerCallback):
    def __init__(self,
                 corpus_data: dict[int | str, dict[str, str | list[int]]],
                 triples_data: dict[int | str, dict[str, str | list[int] | dict[int | str, int]]],
                 num_warmup_epochs: int = 4,
                 every_k_epochs: int = 4,
                 min_rank_for_hardnegatives: int = 49,
                 num_positives: int = 1,
                 num_negatives: int = 1,
                 num_retain_hardnegatives: int = 1):
        self.corpus_data = corpus_data
        self.queries_data = {k: {"text": v["query_text"], "aspects": v["query_aspects"]}
                                for k, v in triples_data.items()}
        self.triples_data = {k: {"positives_id": sorted(v["positives_id"].items(),
                                                        key=operator.itemgetter(1), reverse=False),
                                 "negatives_id": sorted(v["negatives_id"].items(),
                                                        key=operator.itemgetter(1), reverse=False),
                                 "hardnegatives_id": sorted(v["hardnegatives_id"].items(),
                                                            key=operator.itemgetter(1), reverse=False)[:num_retain_hardnegatives]
                                 } for k, v in triples_data.items()}
        self.triples_data = {k: v for k, v in self.triples_data.items()
                             if len(v["positives_id"]) >= num_positives and
                             (len(v["negatives_id"]) + len(v["hardnegatives_id"])) >= num_negatives}
        self.triples_keys = sorted(self.triples_data.keys())

        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.num_retain_hardnegatives = num_retain_hardnegatives

        self.min_rank_for_hardnegatives = min_rank_for_hardnegatives
        self.sub_k_epochs = 5
        self.min_min_rank_for_hardnegatives = 29

        self.epoch = 0
        self.num_warmup_epochs = num_warmup_epochs
        self.every_k_epochs = every_k_epochs

    def update_triples(self,
                       run_data: dict[int | str, list[int | str]],
                       queries_keys: dict[int | str, int],
                       queries_embedding: torch.Tensor,
                       corpus_keys: dict[int | str, int],
                       index_filename: str,
                       index_header_size: int = 45
                       ) -> None:
        # --------------------------------------------------------------------------------------------------------------
        # Update the hard negatives, using the run data.
        # --------------------------------------------------------------------------------------------------------------
        for query_id in run_data.keys() & self.triples_data.keys():
            query_triple = self.triples_data[query_id]
            eval_id = set([k for k, _ in query_triple.get("positives_id", [])]) | \
                      set([k for k, _ in query_triple.get("negatives_id", [])])
            del query_triple

            num_k_epochs = (self.epoch - self.num_warmup_epochs) // self.every_k_epochs
            min_rank_hn = max(self.min_rank_for_hardnegatives - self.sub_k_epochs * num_k_epochs,
                              self.min_min_rank_for_hardnegatives)
            del num_k_epochs

            # if self.epoch <= self.num_warmup_epochs:
            #     min_rank_hn = self.min_rank_for_hardnegatives  # 4 => 49
            # elif self.epoch <= self.num_warmup_epochs + self.every_k_epochs:
            #     min_rank_hn = self.min_rank_for_hardnegatives  # 8 => 49
            # elif self.epoch <= self.num_warmup_epochs + 2 * self.every_k_epochs:
            #     min_rank_hn = max(self.min_rank_for_hardnegatives - 10, 0)  # 12 => 39
            # elif self.epoch <= self.num_warmup_epochs + 3 * self.every_k_epochs:
            #     min_rank_hn = max(self.min_rank_for_hardnegatives - 20, 0)  # 16 => 29
            # else:
            #     min_rank_hn = max(self.min_rank_for_hardnegatives - 30, 0)  # 20 => 19

            hardnegatives_id = [(k, i) for i, k in enumerate(run_data[query_id][min_rank_hn:],
                                                             # start=self.min_rank_for_hardnegatives)
                                                             start=min_rank_hn)
                                if k not in eval_id][:self.num_retain_hardnegatives]
            del min_rank_hn
            del eval_id
            if len(hardnegatives_id) > 0:
                self.triples_data[query_id]["hardnegatives_id"] = hardnegatives_id

            del hardnegatives_id
        try:
            del query_id
        except UnboundLocalError:
            pass

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
            for query_id, query_triple in self.triples_data.items():
                query_embedding = queries_embedding[queries_keys[query_id]].view(1, -1).to(dtype=torch.float32)

                # Compute the score for each positive document.
                positives_idx = [(k, corpus_keys[k]) for k, _ in query_triple["positives_id"]]
                if len(positives_idx) > 0:
                    positives_score = torch.stack([_get_doc_embedding(fi,
                                                                      index_header_size + v * emb_bytes,
                                                                      query_embedding.shape[-1])
                                                   for _, v in positives_idx], dim=0)
                    positives_score = torch.matmul(query_embedding, positives_score.transpose(0, 1)).view(-1)
                    positives_score = sorted([(k1[0], -v1) for k1, v1 in zip(positives_idx, positives_score.tolist())],
                                             key=operator.itemgetter(1), reverse=True)
                else:
                    positives_score = []
                del positives_idx

                # Compute the score for each negative document.
                negatives_idx = [(k, corpus_keys[k]) for k, _ in query_triple["negatives_id"]]
                if len(negatives_idx) > 0:
                    negatives_score = torch.stack([_get_doc_embedding(fi,
                                                                      index_header_size + v * emb_bytes,
                                                                      query_embedding.shape[-1])
                                                   for _, v in negatives_idx], dim=0)
                    negatives_score = torch.matmul(query_embedding, negatives_score.transpose(0, 1)).view(-1)
                    negatives_score = sorted([(k1[0], -v1) for k1, v1 in zip(negatives_idx, negatives_score.tolist())],
                                             key=operator.itemgetter(1), reverse=True)
                else:
                    negatives_score = []
                del negatives_idx

                # Compute the score for each hard negative document.
                hardnegatives_idx = [(k, corpus_keys[k]) for k, _ in query_triple["hardnegatives_id"]]
                if len(hardnegatives_idx) > 0:
                    hardnegatives_score = torch.stack([_get_doc_embedding(fi,
                                                                          index_header_size + v * emb_bytes,
                                                                          query_embedding.shape[-1])
                                                       for _, v in hardnegatives_idx], dim=0)
                    hardnegatives_score = torch.matmul(query_embedding,
                                                       hardnegatives_score.transpose(0, 1)).view(-1)
                    hardnegatives_score = sorted([(k1[0], -v1) for k1, v1 in zip(hardnegatives_idx,
                                                                                 hardnegatives_score.tolist())],
                                                 key=operator.itemgetter(1), reverse=True)
                else:
                    hardnegatives_score = []
                del hardnegatives_idx

                self.triples_data[query_id] = {
                    "positives_id": positives_score,
                    "negatives_id": negatives_score,
                    "hardnegatives_id": hardnegatives_score
                }
                del positives_score, negatives_score, hardnegatives_score, query_embedding
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
        assert state.epoch is not None
        epoch = int(state.epoch + 1e-4)
        print(f"** EPOCH BEGIN: {epoch} **", flush=True)

        random.shuffle(self.triples_keys)

    def on_epoch_end(self,
                     args: transformers.TrainingArguments,
                     state: transformers.TrainerState,
                     control: transformers.TrainerControl,
                     **kwargs):
        assert state.epoch is not None
        epoch = int(state.epoch + 1e-4)
        print(f"** EPOCH END: {epoch} **", flush=True)

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
        random.shuffle(self.triples_keys)
        return self

    def take(self, n: int) -> datasets.Dataset:
        self.triples_keys = self.triples_keys[:n]
        return self

    def __len__(self):
        return len(self.triples_keys)

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
            query_id = self.triples_keys[i1]
            query_triple = self.triples_data[query_id]

            # Select the positive document(s).
            positives_id = query_triple["positives_id"]
            if self.epoch < self.num_warmup_epochs:  # + self.every_k_epochs:
                positives_id = [positives_id[(self.epoch * self.num_positives + i) % len(positives_id)][0]
                                for i in range(len(positives_id))][:self.num_positives]
                # positives_id = [k for k, _ in positives_id[:self.num_positives]]
            else:
                epoch_idx = (self.epoch - self.num_warmup_epochs) % self.every_k_epochs
                positives_id = [positives_id[len(positives_id) - 1 - \
                                             ((epoch_idx * self.num_positives + i) % len(positives_id))][0]
                                for i in range(len(positives_id))][:self.num_positives]
                # positives_id = [k for k, _ in positives_id[::-1][:self.num_positives]]
                del epoch_idx
            assert len(positives_id) == self.num_positives

            # Select the negative document(s).
            negatives_id = sorted(query_triple["negatives_id"] + query_triple["hardnegatives_id"],
                                  key=operator.itemgetter(1), reverse=False)
            if self.epoch < self.num_warmup_epochs:
                negatives_id = [negatives_id[(self.epoch * self.num_negatives + i) % len(negatives_id)][0]
                                for i in range(len(negatives_id))][:self.num_negatives]
                # negatives_id = [k for k, _ in negatives_id[:self.num_negatives]]
            else:
                epoch_idx = (self.epoch - self.num_warmup_epochs) % self.every_k_epochs
                negatives_id = [negatives_id[(epoch_idx * self.num_negatives + i) % len(negatives_id)][0]
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
