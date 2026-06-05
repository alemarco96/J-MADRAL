import datasets
import numpy
import operator
import random
import torch
import transformers
import typing


class MultiPretrainDataset(datasets.Dataset, transformers.TrainerCallback):
    # noinspection PyMissingConstructor
    def __init__(self,
                 corpora_data: list[dict[int | str, dict[str, str | list[int]]]],
                 corpora_samples_per_epoch: list[int],
                 input_text_field: str = "text",
                 input_aspects_field: str = "aspects",
                 ):
        # Verify that the given data is of the correct type.
        assert corpora_data is not None
        assert isinstance(corpora_data, list)

        assert corpora_samples_per_epoch is not None
        assert isinstance(corpora_samples_per_epoch, list)
        assert all(isinstance(v, int) for v in corpora_samples_per_epoch)

        assert len(corpora_data) == len(corpora_samples_per_epoch)
        assert len(corpora_data) > 0

        assert input_text_field is not None
        assert isinstance(input_aspects_field, str)

        assert input_aspects_field is not None
        assert isinstance(input_aspects_field, str)

        # Store the given data.
        self.corpora_data = corpora_data
        self.corpora_samples_per_epoch = corpora_samples_per_epoch
        self.text_field = input_text_field
        self.aspects_field = input_aspects_field

        # Setup empty train keys data.
        self.train_keys = []
        self.on_epoch_begin(None, None, None)


    def on_epoch_begin(self,
                       args: transformers.TrainingArguments,
                       state: transformers.TrainerState,
                       control: transformers.TrainerControl,
                       **kwargs):
        self.train_keys = []
        for corpus_idx, corpus_data, corpus_samples in \
                zip(range(len(self.corpora_data)), self.corpora_data, self.corpora_samples_per_epoch):
            # Select "corpus_sample" random keys for the given corpus.
            keys = list(corpus_data.keys())
            random.shuffle(keys)
            keys = keys[:corpus_samples]

            # Add the selected keys to the train data.
            self.train_keys.extend([(corpus_idx, v) for v in keys])
            del keys
        try:
            del corpus_data, corpus_samples
        except UnboundLocalError:
            pass

        random.shuffle(self.train_keys)


    def shuffle(self,
                seed: int | None = None,
                generator: numpy.random.Generator | None = None,
                keep_in_memory: bool = False,
                load_from_cache_file: bool | None = None,
                indices_cache_file_name: str | None = None,
                writer_batch_size: int | None = 1000,
                new_fingerprint: str | None = None
                ) -> datasets.Dataset:
        random.shuffle(self.train_keys)
        return self


    def take(self, n: int) -> datasets.Dataset:
        # Subdivide the train keys according to their corpus.
        corpora_train_keys = [[] for _ in self.corpora_data]
        for i, v in enumerate(self.train_keys):
            corpora_train_keys[v[0]].append((i, v[0], v[1]))
        try:
            del i, v
        except UnboundLocalError:
            pass

        # Sample the first n keys for each corpus.
        self.corpora_samples_per_epoch = [min(n, v) for v in self.corpora_samples_per_epoch]
        corpora_train_keys = [v[:n] for v, n in zip(corpora_train_keys, self.corpora_samples_per_epoch)]

        # Rebuild the train keys, maintaining the ordering of train samples.
        self.train_keys = sorted(*corpora_train_keys, key=operator.itemgetter(0), reverse=False)
        self.train_keys = [(v1, v2) for _, v1, v2 in self.train_keys]

        return self


    def __len__(self) -> int:
        return len(self.train_keys)


    def __getitem__(self, indexes) -> dict[str, typing.Any]:
        # Convert arguments to list of ints.
        if isinstance(indexes, torch.Tensor):
            indexes = indexes.tolist()
            if len(indexes) <= 0:
                raise ValueError("The given tensor is empty.")
            if not isinstance(indexes[0], int):
                raise ValueError("The given tensor doesn't have integer dtype.")
        elif isinstance(indexes, list):
            if len(indexes) <= 0:
                raise ValueError("The given list is empty.")
        elif isinstance(indexes, int):
            indexes = [indexes]
        else:
            raise NotImplementedError
        
        length_dataset = len(self.train_keys)

        # Extract the text and aspects of all elements in the current batch.
        batch_text = []
        batch_aspects = []
        for idx in indexes:
            # Ensure the given index is valid.
            if idx >= length_dataset:
                raise ValueError(f"The given index is out of range: Found {idx}, "
                                 f"but the dataset has {length_dataset} samples.")

            # Extract the correct data from the corpora.
            n_corpus, key = self.train_keys[idx]
            data = self.corpora_data[n_corpus][key]
            del n_corpus, key

            # Append the data to the current batch.
            batch_text.append(data[self.text_field])
            batch_aspects.append(data[self.aspects_field])
            del data
        try:
            del idx
        except UnboundLocalError:
            pass
        del length_dataset

        return {"text": batch_text, "aspects": batch_aspects}
