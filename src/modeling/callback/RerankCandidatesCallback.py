import modeling.dataset.CrossEncoderEVOTrainingTriplesDataset
import modeling.model.TrainCrossEncoderModel
import os
import transformers
import utils


class RerankCandidatesCallback(transformers.TrainerCallback):
    def __init__(self,
                 dataset: modeling.dataset.CrossEncoderEVOTrainingTriplesDataset,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 model: modeling.model.TrainCrossEncoderModel,
                 corpus_data: dict[int | str, str],
                 queries_data: dict[int | str, str],
                 model_folder: str | None,
                 scores_folder: str | None,
                 num_warmup_epochs: int = 4,
                 every_k_epochs: int = 4,
                 batch_size: int = 64,
                 max_num_tokens: int | None = 160):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model

        self.corpus_keys = {k: i for i, k in enumerate(sorted(corpus_data.keys()))}
        self.corpus_data = corpus_data
        self.queries_keys = {k: i for i, k in enumerate(sorted(queries_data.keys()))}
        self.queries_data = queries_data

        if model_folder is not None:
            assert os.path.exists(model_folder)
            assert os.path.isdir(model_folder)
        self.model_folder = model_folder

        if scores_folder is not None:
            assert os.path.exists(scores_folder)
            assert os.path.isdir(scores_folder)
        self.scores_folder = scores_folder

        self.num_warmup_epochs = num_warmup_epochs
        self.every_k_epochs = every_k_epochs

        self.batch_size = batch_size
        self.max_num_tokens = max_num_tokens

    def on_epoch_begin(self,
                       args: transformers.TrainingArguments,
                       state: transformers.TrainerState,
                       control: transformers.TrainerControl,
                       **kwargs):
        assert state.epoch is not None
        epoch = int(state.epoch + 1e-4)
        print(f"** EPOCH BEGIN: {epoch} **", flush=True)

    def on_epoch_end(self,
                     args: transformers.TrainingArguments,
                     state: transformers.TrainerState,
                     control: transformers.TrainerControl,
                     **kwargs):
        assert state.epoch is not None
        epoch = int(state.epoch + 1e-4)
        print(f"** EPOCH END: {epoch} **", flush=True)

        if epoch < self.num_warmup_epochs or epoch >= state.num_train_epochs or \
                ((epoch - self.num_warmup_epochs) % self.every_k_epochs) != 0:
            del epoch
            return

        idx_epoch = str((epoch - self.num_warmup_epochs) // self.every_k_epochs)
        del epoch

        # --------------------------------------------------------------------------------------------------------------
        # Create the auxiliary folders.
        # --------------------------------------------------------------------------------------------------------------
        if self.model_folder is not None:
            if not os.path.exists(os.path.join(self.model_folder, idx_epoch)):
                os.mkdir(os.path.join(self.model_folder, idx_epoch))
            if not os.path.exists(os.path.join(os.path.join(self.model_folder, idx_epoch), "infer_model")):
                os.mkdir(os.path.join(os.path.join(self.model_folder, idx_epoch), "infer_model"))
            if not os.path.exists(os.path.join(os.path.join(self.model_folder, idx_epoch), "train_model")):
                os.mkdir(os.path.join(os.path.join(self.model_folder, idx_epoch), "train_model"))
        if self.scores_folder is not None:
            if not os.path.exists(os.path.join(self.scores_folder, idx_epoch)):
                os.mkdir(os.path.join(self.scores_folder, idx_epoch))

        # --------------------------------------------------------------------------------------------------------------
        # Save the training model to disk.
        # --------------------------------------------------------------------------------------------------------------
        if self.model_folder is not None:
            try:
                self.tokenizer.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "infer_model"))
                self.model.model.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "infer_model"),
                                                   safe_serialization=True)
            except:
                print("Unable to save the infer model: not saved.", flush=True)

            try:
                self.tokenizer.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "train_model"))
                self.model.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "train_model"),
                                           safe_serialization=True)
            except:
                print("Unable to save the train model: not saved.", flush=True)

        # --------------------------------------------------------------------------------------------------------------
        # Perform the operation.
        # --------------------------------------------------------------------------------------------------------------
        candidate_data = {k1: v1["e_id"] + v1["s_id"] + v1["c_id"] + v1["i_id"] + v1["sp_id"] + v1["hn_id"]
                          for k1, v1 in self.dataset.train_data.items()}

        # noinspection PyTypeChecker
        rerank_data = utils.rerank_candidates_while_training(
            tokenizer=self.tokenizer,
            model=self.model,
            candidates=candidate_data,
            corpus=self.corpus_data,
            queries=self.queries_data,
            scores_filename=os.path.join(os.path.join(self.scores_folder, idx_epoch), "train_esci.jsonl")
                if self.scores_folder is not None else None,
            batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens
        )

        self.dataset.update_triples(rerank_data)
        del candidate_data, rerank_data, idx_epoch
