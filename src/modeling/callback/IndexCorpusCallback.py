import modeling.dataset.BiEncoderTrainingTriplesDataset
import modeling.model.TrainDoubleBiEncoderModel
import os
import transformers
import utils


class IndexCorpusCallback(transformers.TrainerCallback):
    def __init__(self,
                 dataset: modeling.dataset.BiEncoderTrainingTriplesDataset,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 model: modeling.model.TrainDoubleBiEncoderModel,
                 corpus_data: dict[int | str, str],
                 queries_data: dict[int | str, str],
                 model_folder: str,
                 doc_index_folder: str,
                 query_index_folder: str,
                 num_warmup_epochs: int = 4,
                 every_k_epochs: int = 4,
                 chunk_size: int = 4096,
                 batch_size: int = 64,
                 max_num_tokens: int | None = 128,
                 topk_documents: int = 100):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model

        self.corpus_keys = {k: i for i, k in enumerate(sorted(corpus_data.keys()))}
        self.corpus_data = corpus_data
        self.queries_keys = {k: i for i, k in enumerate(sorted(queries_data.keys()))}
        self.queries_data = queries_data

        assert os.path.exists(model_folder)
        assert os.path.isdir(model_folder)
        self.model_folder = model_folder
        assert os.path.exists(doc_index_folder)
        assert os.path.isdir(doc_index_folder)
        self.doc_index_folder = doc_index_folder
        assert os.path.exists(query_index_folder)
        assert os.path.isdir(query_index_folder)
        self.query_index_folder = query_index_folder

        self.chunk_size = chunk_size
        self.batch_size = batch_size

        self.max_num_tokens = max_num_tokens
        self.topk_documents = topk_documents
        self.every_k_epochs = every_k_epochs
        self.num_warmup_epochs = num_warmup_epochs

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
        if not os.path.exists(os.path.join(self.model_folder, idx_epoch)):
            os.mkdir(os.path.join(self.model_folder, idx_epoch))
        if not os.path.exists(os.path.join(os.path.join(self.model_folder, idx_epoch), "q_model")):
            os.mkdir(os.path.join(os.path.join(self.model_folder, idx_epoch), "q_model"))
        if not os.path.exists(os.path.join(os.path.join(self.model_folder, idx_epoch), "d_model")):
            os.mkdir(os.path.join(os.path.join(self.model_folder, idx_epoch), "d_model"))
        if not os.path.exists(os.path.join(os.path.join(self.model_folder, idx_epoch), "train_model")):
            os.mkdir(os.path.join(os.path.join(self.model_folder, idx_epoch), "train_model"))
        if not os.path.exists(os.path.join(self.doc_index_folder, idx_epoch)):
            os.mkdir(os.path.join(self.doc_index_folder, idx_epoch))
        if not os.path.exists(os.path.join(self.query_index_folder, idx_epoch)):
            os.mkdir(os.path.join(self.query_index_folder, idx_epoch))

        # --------------------------------------------------------------------------------------------------------------
        # Save the training model to disk.
        # --------------------------------------------------------------------------------------------------------------
        try:
            self.tokenizer.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "q_model"))
            self.model.q_model.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "q_model"),
                                               safe_serialization=False)
        except:
            print("Unable to save the query model: not saved.", flush=True)

        try:
            self.tokenizer.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "d_model"))
            self.model.d_model.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "d_model"),
                                               safe_serialization=False)
        except:
            print("Unable to save the document model: not saved.", flush=True)

        try:
            self.tokenizer.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "train_model"))
            self.model.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "train_model"),
                                       safe_serialization=False)
        except:
            print("Unable to save the train model: not saved.", flush=True)

        # --------------------------------------------------------------------------------------------------------------
        # Perform the operation.
        # --------------------------------------------------------------------------------------------------------------
        queries_embedding, run_data = utils.index_corpus_while_training(
            tokenizer=self.tokenizer,
            model=self.model,
            corpus=self.corpus_data,
            queries=self.queries_data,
            doc_index_folder=os.path.join(self.doc_index_folder, idx_epoch),
            query_index_folder=os.path.join(self.query_index_folder, idx_epoch),
            chunk_size=self.chunk_size,
            batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            topk_documents=self.topk_documents
        )

        self.dataset.update_triples(
            run_data=run_data,
            queries_keys=self.queries_keys,
            queries_embedding=queries_embedding,
            corpus_keys=self.corpus_keys,
            index_filename=os.path.join(os.path.join(self.doc_index_folder, idx_epoch), "index")
        )
        del queries_embedding, run_data, idx_epoch
