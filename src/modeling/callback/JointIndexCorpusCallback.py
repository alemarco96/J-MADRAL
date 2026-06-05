import modeling.dataset.JointFinetuningDataset
import modeling.model.TrainDoubleBiEncoderModel
import os
import transformers
import utils


class JointIndexCorpusCallback(transformers.TrainerCallback):
    def __init__(self,
                 dataset: modeling.dataset.JointFinetuningDataset,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 model: modeling.model.TrainDoubleBiEncoderModel,
                 model_folder: str,
                 r_corpus_index_folder: str,
                 s_corpus_index_folder: str,
                 r_query_index_folder: str,
                 s_query_index_folder: str,
                 rs_query_index_folder: str,
                 sr_query_index_folder: str,
                 chunk_size: int = 4096,
                 batch_size: int = 64,
                 max_num_tokens: int | None = 128,
                 topk_documents: int = 100):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model

        self.r_corpus_data = self.dataset.r_corpus_data
        self.s_corpus_data = self.dataset.s_corpus_data

        self.r_queries_data = self.dataset.r_queries_data
        self.s_queries_data = self.dataset.s_queries_data
        self.rs_queries_data = self.dataset.rs_queries_data
        self.sr_queries_data = self.dataset.sr_queries_data

        assert os.path.exists(model_folder)
        assert os.path.isdir(model_folder)
        self.model_folder = model_folder

        assert os.path.exists(r_corpus_index_folder)
        assert os.path.isdir(r_corpus_index_folder)
        self.r_corpus_index_folder = r_corpus_index_folder
        assert os.path.exists(s_corpus_index_folder)
        assert os.path.isdir(s_corpus_index_folder)
        self.s_corpus_index_folder = s_corpus_index_folder

        assert os.path.exists(r_query_index_folder)
        assert os.path.isdir(r_query_index_folder)
        self.r_query_index_folder = r_query_index_folder
        assert os.path.exists(s_query_index_folder)
        assert os.path.isdir(s_query_index_folder)
        self.s_query_index_folder = s_query_index_folder
        assert os.path.exists(rs_query_index_folder)
        assert os.path.isdir(rs_query_index_folder)
        self.rs_query_index_folder = rs_query_index_folder
        assert os.path.exists(sr_query_index_folder)
        assert os.path.isdir(sr_query_index_folder)
        self.sr_query_index_folder = sr_query_index_folder

        self.chunk_size = chunk_size
        self.batch_size = batch_size

        self.num_warmup_epochs = self.dataset.num_warmup_epochs
        self.every_k_epochs = self.dataset.every_k_epochs

        self.max_num_tokens = max_num_tokens
        self.topk_documents = topk_documents


    def on_epoch_begin(self,
                       args: transformers.TrainingArguments,
                       state: transformers.TrainerState,
                       control: transformers.TrainerControl,
                       **kwargs) -> None:
        assert state.epoch is not None
        epoch = int(state.epoch + 1e-4)
        print(f"** EPOCH BEGIN: {epoch} **", flush=True)

    def on_epoch_end(self,
                     args: transformers.TrainingArguments,
                     state: transformers.TrainerState,
                     control: transformers.TrainerControl,
                     **kwargs) -> None:
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

        if not os.path.exists(os.path.join(self.r_corpus_index_folder, idx_epoch)):
            os.mkdir(os.path.join(self.r_corpus_index_folder, idx_epoch))
        if not os.path.exists(os.path.join(self.s_corpus_index_folder, idx_epoch)):
            os.mkdir(os.path.join(self.s_corpus_index_folder, idx_epoch))

        if not os.path.exists(os.path.join(self.r_query_index_folder, idx_epoch)):
            os.mkdir(os.path.join(self.r_query_index_folder, idx_epoch))
        if not os.path.exists(os.path.join(self.s_query_index_folder, idx_epoch)):
            os.mkdir(os.path.join(self.s_query_index_folder, idx_epoch))
        if not os.path.exists(os.path.join(self.rs_query_index_folder, idx_epoch)):
            os.mkdir(os.path.join(self.rs_query_index_folder, idx_epoch))
        if not os.path.exists(os.path.join(self.sr_query_index_folder, idx_epoch)):
            os.mkdir(os.path.join(self.sr_query_index_folder, idx_epoch))

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
        # Perform the indexing.
        # --------------------------------------------------------------------------------------------------------------
        # R corpus: products.
        utils.index_corpus(
            tokenizer=self.tokenizer,
            model=self.model.d_model,
            corpus=self.r_corpus_data,
            index_folder=os.path.join(self.r_corpus_index_folder, idx_epoch),
            chunk_size=self.chunk_size,
            batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens
        )

        # # S corpus: reviews.
        # utils.index_corpus(
        #     tokenizer=self.tokenizer,
        #     model=self.model.d_model,
        #     corpus=self.s_corpus_data,
        #     index_folder=os.path.join(self.s_corpus_index_folder, idx_epoch),
        #     chunk_size=self.chunk_size,
        #     batch_size=self.batch_size,
        #     max_num_tokens=self.max_num_tokens
        # )

        # R train queries.
        utils.index_corpus(
            tokenizer=self.tokenizer,
            model=self.model.q_model,
            corpus=self.r_queries_data,
            index_folder=os.path.join(self.r_query_index_folder, idx_epoch),
            chunk_size=self.chunk_size,
            batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens
        )

        # # S train queries.
        # utils.index_corpus(
        #     tokenizer=self.tokenizer,
        #     model=self.model.q_model,
        #     corpus=self.s_queries_data,
        #     index_folder=os.path.join(self.s_query_index_folder, idx_epoch),
        #     chunk_size=self.chunk_size,
        #     batch_size=self.batch_size,
        #     max_num_tokens=self.max_num_tokens
        # )

        # # RS train queries.
        # utils.index_corpus(
        #     tokenizer=self.tokenizer,
        #     model=self.model.d_model,
        #     corpus=self.rs_queries_data,
        #     index_folder=os.path.join(self.rs_query_index_folder, idx_epoch),
        #     chunk_size=self.chunk_size,
        #     batch_size=self.batch_size,
        #     max_num_tokens=self.max_num_tokens
        # )

        # # SR train queries.
        # utils.index_corpus(
        #     tokenizer=self.tokenizer,
        #     model=self.model.d_model,
        #     corpus=self.sr_queries_data,
        #     index_folder=os.path.join(self.sr_query_index_folder, idx_epoch),
        #     chunk_size=self.chunk_size,
        #     batch_size=self.batch_size,
        #     max_num_tokens=self.max_num_tokens
        # )

        # --------------------------------------------------------------------------------------------------------------
        # Perform retrieval and update the train data of the dataset.
        # --------------------------------------------------------------------------------------------------------------
        # R train.
        run_data = utils.retrieval_corpus(
            corpus_index_folder=os.path.join(self.r_corpus_index_folder, idx_epoch),
            query_index_folder=os.path.join(self.r_query_index_folder, idx_epoch),
            chunk_size=self.chunk_size,
            topk_documents=self.topk_documents
        )
        self.dataset.update_train_data("r", run_data)
        del run_data

        # # S train.
        # run_data = utils.retrieval_corpus(
        #     corpus_index_folder=os.path.join(self.s_corpus_index_folder, idx_epoch),
        #     query_index_folder=os.path.join(self.s_query_index_folder, idx_epoch),
        #     chunk_size=self.chunk_size,
        #     topk_documents=self.topk_documents
        # )
        # self.dataset.update_train_data("s", run_data)
        # del run_data

        # # RS train.
        # run_data = utils.retrieval_corpus(
        #     corpus_index_folder=os.path.join(self.s_corpus_index_folder, idx_epoch),
        #     query_index_folder=os.path.join(self.rs_query_index_folder, idx_epoch),
        #     chunk_size=self.chunk_size,
        #     topk_documents=self.topk_documents
        # )
        # self.dataset.update_train_data("rs", run_data)
        # del run_data

        # # SR train.
        # run_data = utils.retrieval_corpus(
        #     corpus_index_folder=os.path.join(self.r_corpus_index_folder, idx_epoch),
        #     query_index_folder=os.path.join(self.sr_query_index_folder, idx_epoch),
        #     chunk_size=self.chunk_size,
        #     topk_documents=self.topk_documents
        # )
        # self.dataset.update_train_data("sr", run_data)
        # del run_data

        del idx_epoch
