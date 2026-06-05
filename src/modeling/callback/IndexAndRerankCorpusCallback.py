import json
import modeling.dataset.BiEncoderEVOTrainingTriplesDataset
import modeling.model.TrainDoubleBiEncoderModel
import modeling.model.TrainCrossEncoderModel
import operator
import os
import transformers
import utils


class IndexAndRerankCorpusCallback(transformers.TrainerCallback):
    def __init__(self,
                 dataset: modeling.dataset.BiEncoderEVOTrainingTriplesDataset,
                 bi_tokenizer: transformers.PreTrainedTokenizerBase,
                 bi_model: modeling.model.TrainDoubleBiEncoderModel,
                 cross_tokenizer: transformers.PreTrainedTokenizerBase,
                 cross_model: modeling.model.TrainCrossEncoderModel,
                 corpus_data: dict[int | str, str],
                 queries_data: dict[int | str, str],
                 scores_data: dict[int | str, dict[int | str, float]],
                 model_folder: str,
                 doc_index_folder: str,
                 query_index_folder: str,
                 scores_folder: str,
                 num_warmup_epochs: int = 4,
                 every_k_epochs: int = 4,
                 chunk_size: int = 4096,
                 batch_size: int = 64,
                 bi_max_num_tokens: int | None = 128,
                 cross_max_num_tokens: int | None = 160,
                 topk_documents: int = 100):
        self.dataset = dataset
        self.bi_tokenizer = bi_tokenizer
        self.bi_model = bi_model
        self.cross_tokenizer = cross_tokenizer
        self.cross_model = cross_model

        self.corpus_keys = {k: i for i, k in enumerate(sorted(corpus_data.keys()))}
        self.corpus_data = corpus_data
        self.queries_keys = {k: i for i, k in enumerate(sorted(queries_data.keys()))}
        self.queries_data = queries_data
        self.scores_data = scores_data

        assert os.path.exists(model_folder)
        assert os.path.isdir(model_folder)
        self.model_folder = model_folder
        assert os.path.exists(doc_index_folder)
        assert os.path.isdir(doc_index_folder)
        self.doc_index_folder = doc_index_folder
        assert os.path.exists(query_index_folder)
        assert os.path.isdir(query_index_folder)
        self.query_index_folder = query_index_folder
        assert os.path.exists(scores_folder)
        assert os.path.isdir(scores_folder)
        self.scores_folder = scores_folder

        self.num_warmup_epochs = num_warmup_epochs
        self.every_k_epochs = every_k_epochs

        self.chunk_size = chunk_size
        self.batch_size = batch_size

        self.bi_max_num_tokens = bi_max_num_tokens
        self.cross_max_num_tokens = cross_max_num_tokens
        self.topk_documents = topk_documents


    def on_epoch_begin(self,
                       args: transformers.TrainingArguments,
                       state: transformers.TrainerState,
                       control: transformers.TrainerControl,
                       **kwargs):
        assert state.epoch is not None
        epoch = int(state.epoch + 1e-4)
        print(f"** EPOCH BEGIN: {epoch} **\n", flush=True)

    def on_epoch_end(self,
                     args: transformers.TrainingArguments,
                     state: transformers.TrainerState,
                     control: transformers.TrainerControl,
                     **kwargs):
        assert state.epoch is not None
        epoch = int(state.epoch + 1e-4)
        print(f"** EPOCH END: {epoch} **\n", flush=True)

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
        if not os.path.exists(os.path.join(self.scores_folder, idx_epoch)):
            os.mkdir(os.path.join(self.scores_folder, idx_epoch))

        # --------------------------------------------------------------------------------------------------------------
        # Save the training model to disk.
        # --------------------------------------------------------------------------------------------------------------
        try:
            self.bi_tokenizer.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "q_model"))
            self.bi_model.q_model.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "q_model"),
                                                  safe_serialization=False)
        except:
            print("Unable to save the query model: not saved.", flush=True)

        try:
            self.bi_tokenizer.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "d_model"))
            self.bi_model.d_model.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "d_model"),
                                                  safe_serialization=False)
        except:
            print("Unable to save the document model: not saved.", flush=True)

        try:
            self.bi_tokenizer.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "train_model"))
            self.bi_model.save_pretrained(os.path.join(os.path.join(self.model_folder, idx_epoch), "train_model"),
                                          safe_serialization=False)
        except:
            print("Unable to save the train model: not saved.", flush=True)

        # --------------------------------------------------------------------------------------------------------------
        # Set the bi encoder model to eval mode.
        # --------------------------------------------------------------------------------------------------------------
        self.bi_model.eval()

        # --------------------------------------------------------------------------------------------------------------
        # Perform the corpus indexing and retrieval of the train queries.
        # --------------------------------------------------------------------------------------------------------------
        queries_embedding, run_data = utils.index_corpus_while_training(
            tokenizer=self.bi_tokenizer,
            model=self.bi_model,
            corpus=self.corpus_data,
            queries=self.queries_data,
            doc_index_folder=os.path.join(self.doc_index_folder, idx_epoch),
            query_index_folder=os.path.join(self.query_index_folder, idx_epoch),
            chunk_size=self.chunk_size,
            batch_size=self.batch_size,
            max_num_tokens=self.bi_max_num_tokens,
            topk_documents=self.topk_documents
        )

        # --------------------------------------------------------------------------------------------------------------
        # Set the bi encoder model to train mode.
        # --------------------------------------------------------------------------------------------------------------
        self.bi_model.train()

        # --------------------------------------------------------------------------------------------------------------
        # Rerank query-document pairs not already scored by the cross encoder model.
        # --------------------------------------------------------------------------------------------------------------
        assert isinstance(run_data, dict)
        assert all(isinstance(v1, list) and isinstance(v1[0], int | str) for v1 in run_data.values())

        # Retain only top-k candidates for each query.
        candidates_data = {k1: v1[:self.topk_documents] for k1, v1 in run_data.items()}

        # Find which documents have never been scored by the cross encoder model.
        candidates_data = {k1: [v2 for v2 in v1 if v2 not in set(self.scores_data.get(k1, {}).keys())]
                           for k1, v1 in candidates_data.items()}
        candidates_data = {k1: v1 for k1, v1 in candidates_data.items() if len(v1) > 0}

        # Rerank all query-document pairs never evaluated using the cross encoder model.
        reranked_data = utils.rerank_candidates_while_training(
            tokenizer=self.cross_tokenizer,
            model=self.cross_model,
            candidates=candidates_data,
            corpus=self.corpus_data,
            queries=self.queries_data,
            scores_filename=None,
            batch_size=self.batch_size,
            max_num_tokens=self.cross_max_num_tokens
        )
        del candidates_data
        reranked_data = {k1: {k2: v2 for k2, v2 in v1} for k1, v1 in reranked_data.items()}

        # Update the scores_data stored in memory.
        self.scores_data = {k1: {
            k2: self.scores_data[k1][k2] if k2 in set(self.scores_data.get(k1, {}).keys()) else reranked_data[k1][k2]
            for k2 in set(self.scores_data.get(k1, {}).keys()) | set(reranked_data.get(k1, {}).keys())
        } for k1 in set(self.scores_data.keys() | reranked_data.keys())}

        # Write the scores
        with open(os.path.join(os.path.join(self.scores_folder, idx_epoch), "scores.jsonl"),
                  "wt", encoding="utf-8") as fo:
            for query_id in sorted(self.scores_data.keys()):
                print(json.dumps({
                    "id": query_id,
                    "scores": sorted(self.scores_data[query_id].items(), key=operator.itemgetter(0))
                }), file=fo, flush=False)
            try:
                del query_id
            except UnboundLocalError:
                pass

            fo.flush()
        del fo

        # --------------------------------------------------------------------------------------------------------------
        # Update the train data for the bi encoder model.
        # --------------------------------------------------------------------------------------------------------------
        self.dataset.update_triples(
            run_data=run_data,
            scores_data=self.scores_data,
            queries_keys=self.queries_keys,
            queries_embedding=queries_embedding,
            corpus_keys=self.corpus_keys,
            index_filename=os.path.join(os.path.join(self.doc_index_folder, idx_epoch), "index")
        )
        del queries_embedding, run_data, idx_epoch
