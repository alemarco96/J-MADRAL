import argparse
import json
import modeling
import os
import transformers
import time


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument("--r_corpus_filename",
                      dest="r_corpus_filename",
                      type=str,
                      required=True)
    args.add_argument("--s_corpus_filename",
                      dest="s_corpus_filename",
                      type=str,
                      required=True)
    args.add_argument("--r_train_filename",
                      dest="r_train_filename",
                      type=str,
                      required=True)
    args.add_argument("--s_train_filename",
                      dest="s_train_filename",
                      type=str,
                      required=True)
    args.add_argument("--rs_train_filename",
                      dest="rs_train_filename",
                      type=str,
                      required=True)
    args.add_argument("--sr_train_filename",
                      dest="sr_train_filename",
                      type=str,
                      required=True)
    args.add_argument("--base_model",
                      dest="base_model",
                      type=str,
                      required=True)
    args.add_argument("--device",
                      dest="device",
                      type=str,
                      default="cuda",
                      required=False)
    args.add_argument("--finetune_alpha",
                      dest="finetune_alpha",
                      type=float,
                      default=0.0,
                      required=False)
    args.add_argument("--auxiliary_alpha",
                      dest="auxiliary_alpha",
                      type=float,
                      default=0.0,
                      required=False)
    args.add_argument("--learning_rate",
                      dest="learning_rate",
                      type=float,
                      default=5e-5,
                      required=False)
    args.add_argument("--num_train_epochs",
                      dest="num_train_epochs",
                      type=int,
                      default=20,
                      required=False)
    args.add_argument("--batch_size",
                      dest="batch_size",
                      type=int,
                      default=64,
                      required=False)
    args.add_argument("--gradient_accumulation_steps",
                      dest="gradient_accumulation_steps",
                      type=int,
                      default=1,
                      required=False)
    args.add_argument("--max_num_tokens",
                      dest="max_num_tokens",
                      type=int,
                      default=None,
                      required=False)
    args.add_argument("--gradient_checkpointing",
                      dest="gradient_checkpointing",
                      action="store_true",
                      default=False,
                      required=False)
    args.add_argument("--no_gradient_checkpointing",
                      dest="gradient_checkpointing",
                      action="store_false",
                      required=False)
    args.add_argument("--clone_qd_models",
                      dest="clone_qd_models",
                      action="store_true",
                      default=False,
                      required=False)
    args.add_argument("--no_clone_qd_models",
                      dest="clone_qd_models",
                      action="store_false",
                      required=False)
    args.add_argument("--tie_qd_models",
                      dest="tie_qd_models",
                      action="store_true",
                      default=False,
                      required=False)
    args.add_argument("--no_tie_qd_models",
                      dest="tie_qd_models",
                      action="store_false",
                      required=False)
    args.add_argument("--clone_qd_encoders",
                      dest="clone_qd_encoders",
                      action="store_true",
                      default=False,
                      required=False)
    args.add_argument("--no_clone_qd_encoders",
                      dest="clone_qd_encoders",
                      action="store_false",
                      required=False)
    args.add_argument("--tie_qd_encoders",
                      dest="tie_qd_encoders",
                      action="store_true",
                      default=False,
                      required=False)
    args.add_argument("--no_tie_qd_encoders",
                      dest="tie_qd_encoders",
                      action="store_false",
                      required=False)
    args.add_argument("--clone_qd_tokens_embeddings",
                      dest="clone_qd_tokens_embeddings",
                      action="store_true",
                      default=False,
                      required=False)
    args.add_argument("--no_clone_qd_tokens_embeddings",
                      dest="clone_qd_tokens_embeddings",
                      action="store_false",
                      required=False)
    args.add_argument("--tie_qd_tokens_embeddings",
                      dest="tie_qd_tokens_embeddings",
                      action="store_true",
                      default=False,
                      required=False)
    args.add_argument("--no_tie_qd_tokens_embeddings",
                      dest="tie_qd_tokens_embeddings",
                      action="store_false",
                      required=False)
    args.add_argument("--output_folder",
                      dest="output_folder",
                      type=str,
                      required=True)
    args.add_argument("--loss_logging_filename",
                      dest="loss_logging_filename",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--query_model_folder",
                      dest="query_model_folder",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--document_model_folder",
                      dest="document_model_folder",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--train_model_folder",
                      dest="train_model_folder",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--r_corpus_index_folder",
                      dest="r_corpus_index_folder",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--s_corpus_index_folder",
                      dest="s_corpus_index_folder",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--r_queries_index_folder",
                      dest="r_queries_index_folder",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--s_queries_index_folder",
                      dest="s_queries_index_folder",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--rs_queries_index_folder",
                      dest="rs_queries_index_folder",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--sr_queries_index_folder",
                      dest="sr_queries_index_folder",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--num_positives",
                      dest="num_positives",
                      type=int,
                      default=1,
                      required=False)
    args.add_argument("--num_negatives",
                      dest="num_negatives",
                      type=int,
                      default=1,
                      required=False)
    args.add_argument("--num_warmup_epochs",
                      dest="num_warmup_epochs",
                      type=int,
                      default=4,
                      required=False)
    args.add_argument("--every_k_epochs",
                      dest="every_k_epochs",
                      type=int,
                      default=4,
                      required=False)
    args.add_argument("--min_rank_for_hardnegatives",
                      dest="min_rank_for_hardnegatives",
                      type=int,
                      default=50,
                      required=False)
    args.add_argument("--max_rank_for_hardnegatives",
                      dest="max_rank_for_hardnegatives",
                      type=int,
                      default=100,
                      required=False)
    args.add_argument("--num_retain_hardnegatives",
                      dest="num_retain_hardnegatives",
                      type=int,
                      default=100,
                      required=False)

    return args.parse_args()


def main():
    NS_IN_S = 1000 * 1000 * 1000

    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Log the main train parameters.
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n"
          f"R corpus filename:      {args.r_corpus_filename}.", flush=False)
    print(f"S corpus filename:      {args.s_corpus_filename}.", flush=False)
    print(f"R train data filename:  {args.r_train_filename}.", flush=False)
    print(f"RS train data filename: {args.rs_train_filename}.", flush=False)
    print(f"SR train data filename: {args.sr_train_filename}.", flush=False)
    print(f"Base model:             {args.base_model}.", flush=False)
    print(f"Finetune alpha:         {args.finetune_alpha}.", flush=False)
    print(f"Auxiliary alpha:        {args.auxiliary_alpha}.", flush=False)
    print(f"Learning rate:          {args.learning_rate}.", flush=False)
    print(f"Num train epochs:       {args.num_train_epochs}.", flush=False)
    print(f"Batch size:             {args.batch_size}.", flush=False)
    print(f"Max number of tokens:   {args.max_num_tokens}.", flush=False)
    print(f"Gradient acc. steps:    {args.gradient_accumulation_steps}.", flush=False)
    print(f"Gradient checkpoint:    {args.gradient_checkpointing}.", flush=False)
    print(f"Loss log filename:      {args.loss_logging_filename}.", flush=False)
    print(f"Output folder:          {args.output_folder}.", flush=False)
    print(f"Query output folder:    {args.query_model_folder}.", flush=False)
    print(f"Doc output folder:      {args.document_model_folder}.", flush=False)
    print(f"Train output folder:    {args.train_model_folder}.", flush=False)
    print(f"R corpus index folder:  {args.r_corpus_index_folder}.", flush=False)
    print(f"S corpus index folder:  {args.s_corpus_index_folder}.", flush=False)
    print(f"R query index folder:   {args.r_queries_index_folder}.", flush=False)
    print(f"S query index folder:   {args.s_queries_index_folder}.", flush=False)
    print(f"RS query index folder:  {args.rs_queries_index_folder}.", flush=False)
    print(f"SR query index folder:  {args.sr_queries_index_folder}.", flush=False)
    print(f"Num warmup epochs:      {args.num_warmup_epochs}.", flush=False)
    print(f"Every k epochs:         {args.every_k_epochs}.", flush=False)
    print(f"Max rank for exclude:   {args.min_rank_for_hardnegatives}.", flush=False)
    print(f"Min rank for HN:        {args.min_rank_for_hardnegatives}.", flush=False)
    print(f"Max number of HN:       {args.num_retain_hardnegatives}.", flush=False)

    if args.clone_qd_models:
        print(f"* Clone query and document models.", flush=False)
    if args.tie_qd_models:
        print(f"* Tie query and document models.", flush=False)
    if args.clone_qd_encoders:
        print(f"* Clone query and document encoders.", flush=False)
    if args.tie_qd_encoders:
        print(f"* Tie query and document encoders.", flush=False)
    if args.clone_qd_tokens_embeddings:
        print(f"* Clone query and document tokens_embedding.", flush=False)
    if args.tie_qd_tokens_embeddings:
        print(f"* Tie query and document tokens embedding.", flush=False)

    # ------------------------------------------------------------------------------------------------------------------
    # Ensure that the input files do exist.
    # ------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(args.r_corpus_filename):
        raise ValueError(f"Unable to find the input R corpus file: found {args.r_corpus_filename}.")
    if not os.path.exists(args.s_corpus_filename):
        raise ValueError(f"Unable to find the input S corpus file: found {args.s_corpus_filename}.")
    if not os.path.exists(args.r_train_filename):
        raise ValueError(f"Unable to find the input R train file: found {args.r_train_filename}.")
    if not os.path.exists(args.s_train_filename):
        raise ValueError(f"Unable to find the input S train file: found {args.s_train_filename}.")
    if not os.path.exists(args.rs_train_filename):
        raise ValueError(f"Unable to find the input RS train file: found {args.rs_train_filename}.")
    if not os.path.exists(args.sr_train_filename):
        raise ValueError(f"Unable to find the input SR train file: found {args.sr_train_filename}.")
    if not os.path.exists(args.base_model):
        raise ValueError(f"Unable to find the base model data: found {args.base_model}.")

    # ------------------------------------------------------------------------------------------------------------------
    # Ensure that the output folders do exist.
    # ------------------------------------------------------------------------------------------------------------------
    if args.query_model_folder is not None and not os.path.exists(args.query_model_folder):
        raise ValueError(f"Unable to find the output folder for the query model: found {args.query_model_folder}.")
    if args.document_model_folder is not None and not os.path.exists(args.document_model_folder):
        raise ValueError(f"Unable to find the output folder for the document model: found {args.document_model_folder}.")
    if args.train_model_folder is not None and not os.path.exists(args.train_model_folder):
        raise ValueError(f"Unable to find the output folder for the train model: found {args.train_model_folder}.")
    if args.query_model_folder is None and args.document_model_folder is None and args.train_model_folder is None:
        raise ValueError(f"None of the output folders specified: the training would be pointless.")

    if args.r_corpus_index_folder is None or not os.path.exists(args.r_corpus_index_folder):
        raise ValueError(f"Unable to find the index folder for the R corpus: found {args.r_corpus_index_folder}.")
    if args.s_corpus_index_folder is None or not os.path.exists(args.s_corpus_index_folder):
        raise ValueError(f"Unable to find the index folder for the S corpus: found {args.s_corpus_index_folder}.")
    if args.r_queries_index_folder is None or not os.path.exists(args.r_queries_index_folder):
        raise ValueError(f"Unable to find the index folder for the R queries: found {args.r_queries_index_folder}.")
    if args.s_queries_index_folder is None or not os.path.exists(args.s_queries_index_folder):
        raise ValueError(f"Unable to find the index folder for the S queries: found {args.s_queries_index_folder}.")
    if args.rs_queries_index_folder is None or not os.path.exists(args.rs_queries_index_folder):
        raise ValueError(f"Unable to find the index folder for the RS queries: found {args.rs_queries_index_folder}.")
    if args.sr_queries_index_folder is None or not os.path.exists(args.sr_queries_index_folder):
        raise ValueError(f"Unable to find the index folder for the SR queries: found {args.sr_queries_index_folder}.")

    # ------------------------------------------------------------------------------------------------------------------
    # Load the tokenizer and the model.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()

    tk = transformers.AutoTokenizer.from_pretrained(args.base_model)
    assert isinstance(tk, transformers.PreTrainedTokenizerBase)
    model = modeling.TrainDoubleBiEncoderModel.from_pretrained(args.base_model, device_map=args.device)
    assert isinstance(model, modeling.TrainDoubleBiEncoderModel)

    print(f"Loaded the model in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    print("", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Apply different initialization policies to the model.
    # ------------------------------------------------------------------------------------------------------------------
    if args.clone_qd_models:
        model.clone_qd_models()

    if args.tie_qd_models:
        model.tie_qd_models()

    if args.clone_qd_encoders:
        model.clone_qd_encoders()

    if args.tie_qd_encoders:
        model.tie_qd_encoders()

    if args.clone_qd_tokens_embeddings:
        model.clone_qd_tokens_embeddings()

    if args.tie_qd_tokens_embeddings:
        model.tie_qd_tokens_embeddings()

    # ------------------------------------------------------------------------------------------------------------------
    # Read the training corpora from disk.
    # ------------------------------------------------------------------------------------------------------------------
    r_corpus_data = {}
    s_corpus_data = {}

    for fn, data, d_cast, label in zip([args.r_corpus_filename, args.s_corpus_filename],
                                       [r_corpus_data, s_corpus_data],
                                       [False, False], ["R", "S"]):
        with open(fn, "rt", encoding="utf-8") as fi:
            st = time.time_ns()
            data.update({int(v["id"]) if d_cast else str(v["id"]): \
                             {"text": v["text"], "aspects": v["aspects"]}
                         for v in map(json.loads, fi)})

            print(f"Loaded the {label} corpus in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
            del st
        del fi
    try:
        del fn, data, d_cast, label
    except UnboundLocalError:
        pass

    print("", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Read the training data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    r_train_data = {}
    s_train_data = {}
    rs_train_data = {}
    sr_train_data = {}

    for fn, data, q_cast, d_cast, label in zip(
            [args.r_train_filename, args.s_train_filename, args.rs_train_filename, args.sr_train_filename],
            [r_train_data, s_train_data, rs_train_data, sr_train_data],
            [True, False, False, False], [False, False, False, False], ["r", "s", "rs", "sr"]):
        with open(fn, "rt", encoding="utf-8") as fi:
            st = time.time_ns()
            data.update({int(v["query_id"]) if q_cast else str(v["query_id"]): {
                "task": v["task"],
                "query_text": v["query_text"],
                "query_aspects": v["query_aspects"],
                "exclude_id": [int(k) if d_cast else str(k) for k in v["exclude_id"]],
                "p_rank": [(int(k) if d_cast else str(k), r1, r2) for k, r1, r2 in v["p_rank"]],
                "n_rank": [(int(k) if d_cast else str(k), r1, r2) for k, r1, r2 in v["n_rank"]],
                "hn_rank": [(int(k) if d_cast else str(k), r1, r2) for k, r1, r2 in v["hn_rank"]]
            } for v in map(json.loads, fi)})

            print(f"Loaded the {label} training data in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
            del st
        del fi
    try:
        del fn, data, q_cast, d_cast
    except UnboundLocalError:
        pass

    print("", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Instantiate the training dataset.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    train_dataset = modeling.JointFinetuningDataset(
        r_corpus_data=r_corpus_data,
        s_corpus_data=s_corpus_data,
        r_train_data=r_train_data,
        s_train_data=s_train_data,
        rs_train_data=rs_train_data,
        sr_train_data=sr_train_data,
        batch_size=args.batch_size,
        num_warmup_epochs=args.num_warmup_epochs,
        every_k_epochs=args.every_k_epochs,
        min_rank_for_hardnegatives=args.min_rank_for_hardnegatives,
        max_rank_for_hardnegatives=args.max_rank_for_hardnegatives,
        num_positives=args.num_positives,
        num_negatives=args.num_negatives,
        num_retain_hardnegatives=args.num_retain_hardnegatives
    )
    train_dataset = train_dataset.shuffle()

    print(f"Loaded the train dataset in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Define the callback responsible to reindex the entire catalogue and retrieve new hard negatives.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()

    # noinspection PyTypeChecker
    index_callback = modeling.JointIndexCorpusCallback(
        dataset=train_dataset,
        tokenizer=tk,
        model=model,
        model_folder=args.output_folder,
        r_corpus_index_folder=args.r_corpus_index_folder,
        s_corpus_index_folder=args.s_corpus_index_folder,
        r_query_index_folder=args.r_queries_index_folder,
        s_query_index_folder=args.s_queries_index_folder,
        rs_query_index_folder=args.rs_queries_index_folder,
        sr_query_index_folder=args.sr_queries_index_folder,
        max_num_tokens=args.max_num_tokens
    )

    print(f"Loaded the index corpus callback in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare the collator.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    collator = modeling.BiEncoderFinetuningCollator(tk, max_num_tokens=args.max_num_tokens)

    print(f"Loaded the collator in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare the training arguments.
    # ------------------------------------------------------------------------------------------------------------------
    train_args = transformers.TrainingArguments(
        # Output paths.
        output_dir=args.output_folder,

        # Training duration and batch size.
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,

        # Mixed precision training.
        fp16=False,
        bf16=True,

        # Regularization and training stability.
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,

        # Logging and monitoring training.
        logging_strategy="steps",
        logging_steps=1000,
        include_num_input_tokens_seen="no",

        # Checkpointing and saving.
        save_strategy="no",
        remove_unused_columns=False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Define the trainer.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    trainer = modeling.MadralFinetuningTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[index_callback, train_dataset],
        finetune_alpha=args.finetune_alpha,
        auxiliary_alpha=args.auxiliary_alpha,
        logging_filename=args.loss_logging_filename
    )

    print(f"Loaded the trainer in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    print("", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Perform the training.
    # ------------------------------------------------------------------------------------------------------------------
    print("\n** Starting training... **\n", flush=True)
    trainer.train()

    # ------------------------------------------------------------------------------------------------------------------
    # Save the resulting model to disk.
    # ------------------------------------------------------------------------------------------------------------------
    print("\n** Finished training! **\n", flush=True)
    model.untie_all()

    if args.query_model_folder is not None:
        tk.save_pretrained(args.query_model_folder)
        try:
            model.q_model.save_pretrained(args.query_model_folder, safe_serialization=True)
        except:
            try:
                model.q_model.save_pretrained(args.query_model_folder, safe_serialization=False)
                print("Unable to save the query model: saved with \"safe_serialization=False\".", flush=True)
            except:
                print("Unable to save the query model: not saved.", flush=True)

    if args.document_model_folder is not None:
        tk.save_pretrained(args.document_model_folder)
        try:
            model.d_model.save_pretrained(args.document_model_folder, safe_serialization=True)
        except:
            try:
                model.d_model.save_pretrained(args.document_model_folder, safe_serialization=False)
                print("Unable to save the document model: saved with \"safe_serialization=False\".", flush=True)
            except:
                print("Unable to save the document model: not saved.", flush=True)

    if args.train_model_folder is not None:
        tk.save_pretrained(args.train_model_folder)
        try:
            model.save_pretrained(args.train_model_folder, safe_serialization=True)
        except:
            try:
                model.save_pretrained(args.train_model_folder, safe_serialization=False)
                print("Unable to save the full model: saved with \"safe_serialization=False\".", flush=True)
            except:
                print("Unable to save the full model: not saved.", flush=True)

    print("\nDone!\n", flush=True)


if __name__ == '__main__':
    main()
