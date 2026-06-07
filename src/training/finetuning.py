import argparse
import json
import modeling
import os
import transformers
import time


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument("--base_model",
                      dest="base_model",
                      type=str,
                      required=True)
    args.add_argument("--device",
                      dest="device",
                      type=str,
                      default="cuda",
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
    args.add_argument("--loss_logging_filename",
                      dest="loss_logging_filename",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--p_corpus_filename",
                      dest="p_corpus_filename",
                      type=str,
                      required=True)
    args.add_argument("--r_corpus_filename",
                      dest="r_corpus_filename",
                      type=str,
                      required=True)
    args.add_argument("--p_queries_filename",
                      dest="p_queries_filename",
                      type=str,
                      required=True)
    args.add_argument("--r_queries_filename",
                      dest="r_queries_filename",
                      type=str,
                      required=True)
    args.add_argument("--p_train_filename",
                      dest="p_train_filename",
                      type=str,
                      required=True)
    args.add_argument("--r_train_filename",
                      dest="r_train_filename",
                      type=str,
                      required=True)
    args.add_argument("--num_train_epochs",
                      dest="num_train_epochs",
                      type=int,
                      default=20,
                      required=False)
    args.add_argument("--learning_rate",
                      dest="learning_rate",
                      type=float,
                      default=5e-5,
                      required=False)
    args.add_argument("--batch_size",
                      dest="batch_size",
                      type=int,
                      default=64,
                      required=False)
    args.add_argument("--max_num_tokens",
                      dest="max_num_tokens",
                      type=int,
                      default=None,
                      required=False)
    args.add_argument("--gradient_accumulation_steps",
                      dest="gradient_accumulation_steps",
                      type=int,
                      default=1,
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
    args.add_argument("--finetune_alpha",
                      dest="finetune_alpha",
                      type=float,
                      default=0.0,
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
          f"Base model:             {args.base_model}.", flush=False)
    print(f"Query output folder:    {args.query_model_folder}.", flush=False)
    print(f"Doc output folder:      {args.document_model_folder}.", flush=False)
    print(f"Train output folder:    {args.train_model_folder}.", flush=False)
    print(f"Loss logging filename:  {args.loss_logging_filename}.", flush=False)
    print("--------------------------------------------------", flush=False)
    print(f"P corpus filename:      {args.p_corpus_filename}.", flush=False)
    print(f"R corpus filename:      {args.r_corpus_filename}.", flush=False)
    print(f"P queries filename:     {args.p_queries_filename}.", flush=False)
    print(f"R queries filename:     {args.r_queries_filename}.", flush=False)
    print(f"P train data filename:  {args.p_train_filename}.", flush=False)
    print(f"R train data filename:  {args.r_train_filename}.", flush=False)
    print("--------------------------------------------------", flush=False)
    print(f"Num train epochs:       {args.num_train_epochs}.", flush=False)
    print(f"Learning rate:          {args.learning_rate}.", flush=False)
    print(f"Batch size:             {args.batch_size}.", flush=False)
    print(f"Max number of tokens:   {args.max_num_tokens}.", flush=False)
    print(f"Gradient acc. steps:    {args.gradient_accumulation_steps}.", flush=False)
    print(f"Gradient checkpointing: {args.gradient_checkpointing}.", flush=False)
    print(f"Fine-tune alpha:        {args.finetune_alpha}.", flush=False)
    print(f"Num positives:          {args.num_positives}.", flush=False)
    print(f"Num negatives:          {args.num_negatives}.", flush=False)

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

    print(f"\n", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Ensure that the input files do exist.
    # ------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(args.p_corpus_filename):
        raise ValueError(f"Unable to find the input P corpus file: found {args.p_corpus_filename}.")
    if not os.path.exists(args.r_corpus_filename):
        raise ValueError(f"Unable to find the input R corpus file: found {args.r_corpus_filename}.")
    if not os.path.exists(args.p_train_filename):
        raise ValueError(f"Unable to find the input P train file: found {args.p_train_filename}.")
    if not os.path.exists(args.r_train_filename):
        raise ValueError(f"Unable to find the input R train file: found {args.r_train_filename}.")
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

    # ------------------------------------------------------------------------------------------------------------------
    # Load the tokenizer and the model.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()

    tk = transformers.AutoTokenizer.from_pretrained(args.base_model)
    assert isinstance(tk, transformers.PreTrainedTokenizerBase)
    model = modeling.TrainDoubleBiEncoderModel.from_pretrained(args.base_model, device_map=args.device)
    assert isinstance(model, modeling.TrainDoubleBiEncoderModel)

    print(f"Model loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
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
    # Read the product search corpus data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    with open(args.p_corpus_filename, "rt", encoding="utf-8") as fi:
        # p_corpus_data = {v["id"]: {"text": v["text"], "aspects": v["aspects"]} for v in map(json.loads, fi)}
        p_corpus_data = {v["id"]: {
            "text": f"{v['title']}\n{v['description']}\n{v['bullet_point']}".strip(),
            "aspects": [v for r, v in zip(args.retain_aspects, v["aspects"]) if r]
        } for v in map(json.loads, fi)}
    del fi

    print(f"P corpus loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Read the review search corpus data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    with open(args.r_corpus_filename, "rt", encoding="utf-8") as fi:
        # r_corpus_data = {v["id"]: {"text": v["text"], "aspects": v["aspects"]} for v in map(json.loads, fi)}
        r_corpus_data = {v["id"]: {
            "text": f"{v['title']}\n{v['text']}".strip(),
            "aspects": [v for r, v in zip(args.retain_aspects, v["aspects"]) if r]
        } for v in map(json.loads, fi)}
    del fi

    print(f"R corpus loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Read the product search queries data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    with open(args.p_queries_filename, "rt", encoding="utf-8") as fi:
        # p_queries_data = {int(v["id"]): {"text": v["text"], "aspects": v["aspects"]} for v in map(json.loads, fi)}
        p_queries_data = {int(v["id"]): {
            "text": v["text"].strip(),
            "aspects": [v for r, v in zip(args.retain_aspects, v["aspects"]) if r]
        } for v in map(json.loads, fi)}
    del fi

    print(f"P queries loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Read the review search queries data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    with open(args.r_queries_filename, "rt", encoding="utf-8") as fi:
        # r_queries_data = {v["id"]: {"text": v["text"], "aspects": v["aspects"]} for v in map(json.loads, fi)}
        r_queries_data = {v["id"]: {
            "text": v["text"].strip(),
            "aspects": [v for r, v in zip(args.retain_aspects, v["aspects"]) if r]
        } for v in map(json.loads, fi)}
    del fi

    print(f"R queries loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Read the product search train data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    with open(args.p_train_filename, "rt", encoding="utf-8") as fi:
        p_train_data = {int(v["query_id"]): {
            "task": "p",
            "exclude_id": [k for k in v["exclude_id"]],
            "p_rank": [(k, r1, r2) for k, r1, r2 in v["p_rank"]],
            "n_rank": [(k, r1, r2) for k, r1, r2 in v["n_rank"]],
            "hn_rank": [(k, r1, r2) for k, r1, r2 in v["hn_rank"]]
        } for v in map(json.loads, fi)}
    del fi

    print(f"P train data loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Read the review search train data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    with open(args.r_train_filename, "rt", encoding="utf-8") as fi:
        r_train_data = {int(v["query_id"]): {
            "task": "r",
            "exclude_id": [k for k in v["exclude_id"]],
            "p_rank": [(k, r1, r2) for k, r1, r2 in v["p_rank"]],
            "n_rank": [(k, r1, r2) for k, r1, r2 in v["n_rank"]],
            "hn_rank": [(k, r1, r2) for k, r1, r2 in v["hn_rank"]]
        } for v in map(json.loads, fi)}
    del fi

    print(f"R train data loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    print("", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Instantiate the training dataset.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    train_dataset = modeling.FinetuningDataset(
        p_corpus_data=p_corpus_data,
        r_corpus_data=r_corpus_data,
        p_queries_data=p_queries_data,
        r_queries_data=r_queries_data,
        p_train_data=p_train_data,
        r_train_data=r_train_data,
        batch_size=args.batch_size,
        num_positives=args.num_positives,
        num_negatives=args.num_negatives
    )
    train_dataset = train_dataset.shuffle()

    print(f"Loaded the train dataset in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare the collator.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    collator = modeling.FinetuningCollator(tk, max_num_tokens=args.max_num_tokens)

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
        logging_strategy="no",

        # Checkpointing and saving.
        save_strategy="no",
        remove_unused_columns=False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Define the trainer.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    trainer = modeling.FinetuningTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        callbacks=[train_dataset],
        data_collator=collator,
        finetune_alpha=args.finetune_alpha,
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
    model.untie_all()
    print("\n** Finished training! **\n", flush=True)

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
