import argparse
import json
import modeling
import os
import time
import transformers


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
    args.add_argument("--output_folder",
                      dest="output_folder",
                      type=str,
                      required=True)
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
    args.add_argument("--num_train_epochs",
                      dest="num_train_epochs",
                      type=int,
                      default=20,
                      required=False)
    args.add_argument("--learning_rate",
                      dest="learning_rate",
                      type=float,
                      default=1e-4,
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
    args.add_argument("--mlm_probability",
                      dest="mlm_probability",
                      type=float,
                      default=0.15,
                      required=False)
    args.add_argument("--mask_replace_probability",
                      dest="mask_replace_probability",
                      type=float,
                      default=0.80,
                      required=False)
    args.add_argument("--random_replace_probability",
                      dest="random_replace_probability",
                      type=float,
                      default=0.10,
                      required=False)
    args.add_argument("--pretrain_alpha",
                      dest="pretrain_alpha",
                      type=float,
                      default=0.0,
                      required=False)

    return args.parse_args()


def main():
    NS_IN_S = 1000 * 1000 * 1000

    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize the model-specific parameters.
    # ------------------------------------------------------------------------------------------------------------------
    args.task = args.task.strip().lower()
    if args.task == "product":
        args.retain_aspects = [True, True, True, True, True, True, True, True, False, False]
    elif args.task == "review":
        args.retain_aspects = [False, True, False, False, False, False, False, False, True, True]
    elif args.task == "joint":
        args.retain_aspects = [True, True, True, True, True, True, True, True, True, True]
    else:
        raise ValueError(f"Invalid args.task: found {args.task}.")


    # ------------------------------------------------------------------------------------------------------------------
    # Log the main train parameters.
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n"
          f"Base model:             {args.base_model}.", flush=False)
    print(f"Output folder:          {args.output_folder}.", flush=False)
    print(f"Loss logging filename:  {args.loss_logging_filename}.", flush=False)
    print("--------------------------------------------------", flush=False)
    print(f"P corpus filename:      {args.p_corpus_filename}.", flush=False)
    print(f"R corpus filename:      {args.r_corpus_filename}.", flush=False)
    print(f"P queries filename:     {args.p_queries_filename}.", flush=False)
    print(f"R queries filename:     {args.r_queries_filename}.", flush=False)
    print("--------------------------------------------------", flush=False)
    print(f"Num train epochs:       {args.num_train_epochs}.", flush=False)
    print(f"Learning rate:          {args.learning_rate}.", flush=False)
    print(f"Batch size:             {args.batch_size}.", flush=False)
    print(f"Max number of tokens:   {args.max_num_tokens}.", flush=False)
    print(f"Gradient acc. steps:    {args.gradient_accumulation_steps}.", flush=False)
    print(f"Gradient checkpointing: {args.gradient_checkpointing}.", flush=False)
    print(f"Pre-train alpha:        {args.pretrain_alpha}.", flush=False)
    print(f"\n", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Ensure that the output folder do exist. If needed, create such folder on disk.
    # ------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    if not os.path.exists(args.output_folder):
        raise ValueError(f"Unable to create the output folder for the model: found {args.output_folder}.")

    # ------------------------------------------------------------------------------------------------------------------
    # Load the tokenizer and the model.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()

    tk = transformers.AutoTokenizer.from_pretrained(args.base_model)
    assert isinstance(tk, transformers.PreTrainedTokenizerBase)
    model = modeling.TrainBiEncoderModel.from_pretrained(args.base_model, device_map=args.device)
    assert isinstance(model, modeling.TrainBiEncoderModel)

    print(f"Model loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    print("", flush=True)

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

    print("", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Instantiate the training dataset.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    train_dataset = modeling.PretrainDataset(
        corpora_data=[p_corpus_data, r_corpus_data, p_queries_data, r_queries_data],
        corpora_samples_per_epoch=[len(p_corpus_data), len(r_corpus_data), len(p_queries_data), len(r_queries_data)]
    )

    print(f"Train dataset loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Instantiate the collator.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    collator = modeling.PretrainingCollator(
        tk,
        max_num_tokens=args.max_num_tokens,
        mlm_probability=args.mlm_probability,
        mask_replace_probability=args.mask_replace_probability,
        random_replace_probability=args.random_replace_probability
    )

    print(f"Collator loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Instantiate the training arguments.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()

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
        max_grad_norm=1.0,

        # Logging and monitoring training.
        logging_strategy="no",

        # Checkpointing and saving.
        save_strategy="no",
        remove_unused_columns = False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Instantiate the trainer.
    # ------------------------------------------------------------------------------------------------------------------
    trainer = modeling.PretrainingTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        callbacks=[train_dataset],
        data_collator=collator,
        pretrain_alpha=args.pretrain_alpha,
        logging_filename=args.logging_filename
    )

    print(f"Trainer loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
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

    tk.save_pretrained(args.output_folder)
    try:
        model.save_pretrained(args.output_folder, safe_serialization=True)
    except:
        try:
            model.save_pretrained(args.output_folder, safe_serialization=False)
            print("Unable to save the model: saved with \"safe_serialization=False\".", flush=True)
        except:
            print("Unable to save the model: not saved.", flush=True)

    print("\nDone!\n", flush=True)


if __name__ == '__main__':
    main()
