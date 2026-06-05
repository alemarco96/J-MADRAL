import argparse
import modeling
import os
import transformers
import utils


def parse_args() -> argparse.Namespace:
    args = argparse.Namespace()
    args.device = "cuda:0"

    args.base_model_name = f"{os.environ['ESCISOTA_DATA']}{os.sep}models{os.sep}bi_encoders{os.sep}init{os.sep}search-bibert-tdb_bert_init"
    # args.base_model_name = f"{os.environ['ESCISOTA_DATA']}{os.sep}models{os.sep}bi_encoders{os.sep}init{os.sep}search-mtbert-tdb_bert_init"
    # args.base_model_name = f"{os.environ['ESCISOTA_DATA']}{os.sep}models{os.sep}bi_encoders{os.sep}init{os.sep}search-madral-tdb_bert_init"

    args.pretrain_alpha = 0.0  # 0.1
    args.mlm_probability = 0.15
    args.mask_replace_probability = 0.80
    args.random_replace_probability = 0.10
    args.learning_rate = 1e-4
    args.num_train_epochs = 1
    args.batch_size = 32
    args.gradient_accumulation_steps = 1
    args.max_num_tokens = 128

    args.s_corpus_filename = f"{os.environ['ESCISOTA_DATA']}{os.sep}datasets{os.sep}AmazonESCI_Reviews{os.sep}final{os.sep}corpus{os.sep}s{os.sep}search_s_corpus.jsonl"
    args.s_queries_filename = f"{os.environ['ESCISOTA_DATA']}{os.sep}datasets{os.sep}AmazonESCI_Reviews{os.sep}final{os.sep}queries{os.sep}s{os.sep}search_s_train_queries.jsonl"

    args.final_model_name = "search-bibert-tdb_bert_pretrained"
    # args.final_model_name = "search-mtbert-tdb_bert_pretrained"
    # args.final_model_name = "search-madral-tdb_bert_pretrained"
    args.output_folder = f"{os.environ['ESCISOTA_DATA']}{os.sep}models{os.sep}bi_encoders{os.sep}pretrained{os.sep}{args.final_model_name}"
    args.logging_filename = f"{args.output_folder}{os.sep}loss_logging.tsv"
    args.train_output_folder = f"{args.output_folder}"
    return args


def main():
    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Log the main train parameters.
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n"
          f"Base model:          {args.base_model_name}.", flush=False)
    print(f"Output folder:       {args.output_folder}.", flush=False)
    print(f"Train output folder: {args.train_output_folder}.", flush=False)
    print(f"Learning rate:       {args.learning_rate}.", flush=False)
    print(f"Num train epochs:    {args.num_train_epochs}.", flush=False)
    print(f"Batch size:          {args.batch_size}.\n\n", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Ensure that the output folders do exist. If needed, create such folders on disk.
    # ------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(args.train_output_folder):
        os.mkdir(args.train_output_folder)

    if not os.path.exists(args.train_output_folder):
        raise ValueError(f"Unable to create the output folder for the train model: found {args.train_output_folder}.")

    # ------------------------------------------------------------------------------------------------------------------
    # Load the tokenizer and the model.
    # ------------------------------------------------------------------------------------------------------------------
    tk = transformers.AutoTokenizer.from_pretrained(args.base_model_name)
    assert isinstance(tk, transformers.PreTrainedTokenizerBase)
    model = modeling.TrainDoubleBiEncoderModel.from_pretrained(args.base_model_name, device_map=args.device)
    assert isinstance(model, modeling.TrainDoubleBiEncoderModel)

    print("Model loaded!\n", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Read the search corpus data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    s_corpus_data = utils.read_corpus(args.s_corpus_filename, convert_doc_id_to_int=False)
    print("S corpus loaded!", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Read the search queries data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    s_queries_data = utils.read_corpus(args.s_queries_filename, convert_doc_id_to_int=False)
    print("S queries loaded!", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Instantiate the training dataset.
    # ------------------------------------------------------------------------------------------------------------------
    train_dataset = modeling.MultiPretrainDataset(
        corpora_data=[s_corpus_data, s_queries_data],
        corpora_samples_per_epoch=[len(s_corpus_data), len(s_queries_data)]
    )
    print("Train dataset loaded!", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Instantiate the collator.
    # ------------------------------------------------------------------------------------------------------------------
    collator = modeling.JointPretrainingCollator(tk,
                                                 max_num_tokens=args.max_num_tokens,
                                                 mlm_probability=args.mlm_probability,
                                                 mask_replace_probability=args.mask_replace_probability,
                                                 random_replace_probability=args.random_replace_probability)
    print("Collator loaded!", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Instantiate the training arguments.
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
        max_grad_norm=1.0,

        # Logging and monitoring training.
        logging_strategy="steps",
        logging_steps=1000,
        include_num_input_tokens_seen="no",

        # Checkpointing and saving.
        save_strategy="no",  # "epoch",
        run_name=args.final_model_name,

        remove_unused_columns = False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Instantiate the trainer.
    # ------------------------------------------------------------------------------------------------------------------
    trainer = modeling.MadralPretrainingTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        callbacks=[train_dataset],
        data_collator=collator,
        pretrain_alpha=args.pretrain_alpha,
        logging_filename=args.logging_filename
    )
    print("Trainer loaded!\n", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Perform the training.
    # ------------------------------------------------------------------------------------------------------------------
    trainer.train()

    # ------------------------------------------------------------------------------------------------------------------
    # Save the resulting model to disk.
    # ------------------------------------------------------------------------------------------------------------------
    model.untie_all()

    tk.save_pretrained(args.train_output_folder)
    try:
        model.save_pretrained(args.train_output_folder, safe_serialization=True)
    except:
        try:
            model.save_pretrained(args.train_output_folder, safe_serialization=False)
            print("Unable to save the train model: saved with \"safe_serialization=False\".", flush=True)
        except:
            print("Unable to save the train model: not saved.", flush=True)

    print("\nDone!\n", flush=True)


if __name__ == '__main__':
    main()
