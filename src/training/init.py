import argparse
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
    args.add_argument("--architecture",
                      dest="architecture",
                      type=str,
                      required=True)
    args.add_argument("--task",
                      dest="task",
                      type=str,
                      required=True)
    args.add_argument("--output_folder",
                      dest="output_folder",
                      type=str,
                      required=True)

    return args.parse_args()


def main():
    NS_IN_S = 1000 * 1000 * 1000

    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize the fixed parameters.
    # ------------------------------------------------------------------------------------------------------------------
    args.aspects_size = [
        # Product Brand
        5171,
        # Product Category 0
        58,
        # Product Category 1
        444,
        # Product Category 2
        2294,
        # Product Category 3
        5241,
        # Product Color
        2208,
        # Product Popularity
        4,
        # Product Price
        13,
        # Review Helpful
        4,
        # Review Rating
        2
    ]

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize the model-specific parameters.
    # ------------------------------------------------------------------------------------------------------------------
    args.base_model = args.base_model.strip().lower()
    if args.base_model == "bert":
        args.encoder_module = "BERT"
        args.base_model_name = "google-bert/bert-base-uncased"
    elif args.base_model == "modernbert":
        args.encoder_module = "ModernBERT"
        args.base_model_name = "answerdotai/ModernBERT-base"
    else:
        raise ValueError(f"Invalid args.base_model: found {args.base_model}.")

    args.architecture = args.architecture.strip().lower()
    if args.architecture == "bibert":
        args.q_pooler_module = "Linear"
        args.d_pooler_module = "Linear"
        args.q_aen_module = None
        args.d_aen_module = None
        args.q_afn_module = None
        args.d_afn_module = None
    elif args.architecture == "madral":
        args.q_pooler_module = "Aspects"
        args.d_pooler_module = "Aspects"
        args.q_aen_module = "MadralAEN"
        args.d_aen_module = "MadralAEN"
        args.q_afn_module = "AspectsGatingAFN"
        args.d_afn_module = "AspectsGatingAFN"
    else:
        raise ValueError(f"Invalid args.architecture: found {args.architecture}.")

    args.task = args.task.strip().lower()
    if args.architecture == "bibert":
        args.aspects_size = None
    else:
        if args.task == "product":
            retain_aspects = [True, True, True, True, True, True, True, True, False, False]
            args.aspects_size = [v for r, v in zip(retain_aspects, args.aspects_size) if r]
            del retain_aspects
        elif args.task == "review":
            retain_aspects = [False, True, False, False, False, False, False, False, True, True]
            args.aspects_size = [v for r, v in zip(retain_aspects, args.aspects_size) if r]
            del retain_aspects
        elif args.task == "joint":
            retain_aspects = [True, True, True, True, True, True, True, True, True, True]
            args.aspects_size = [v for r, v in zip(retain_aspects, args.aspects_size) if r]
            del retain_aspects
        else:
            raise ValueError(f"Invalid args.task: found {args.task}.")

    # ------------------------------------------------------------------------------------------------------------------
    # Log the main train parameters.
    # ------------------------------------------------------------------------------------------------------------------
    print(f"\n"
          f"Base model:             {args.base_model}.", flush=False)
    print(f"Architecture:           {args.architecture}.", flush=False)
    print(f"Task:                   {args.task}.", flush=False)
    print("--------------------------------------------------", flush=False)
    print(f"Base model name:        {args.base_model_name}.", flush=False)
    print(f"Encoder module:         {args.encoder_module}.", flush=False)
    print(f"Query pooler module:    {args.q_pooler_module}.", flush=False)
    print(f"Query AEN module:       {args.q_aen_module}.", flush=False)
    print(f"Query AFN module:       {args.q_afn_module}.", flush=False)
    print(f"Doc pooler module:      {args.d_pooler_module}.", flush=False)
    print(f"Doc AEN module:         {args.d_aen_module}.", flush=False)
    print(f"Doc AFN module:         {args.d_afn_module}.", flush=False)
    print(f"Aspects size:           {args.aspects_size}.", flush=False)
    print("--------------------------------------------------", flush=False)
    print(f"Output folder:          {args.output_folder}.", flush=False)
    print(f"\n", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Ensure that the output folder do exist. If needed, create such folder on disk.
    # ------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    if not os.path.exists(args.output_folder):
        raise ValueError(f"Unable to create the output folder for the model: found {args.output_folder}.")

    # ------------------------------------------------------------------------------------------------------------------
    # Load the tokenizer and initialize the base model.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()

    tk = transformers.AutoTokenizer.from_pretrained(args.base_model)
    assert isinstance(tk, transformers.PreTrainedTokenizerBase)
    base = transformers.AutoModelForMaskedLM.from_pretrained(args.base_model_name, device_map="cpu")
    assert isinstance(base, transformers.PreTrainedModel)

    print(f"Base model loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    print("", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Create the new model.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()

    if args.encoder_module == "BERT":
        q_config = transformers.BertConfig.from_dict(base.config.to_dict())
        d_config = transformers.BertConfig.from_dict(base.config.to_dict())
    elif args.encoder_module == "ModernBERT":
        q_config = transformers.ModernBertConfig.from_dict(base.config.to_dict())
        d_config = transformers.ModernBertConfig.from_dict(base.config.to_dict())

        if "reference_compile" not in q_config or q_config.reference_compile is None:
            q_config.reference_compile = False
        if "reference_compile" not in d_config or d_config.reference_compile is None:
            d_config.reference_compile = False
    else:
        raise ValueError(f"Invalid args.encoder_module: found {args.encoder_module}.")

    q_config.encoder_module = args.encoder_module
    q_config.pooler_module = args.q_pooler_module
    q_config.aen_module = args.q_aen_module
    q_config.afn_module = args.q_afn_module
    q_config.num_aspects = len(args.aspects_size) if args.q_pooler_module == "Aspects" else None
    q_config.aspects_size = args.aspects_size if args.q_pooler_module == "Aspects" else None
    q_config.architectures = ["BiEncoderModel"]

    d_config.encoder_module = args.encoder_module
    d_config.pooler_module = args.d_pooler_module
    d_config.aen_module = args.d_aen_module
    d_config.afn_module = args.d_afn_module
    d_config.num_aspects = len(args.aspects_size) if args.d_pooler_module == "Aspects" else None
    d_config.aspects_size = args.aspects_size if args.d_pooler_module == "Aspects" else None
    d_config.architectures = ["BiEncoderModel"]

    config = modeling.TrainBiEncoderConfig.build_from_configs(q_config, d_config)
    model = modeling.TrainBiEncoderModel(config)
    assert isinstance(model, modeling.TrainBiEncoderModel)

    print(f"Model initialized in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Translate the tensors name.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()

    base_sd = base.state_dict()
    if args.encoder_module == "BERT":
        tensors_map = {f"q_model.encoder.encoder.{k[5:]}": v for k, v in base_sd.items()
                       if k.startswith("bert.")} | \
                      {f"d_model.encoder.encoder.{k[5:]}": v for k, v in base_sd.items()
                       if k.startswith("bert.")} | \
                      {f"d_mlm_head.mlm_head.{k[4:]}": v for k, v in base_sd.items() if k.startswith("cls.")}
    elif args.encoder_module == "ModernBERT":
        tensors_map = {f"q_model.encoder.encoder.{k[6:]}": v for k, v in base_sd.items()
                       if k.startswith("model.")} | \
                      {f"d_model.encoder.encoder.{k[6:]}": v for k, v in base_sd.items()
                       if k.startswith("model.")} | \
                      {f"d_mlm_head.{k}": v for k, v in base_sd.items() if not k.startswith("model.")}
    else:
        raise ValueError(f"Invalid args.encoder_module: found {args.encoder_module}.")

    missing, unexpected = model.load_state_dict(tensors_map, strict=False)
    assert len(unexpected) <= 0

    print(f"Model weights loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    print("", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Make sure the model does not have tied weights.
    # ------------------------------------------------------------------------------------------------------------------
    # Required to make model.save_pretrained() not crash due to tied weights.
    model.untie_all()

    # ------------------------------------------------------------------------------------------------------------------
    # Save the resulting model to disk.
    # ------------------------------------------------------------------------------------------------------------------
    tk.save_pretrained(args.output_folder)
    model.save_pretrained(args.output_folder, safe_serialization=True)

    print("\nDone!\n", flush=True)


if __name__ == '__main__':
    main()
