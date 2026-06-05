import argparse
import modeling
import os
import tokenizers.processors
import torch
import transformers


def parse_args() -> argparse.Namespace:
    args = argparse.Namespace()

    args.base_model_name = "google-bert/bert-base-uncased"
    # args.base_model_name = "answerdotai/ModernBERT-base"

    args.custom_tokenizer = False
    # args.custom_tokenizer = True
    # args.new_tokens_are_special = False
    args.new_tokens_are_special = True
    args.add_new_tokens_at_start = False
    # args.add_new_tokens_at_start = True
    args.new_tokens = ["[BRAND]", "[COLOR]", "[CATEGORY_1]", "[CATEGORY_2]", "[CATEGORY_3]", "[CONTENT]"]

    # Aspects: BRAND (5382), COLOR (2358), CAT_1 (58), CAT_2 (454), CAT_3 (2302).
    # args.aspects_size = None
    # args.aspects_size = [5382, 2358, 58, 454, 2302]
    args.aspects_size = [
        # Task
        2,
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
        # Review Month
        12,
        # Review Rating
        2,
        # Review Year
        4
    ]

    args.encoder_module = "BERT"
    # args.encoder_module = "ModernBERT"
    args.q_pooler_module = "Linear"
    args.d_pooler_module = "Linear"
    # args.q_pooler_module = "Aspects"
    # args.d_pooler_module = "Aspects"
    args.q_aen_module = None
    args.d_aen_module = None
    # args.q_aen_module = "MtBertAEN"
    # args.d_aen_module = "MtBertAEN"
    # args.q_aen_module = "MadralAEN"
    # args.d_aen_module = "MadralAEN"
    args.q_afn_module = None
    args.d_afn_module = None
    # args.q_afn_module = "AspectsGatingAFN"
    # args.d_afn_module = "AspectsGatingAFN"

    # args.output_folder = f"{os.environ['ESCISOTA_DATA']}{os.sep}models{os.sep}bi_encoders{os.sep}init{os.sep}rec_bibert-tdb_bert_init"
    # args.output_folder = f"{os.environ['ESCISOTA_DATA']}{os.sep}models{os.sep}bi_encoders{os.sep}init{os.sep}rec-mtbert-tdb_bert_init"
    # args.output_folder = f"{os.environ['ESCISOTA_DATA']}{os.sep}models{os.sep}bi_encoders{os.sep}init{os.sep}rec-madral-tdb_bert_init"

    args.output_folder = f"{os.environ['ESCISOTA_DATA']}{os.sep}models{os.sep}bi_encoders{os.sep}init{os.sep}search_bibert-tdb_bert_init"
    # args.output_folder = f"{os.environ['ESCISOTA_DATA']}{os.sep}models{os.sep}bi_encoders{os.sep}init{os.sep}search-mtbert-tdb_bert_init"
    # args.output_folder = f"{os.environ['ESCISOTA_DATA']}{os.sep}models{os.sep}bi_encoders{os.sep}init{os.sep}search-madral-tdb_bert_init"

    # args.output_folder = f"{os.environ['ESCISOTA_DATA']}{os.sep}models{os.sep}bi_encoders{os.sep}init{os.sep}joint-bibert-tdb_bert_init"
    # args.output_folder = f"{os.environ['ESCISOTA_DATA']}{os.sep}models{os.sep}bi_encoders{os.sep}init{os.sep}joint-mtbert-tdb_bert_init"
    # args.output_folder = f"{os.environ['ESCISOTA_DATA']}{os.sep}models{os.sep}bi_encoders{os.sep}init{os.sep}joint-madral-tdb_bert_init"
    return args


def main():
    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Load the tokenizer and initialize the base model.
    # ------------------------------------------------------------------------------------------------------------------
    tk = transformers.AutoTokenizer.from_pretrained(args.base_model_name)
    assert isinstance(tk, transformers.PreTrainedTokenizerBase)
    base = transformers.AutoModelForMaskedLM.from_pretrained(args.base_model_name, device_map="cpu")
    assert isinstance(base, transformers.PreTrainedModel)

    # ------------------------------------------------------------------------------------------------------------------
    # Customize the tokenizer and the model with the new tokens.
    # ------------------------------------------------------------------------------------------------------------------
    if args.custom_tokenizer:
        # Extract the tokens composing the text associated with each new custom token.
        new_tokens_tokens = [tk(v1, add_special_tokens=False).input_ids for v1 in args.new_tokens]

        # Compute the initial embedding for each new custom token, as mean pooling of the tokens embedding
        # of every pre-existing token composing the text associated with it.
        if args.encoder_module == "BERT":
            with torch.no_grad():
                new_tokens_embedding = [torch.mean(torch.stack([base.bert.embeddings.word_embeddings.weight[v2]
                                                                for v2 in v1], dim=0), dim=0)
                                        for v1 in new_tokens_tokens]
        elif args.encoder_module == "ModernBERT":
            with torch.no_grad():
                new_tokens_embedding = [torch.mean(torch.stack([base.model.embeddings.tok_embeddings.weight[v2]
                                                                for v2 in v1], dim=0), dim=0)
                                        for v1 in new_tokens_tokens]
        else:
            raise ValueError(f"Invalid args.encoder_module: found {args.encoder_module}.")
        del new_tokens_tokens

        # Add the custom tokens to the tokenizer.
        if args.new_tokens_are_special:
            num_new_tokens = tk.add_special_tokens({"additional_special_tokens": args.new_tokens})
        else:
            num_new_tokens = tk.add_tokens(args.new_tokens)
        assert num_new_tokens == len(args.new_tokens)
        del num_new_tokens

        # Obtain the id associated to each new token.
        new_token_ids = [tk.added_tokens_encoder[k] for k in args.new_tokens]

        # Resize the tokens embedding matrix.
        model_embeddings = base.resize_token_embeddings(len(tk))

        # Set the embedding for each new custom token.
        with torch.no_grad():
            for t_id, t_emb in zip(new_token_ids, new_tokens_embedding):
                model_embeddings.weight[t_id] = t_emb
            try:
                del t_id, t_emb
            except UnboundLocalError:
                pass

        # If required, modify the tokenizer behavior in order to automatically add the new custom tokens at the start.
        if args.add_new_tokens_at_start:
            tk._tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
                single=f"{tk.cls_token}:0 {' '.join(f'{t}:0' for t in args.new_tokens)} $A:0 {tk.sep_token}:0",
                pair=f"{tk.cls_token}:0 {' '.join(f'{t}:0' for t in args.new_tokens)} $A:0 "
                     f"{tk.sep_token}:0 $B:1 {tk.sep_token}:1",
                special_tokens=[(tk.cls_token, tk.cls_token_id), (tk.sep_token, tk.sep_token_id)] + \
                               [(k, v) for k, v in zip(args.new_tokens, new_token_ids)]
            )

    # ------------------------------------------------------------------------------------------------------------------
    # Create the new model.
    # ------------------------------------------------------------------------------------------------------------------
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

    config = modeling.TrainDoubleBiEncoderConfig.build_from_configs(q_config, d_config)
    model = modeling.TrainDoubleBiEncoderModel(config)
    assert isinstance(model, modeling.TrainDoubleBiEncoderModel)

    # ------------------------------------------------------------------------------------------------------------------
    # Translate the tensors name.
    # ------------------------------------------------------------------------------------------------------------------
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
