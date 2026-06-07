import argparse
import datasets
datasets.utils.logging.disable_progress_bar()
import os
import modeling
import time
import torch
import transformers


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument("--corpus_filename",
                      dest="corpus_filename",
                      type=str,
                      required=True)
    args.add_argument("--id_field",
                      dest="id_field",
                      type=str,
                      default="id",
                      required=False)
    args.add_argument("--text_field1",
                      dest="text_field1",
                      type=str,
                      required=True)
    args.add_argument("--text_field2",
                      dest="text_field2",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--text_field3",
                      dest="text_field",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--text_field4",
                      dest="text_field",
                      type=str,
                      default=None,
                      required=False)
    args.add_argument("--sort_by_length",
                      dest="sort_by_length",
                      action="store_true",
                      default=True,
                      required=False)
    args.add_argument("--no_sort_by_length",
                      dest="sort_by_length",
                      action="store_false",
                      required=False)
    args.add_argument("--index_folder",
                      dest="index_folder",
                      type=str,
                      required=True)
    args.add_argument("--model",
                      dest="model",
                      type=str,
                      required=True)
    args.add_argument("--custom_model",
                      dest="custom_model",
                      action="store_true",
                      default=True,
                      required=False)
    args.add_argument("--no_custom_model",
                      dest="custom_model",
                      action="store_false",
                      required=False)
    args.add_argument("--device",
                      dest="device",
                      type=str,
                      default="cuda",
                      required=False)
    args.add_argument("--chunk_size",
                      dest="chunk_size",
                      type=int,
                      default=4096,
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
    args.text_fields = [v for v in [args.text_field1, args.text_field2, args.text_field3, args.text_field4]
                        if v is not None]
    args.text_field = "__contents"
    args.len_field = "__len"

    # ------------------------------------------------------------------------------------------------------------------
    # Load the corpus from disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    corpus_data = datasets.load_dataset("json", data_files=args.corpus_filename, num_proc=4)["train"]
    if args.id_field not in corpus_data.column_names:
        raise ValueError(f"Invalid id_field: {args.id_field} not found in the corpus data.")
    if len(args.text_fields) <= 0:
        raise ValueError(f"Invalid text_fields: {args.text_fields} is empty.")
    if any(v not in corpus_data.column_names for v in args.text_fields):
        raise ValueError(f"Invalid text_fields: {args.text_fields} not found in the corpus data.")

    print(f"Corpus loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Load the tokenizer and model from disk.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if args.custom_model:
        model = modeling.BiEncoderModel.from_pretrained(args.model, device_map=args.device, dtype=torch.float32)
        assert isinstance(model, modeling.BiEncoderModel)
    else:
        model = transformers.AutoModel.from_pretrained(args.model, device_map=args.device, dtype=torch.float32)
    model.eval()

    print(f"Model loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Process the corpus, and index it.
    # ------------------------------------------------------------------------------------------------------------------
    def concat_text(_batch):
        return {args.text_field: "\n".join(_batch[k] for k in args.text_fields).strip()}

    def add_text_length(_batch):
        return {args.len_field: [len(x) for x in _batch[args.text_field]]}

    with open(os.path.join(args.index_folder, "docid"), "wt", encoding="utf-8") as fo_d, \
         open(os.path.join(args.index_folder, "index"), "wb") as fo_i:
        # --------------------------------------------------------------------------------------------------------------
        # Write the FAISS header (45 bytes) in the index file.
        # --------------------------------------------------------------------------------------------------------------
        # "fourcc": 'IxFI' magic number (4 bytes).
        fo_i.write((0x49784649).to_bytes(length=4, byteorder="little", signed=False))
        # "d": Embedding size (4 bytes).
        fo_i.write(model.config.hidden_size.to_bytes(length=4, byteorder="little", signed=False))
        # "ntotal": Number of stored vectors (8 bytes).
        fo_i.write(len(corpus_data).to_bytes(length=8, byteorder="little", signed=False))
        # "dummy1": 1<<20 (8 bytes).
        fo_i.write((1<<20).to_bytes(length=8, byteorder="little", signed=False))
        # "dummy2": 1<<20 (8 bytes).
        fo_i.write((1<<20).to_bytes(length=8, byteorder="little", signed=False))
        # "is_trained": 1 (1 byte).
        fo_i.write((1).to_bytes(length=1, byteorder="little", signed=False))
        # "metric_type": 0 (4 bytes).
        fo_i.write((0).to_bytes(length=4, byteorder="little", signed=False))
        # "codes_length": Number of bytes for data (8 bytes).
        fo_i.write((len(corpus_data) * model.config.hidden_size).to_bytes(length=8, byteorder="little", signed=False))
        fo_i.flush()

        # --------------------------------------------------------------------------------------------------------------
        # Process the data in smaller chunks.
        # --------------------------------------------------------------------------------------------------------------
        last = 0
        ctr = 0
        st = time.time_ns()
        for chunk_data in corpus_data.batch(batch_size=args.chunk_size):
            chunk_data = datasets.Dataset.from_dict(chunk_data)

            # Concatenate the textual data.
            chunk_data = chunk_data.map(concat_text, batched=True, num_proc=4)
            chunk_data = chunk_data.remove_columns(args.text_fields)

            # Sort the chunk data by size, in order to minimize padding.
            if args.sort_by_length:
                chunk_data = chunk_data.map(add_text_length, batched=True, num_proc=4)
                chunk_data = chunk_data.sort(args.len_field)
                chunk_data = chunk_data.remove_columns([args.len_field])

            # Process the chunk in smaller batches.
            for batch_data in chunk_data.batch(batch_size=args.batch_size):
                # Tokenize the batch, and move the tensors to the model device.
                tokens = tokenizer(batch_data[args.text_field],
                                   add_special_tokens=True,
                                   truncation=True,
                                   padding=True,
                                   max_length=args.max_num_tokens,
                                   return_tensors="pt").to(model.device)

                # Compute the embeddings with the model.
                with torch.no_grad():
                    if args.custom_model:
                        embeddings = model.forward(input_ids=tokens.input_ids,
                                                   attention_mask=tokens.attention_mask,
                                                   token_type_ids=tokens.token_type_ids,
                                                   output_attentions=False,
                                                   output_hidden_states=False,
                                                   output_logits=False,
                                                   return_dict=True).pooler_output
                    else:
                        embeddings = model.forward(input_ids=tokens.input_ids,
                                                   attention_mask=tokens.attention_mask,
                                                   token_type_ids=tokens.token_type_ids,
                                                   output_attentions=False,
                                                   output_hidden_states=False,
                                                   output_logits=False,
                                                   return_dict=True).last_hidden_state[:, 0, :]
                del tokens

                # Write the id of the documents.
                batch_data[args.id_field] = [str(x) for x in batch_data[args.id_field]]
                print("\n".join(batch_data[args.id_field]), end="\n", file=fo_d, flush=False)

                # Write the embeddings in the index.
                embeddings = embeddings.detach().contiguous().cpu().to(dtype=torch.float32).numpy()
                embeddings = embeddings.astype(embeddings.dtype.newbyteorder("little"))
                fo_i.write(embeddings.tobytes())
                del embeddings

                ctr += len(batch_data[args.text_field])
                if (ctr - last) >= 10000:
                    print(f"Processed {ctr} / {len(corpus_data)} = {ctr * 100 / len(corpus_data):.2f}% "
                          f"documents in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
                    last = ctr
            try:
                del batch_data
            except UnboundLocalError:
                pass

            fo_d.flush()
            fo_i.flush()
        try:
            del chunk_data
        except UnboundLocalError:
            pass

        fo_d.flush()
        fo_i.flush()
    del fo_d, fo_i

    if (ctr - last) > 0:
        print(f"Processed {ctr} / {len(corpus_data)} = {ctr * 100 / len(corpus_data):.2f}% "
              f"documents in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del ctr, last

    print(f"Index completed in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    print("\nDone!\n", flush=True)


if __name__ == '__main__':
    main()
