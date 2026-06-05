import transformers


class FinetuningTokenizer:
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 max_num_tokens: int | None = 128,
                 query_field_prefix: str = "query",
                 document_field_prefix: str = "document",
                 text_field_suffix: str = "_text",
                 aspects_field_suffix: str = "_aspects"
                 ):
        self.tokenizer = tokenizer
        self.max_num_tokens = max_num_tokens
        self.query_field_prefix = query_field_prefix
        self.document_field_prefix = document_field_prefix
        self.text_field_suffix = text_field_suffix
        self.aspects_field_suffix = aspects_field_suffix

    def __call__(self, batch_data):
        result = {}

        # Determine which are the positives and negatives in the current batch.
        num_documents = sorted({int(k[len(self.document_field_prefix):-len(self.text_field_suffix)])
                                for k in batch_data.keys() if k.startswith(self.document_field_prefix) and \
                                k.endswith(self.text_field_suffix)})
        fields_to_tokenize = [(f"{self.query_field_prefix}{self.text_field_suffix}",
                               f"{self.document_field_prefix}{i}{self.text_field_suffix}") for i in num_documents]
        fields_aspects = [f"{self.query_field_prefix}{self.aspects_field_suffix}"] + \
                         [f"{self.document_field_prefix}{i}{self.aspects_field_suffix}" for i in num_documents]
        del num_documents

        # Tokenize each field.
        for q_field, d_field in fields_to_tokenize:
            # Tokenize the current data.
            batch_field = self.tokenizer(batch_data[q_field],
                                         text_pair=batch_data[d_field],
                                         add_special_tokens=True,
                                         truncation=True,
                                         padding=False,
                                         max_length=self.max_num_tokens)

            # Find the location of the first [SEP] token.
            sep_index = [next(i2 for i2, v2 in enumerate(v1) if v2 == self.tokenizer.sep_token_id)
                         for v1 in batch_field.input_ids]

            # Extract the name field without the text suffix (i.e., "_text") at the end.
            field = d_field[:-len(self.text_field_suffix)]

            # Update the dictionary with the current result.
            if "token_type_ids" in batch_field:
                result.update({
                    f"{field}_input_ids": batch_field.input_ids,
                    f"{field}_attention_mask": batch_field.attention_mask,
                    f"{field}_token_type_ids": batch_field.token_type_ids,
                    f"{field}_query_mask": [[v2 if 1 <= i2 <= idx else 0 for i2, v2 in enumerate(v1)]
                                            for v1, idx in zip(batch_field.attention_mask, sep_index)],
                    f"{field}_document_mask": [[v2 if i2 > idx else 0 for i2, v2 in enumerate(v1)]
                                               for v1, idx in zip(batch_field.attention_mask, sep_index)]
                })
            else:
                result.update({
                    f"{field}_input_ids": batch_field.input_ids,
                    f"{field}_attention_mask": batch_field.attention_mask,
                    f"{field}_query_mask": [[v2 if 1 <= i2 <= idx else 0 for i2, v2 in enumerate(v1)]
                                            for v1, idx in zip(batch_field.attention_mask, sep_index)],
                    f"{field}_document_mask": [[v2 if i2 > idx else 0 for i2, v2 in enumerate(v1)]
                                               for v1, idx in zip(batch_field.attention_mask, sep_index)]
                })
            del batch_field, sep_index, field
        try:
            del q_field, d_field
        except UnboundLocalError:
            pass

        # Add the documents aspects data.
        result.update({fa: batch_data[fa] for fa in fields_aspects})

        return result
