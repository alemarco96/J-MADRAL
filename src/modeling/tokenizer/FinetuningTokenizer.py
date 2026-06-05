import transformers


class FinetuningTokenizer:
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 max_num_tokens: int | None = 128,
                 query_field_prefix: str = "query",
                 positive_field_prefix: str = "positive",
                 negative_field_prefix: str = "negative",
                 text_field_suffix: str = "_text",
                 aspects_field_suffix: str = "_aspects"
                 ):
        self.tokenizer = tokenizer
        self.max_num_tokens = max_num_tokens
        self.query_field_prefix = query_field_prefix
        self.positive_field_prefix = positive_field_prefix
        self.negative_field_prefix = negative_field_prefix
        self.text_field_suffix = text_field_suffix
        self.aspects_field_suffix = aspects_field_suffix

    def __call__(self, batch_data):
        result = {}

        # Determine which are the positives and negatives in the current batch.
        num_positives = sorted({int(k[len(self.positive_field_prefix):-len(self.text_field_suffix)])
                                for k in batch_data.keys() if k.startswith(self.positive_field_prefix) and \
                                k.endswith(self.text_field_suffix)})
        num_negatives = sorted({int(k[len(self.negative_field_prefix):-len(self.text_field_suffix)])
                                for k in batch_data.keys() if k.startswith(self.negative_field_prefix) and \
                                k.endswith(self.text_field_suffix)})
        fields_to_tokenize = [f"{self.query_field_prefix}{self.text_field_suffix}"] + \
                             [f"{self.positive_field_prefix}{i}{self.text_field_suffix}" for i in num_positives] + \
                             [f"{self.negative_field_prefix}{i}{self.text_field_suffix}" for i in num_negatives]
        fields_aspects = [f"{self.query_field_prefix}{self.aspects_field_suffix}"] + \
                         [f"{self.positive_field_prefix}{i}{self.aspects_field_suffix}" for i in num_positives] + \
                         [f"{self.negative_field_prefix}{i}{self.aspects_field_suffix}" for i in num_negatives]
        del num_positives, num_negatives

        # Tokenize each field.
        for field in fields_to_tokenize:
            # Tokenize the current data.
            batch_field = self.tokenizer(batch_data[field],
                                         add_special_tokens=True,
                                         truncation=True,
                                         padding=False,
                                         max_length=self.max_num_tokens)

            # Extract the name field without the text suffix (i.e., "_text") at the end.
            field = field[:-len(self.text_field_suffix)]

            # Update the dictionary with the current result.
            if "token_type_ids" in batch_field:
                result.update({
                    f"{field}_input_ids": batch_field.input_ids,
                    f"{field}_attention_mask": batch_field.attention_mask,
                    f"{field}_token_type_ids": batch_field.token_type_ids
                })
            else:
                result.update({
                    f"{field}_input_ids": batch_field.input_ids,
                    f"{field}_attention_mask": batch_field.attention_mask
                })
            del batch_field
        try:
            del field
        except UnboundLocalError:
            pass

        # Add the documents aspects data.
        result.update({fa: batch_data[fa] for fa in fields_aspects})

        return result
