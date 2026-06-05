import transformers


class AttemptFinetuningTokenizer:
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 aspects_tokens: list[str],
                 content_token: str,
                 q_aspects_labels_maps: list[dict[str, int]],
                 d_aspects_labels_maps: list[dict[str, int]],
                 max_num_tokens: int | None = 128,
                 query_field_prefix: str = "query",
                 positive_field_prefix: str = "positive",
                 negative_field_prefix: str = "negative",
                 text_field_suffix: str = "_text",
                 aspects_field_suffix: str = "_aspects"
                 ):
        self.tokenizer = tokenizer
        self.max_num_tokens = max_num_tokens
        # The token associated with each aspect (i.e., "[BRAND]", "[COLOR], "[CATEGORY_1]", ...).
        self.aspects_tokens = aspects_tokens
        self.content_token = content_token
        # Invert the aspects labels map: now id->aspect text.
        self.q_aspects_labels_maps = [{v2: k2 for k2, v2 in v1.items()} for v1 in q_aspects_labels_maps]
        self.d_aspects_labels_maps = [{v2: k2 for k2, v2 in v1.items()} for v1 in d_aspects_labels_maps]
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

        # Determine which are the fields containing the information needed.
        q_fields_aspects = [f"{self.query_field_prefix}{self.aspects_field_suffix}"]
        d_fields_aspects = [f"{self.positive_field_prefix}{i}{self.aspects_field_suffix}"
                            for i in num_positives] + \
                           [f"{self.negative_field_prefix}{i}{self.aspects_field_suffix}"
                            for i in num_negatives]

        qd_fields_text = [f"{self.query_field_prefix}{self.text_field_suffix}"] + \
                         [f"{self.positive_field_prefix}{i}{self.text_field_suffix}" for i in num_positives] + \
                         [f"{self.negative_field_prefix}{i}{self.text_field_suffix}" for i in num_negatives]

        fields_name = [self.query_field_prefix] + \
                      [f"{self.positive_field_prefix}{i}" for i in num_positives] + \
                      [f"{self.negative_field_prefix}{i}" for i in num_negatives]
        del num_positives, num_negatives

        # Obtain the textual representation of each query and document.

        # [A1] a1_text [A2] a2_text ... [Ak] ak_text
        # Concatenate all aspects tokens followed by (if present) the associated aspect text.
        aspects_texts = [[" ".join([f"{self.aspects_tokens[i2]} {self.q_aspects_labels_maps[i2].get(v2, '') }"
                                    for i2, v2 in enumerate(v1)]) for v1 in batch_data[k0]]
                         for k0 in q_fields_aspects] + \
                        [[" ".join([f"{self.aspects_tokens[i2]} {self.d_aspects_labels_maps[i2].get(v2, '') }"
                                    for i2, v2 in enumerate(v1)]) for v1 in batch_data[k0]]
                         for k0 in d_fields_aspects]
        del q_fields_aspects, d_fields_aspects

        # The content text, preceded by the [CONTENT] token.
        content_texts = [[f"{self.content_token} {v1}" for v1 in batch_data[k0]] for k0 in qd_fields_text]
        del qd_fields_text

        # Horizontally stack the aspects and content texts.
        full_texts = [[f"{a2} {b2}" for a2, b2 in zip(a1, b1)] for a1, b1 in zip(aspects_texts, content_texts)]
        del aspects_texts, content_texts

        # Tokenize each field.
        for field, text in zip(fields_name, full_texts):
            # Tokenize the current data.
            batch_field = self.tokenizer(text,
                                         add_special_tokens=True,
                                         truncation=True,
                                         padding=False,
                                         max_length=self.max_num_tokens)

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
            del field, text
        except UnboundLocalError:
            pass
        del fields_name, full_texts

        return result
