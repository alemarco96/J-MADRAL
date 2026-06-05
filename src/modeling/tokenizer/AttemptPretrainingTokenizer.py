import transformers


class AttemptPretrainingTokenizer:
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 aspects_tokens: list[str],
                 content_token: str,
                 aspects_labels_maps: list[dict[str, int]],
                 content_field: str = "text",
                 aspects_field: str = "categories",
                 max_num_tokens: int | None = 128,
                 tokenize_also_content_alone: bool = True):
        self.tokenizer = tokenizer
        # The token associated with each aspect (i.e., "[BRAND]", "[COLOR], "[CATEGORY_1]", ...).
        self.aspects_tokens = aspects_tokens
        self.content_token = content_token
        # Invert the aspects labels map: now id->aspect text.
        self.aspects_labels_maps = [{v2: k2 for k2, v2 in v1.items()} for v1 in aspects_labels_maps]
        self.content_field = content_field
        self.aspects_field = aspects_field
        self.max_num_tokens = max_num_tokens
        self.tokenize_also_content_alone = tokenize_also_content_alone

    def __call__(self, batch_data):
        # [A1] a1_text [A2] a2_text ... [Ak] ak_text
        # Concatenate all aspects tokens followed by (if present) the associated aspect text.
        aspects_text = [" ".join([f"{self.aspects_tokens[i2]} {self.aspects_labels_maps[i2].get(v2, '') }"
                                  for i2, v2 in enumerate(v1)])
                        for v1 in batch_data[self.aspects_field]]

        # [CLS] [A1] a1_text [A2] a2_text ... [Ak] ak_text [SEP] [C] content_text [SEP]
        full_result = self.tokenizer(aspects_text,
                                     text_pair=[f"{self.content_token} {v1}" for v1 in batch_data[self.content_field]],
                                     add_special_tokens=True,
                                     truncation=True,
                                     padding=False,
                                     max_length=self.max_num_tokens,
                                     return_special_tokens_mask=False)

        # Recompute the special tokens map, due to a bug in the tokenizer.
        full_special_tokens_map = [self.tokenizer.get_special_tokens_mask(
            v, already_has_special_tokens=True) for v in full_result.input_ids]

        # [CLS] content_text [SEP]
        if self.tokenize_also_content_alone:
            content_result = self.tokenizer([f"{self.content_token} {v1}" for v1 in batch_data[self.content_field]],
                                            add_special_tokens=True,
                                            truncation=True,
                                            padding=False,
                                            max_length=self.max_num_tokens,
                                            return_special_tokens_mask=False)

            # Recompute the special tokens map, due to a bug in the tokenizer.
            content_special_tokens_map = [self.tokenizer.get_special_tokens_mask(
                v, already_has_special_tokens=True) for v in content_result.input_ids]
        else:
            content_result = None
            content_special_tokens_map = None

        # Return the tokenization results.
        if self.tokenize_also_content_alone:
            assert content_result is not None
            assert content_special_tokens_map is not None

            if "token_type_ids" in full_result.keys():
                return {
                    "full_input_ids": full_result.input_ids,
                    "full_attention_mask": full_result.attention_mask,
                    "full_token_type_ids": full_result.token_type_ids,
                    "full_special_tokens_mask": full_special_tokens_map,
                    "content_input_ids": content_result.input_ids,
                    "content_attention_mask": content_result.attention_mask,
                    "content_token_type_ids": content_result.token_type_ids,
                    "content_special_tokens_mask": content_special_tokens_map
                }
            else:
                return {
                    "full_input_ids": full_result.input_ids,
                    "full_attention_mask": full_result.attention_mask,
                    "full_special_tokens_mask": full_special_tokens_map,
                    "content_input_ids": content_result.input_ids,
                    "content_attention_mask": content_result.attention_mask,
                    "content_special_tokens_mask": content_special_tokens_map
                }
        else:
            if "token_type_ids" in full_result.keys():
                return {
                    "full_input_ids": full_result.input_ids,
                    "full_attention_mask": full_result.attention_mask,
                    "full_token_type_ids": full_result.token_type_ids,
                    "full_special_tokens_mask": full_special_tokens_map
                }
            else:
                return {
                    "full_input_ids": full_result.input_ids,
                    "full_attention_mask": full_result.attention_mask,
                    "full_special_tokens_mask": full_special_tokens_map
                }
