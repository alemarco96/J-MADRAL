import transformers


class TextTokenizer:
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 field: str = "text",
                 max_num_tokens: int | None = 128):
        self.tokenizer = tokenizer
        self.field = field
        self.max_num_tokens = max_num_tokens

    def __call__(self, batch_data):
        return self.tokenizer(batch_data[self.field], add_special_tokens=True, truncation=True, padding=False,
                              max_length=self.max_num_tokens, return_special_tokens_mask=True)
