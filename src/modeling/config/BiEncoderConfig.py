import transformers


class BiEncoderConfig(transformers.PretrainedConfig):
    model_type = "BiEncoderModel"

    def __init__(self,
                 encoder_module: str | None = None,
                 pooler_module: str | None = None,
                 aen_module: str | None = None,
                 afn_module: str | None = None,
                 num_aspects: int | None = None,
                 aspects_size: list[int] | None = None,
                 **kwargs):
        super(BiEncoderConfig, self).__init__(**kwargs)

        self.encoder_module = encoder_module
        self.pooler_module = pooler_module
        self.aen_module = aen_module
        self.afn_module = afn_module
        self.num_aspects = num_aspects
        self.aspects_size = aspects_size
