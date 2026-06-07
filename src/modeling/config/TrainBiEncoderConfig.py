import transformers


class TrainBiEncoderConfig(transformers.PretrainedConfig):
    model_type = "TrainBiEncoderModel"

    def __init__(self, config=None, **kwargs):
        q_config_data = {}
        d_config_data = {}
        other_data = {}

        if config is not None:
            for key1 in config:
                if key1 == "q_config":
                    for key2, value2 in getattr(config, "q_config"):
                        q_config_data[key2] = value2
                elif key1 == "d_config":
                    for key2, value2 in getattr(config, "d_config"):
                        d_config_data[key2] = value2
                else:
                    other_data[key1] = getattr(config, key1)

        if kwargs is not None:
            for key1, value1 in kwargs.items():
                if key1 == "q_config":
                    for key2, value2 in value1.items():
                        q_config_data[key2] = value2
                elif key1 == "d_config":
                    for key2, value2 in value1.items():
                        d_config_data[key2] = value2
                else:
                    other_data[key1] = value1

        super(TrainBiEncoderConfig, self).__init__(**other_data)
        self.q_config = transformers.PretrainedConfig(**q_config_data)
        self.d_config = transformers.PretrainedConfig(**d_config_data)

    @staticmethod
    def build_from_configs(q_config, d_config):
        result = TrainBiEncoderConfig()
        result.q_config = q_config
        result.d_config = d_config
        return result
