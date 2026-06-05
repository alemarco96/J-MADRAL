from modeling.generic.EncoderModule import EncoderModule
from modeling.generic.ModelInput import ModelInput
from modeling.generic.ModelOutput import ModelOutput
import transformers


class ModernBertEncoder(EncoderModule):
    tokens_embedding_weight_name: str | None = "encoder.embeddings.tok_embeddings.weight"

    def __init__(self, config, *args, **kwargs):
        super(ModernBertEncoder, self).__init__()

        # Used to prevent a crash in self.encoder.forward().
        if "reference_compile" not in config or config.reference_compile is None:
            config.reference_compile = False

        self.encoder = transformers.ModernBertModel(config)

    def forward(self, encoder_input: ModelInput) -> ModelOutput:
        # Do inference with the encoder model.
        encoder_output = self.encoder(input_ids=encoder_input.input_ids,
                                      attention_mask=encoder_input.attention_mask,
                                      output_attentions=encoder_input.output_attentions,
                                      output_hidden_states=encoder_input.output_hidden_states)

        # Return the encoder output.
        return ModelOutput.from_base_output(encoder_output)
