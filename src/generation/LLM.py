import torch
import transformers


class LLM:
    def __init__(self, model: str, device: str = "cuda"):
        assert model is not None
        assert isinstance(model, str)

        assert device is not None
        assert isinstance(device, str)
        device = device.lower()
        assert device in {"cpu", "cuda"} or device.startswith("cuda:")

        # Deploy this model on the selected device.
        self.device = torch.device(device)

        # Load the tokenizer used by the model from disk.
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)

        # Load the model from disk and store it on GPU if available, otherwise on CPU.
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model,
                                                                       torch_dtype=torch.bfloat16,
                                                                       device_map=self.device)

    def __call__(self, messages: list[dict[str, str]], **kwargs) -> str:
        assert messages is not None
        assert isinstance(messages, list)
        assert all(isinstance(x, dict) for x in messages)
        assert all(all(isinstance(k, str) & isinstance(v, str) for k, v in x.items()) for x in messages)

        prompt = self.tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True,
                                                    enable_thinking=False)
        model_input = self.tokenizer(prompt, truncation=True, return_tensors="pt").to(self.device)
        del prompt

        model_output = self.model.generate(**model_input, **kwargs)

        # Skip the first tokens, which are a repetition of the input prompt.
        model_output = model_output[0, model_input.input_ids.shape[1]:]
        del model_input

        return self.tokenizer.decode(model_output, skip_special_tokens=True, spaces_between_special_tokens=False)
