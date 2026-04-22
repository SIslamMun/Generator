"""HuggingFace transformers backend — useful inside the Colab notebook right
after training, before you've exported GGUF to Ollama.

Importing at module level is deferred so the rest of the inference package
still imports on machines without torch.
"""

from __future__ import annotations


class HFBackend:
    def __init__(
        self,
        model,
        tokenizer,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
        )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=False)
