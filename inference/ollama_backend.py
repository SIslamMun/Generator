"""Ollama `/api/generate` backend with `raw=true`.

We render the prompt in Python with `tokenizer.apply_chat_template(...)` and
then hand Ollama the final string — skipping Ollama's Go template entirely,
which has ~2-token drift vs. the HF tokenizer. At 270M that drift breaks
tool-call emission.

Sampling defaults match Unsloth's published FunctionGemma recipe:
    temperature=1.0, top_p=0.95, top_k=64
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error


class OllamaBackend:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        num_predict: int = 512,
        num_ctx: int = 8192,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_predict = num_predict
        self.num_ctx = num_ctx

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode())
        return body.get("response", "")
