"""Free in-process Qwen adapter for the upstream chat-completion interface."""

from types import SimpleNamespace

MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"


class LocalChatClient:
    def __init__(self, loader=None):
        self._loader = loader
        self._model = None
        self.calls = 0
        self.failures = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def _load(self):
        if self._model is None:
            if self._loader is None:
                from llama_cpp import Llama

                self._loader = Llama.from_pretrained
            self._model = self._loader(
                repo_id=MODEL_REPO, filename=MODEL_FILE,
                n_ctx=8192, n_threads=4, verbose=False,
            )
        return self._model

    def create(self, *, messages, max_tokens=512, **kwargs):
        """Match OpenAI's response shape without using a model API."""
        try:
            model = self._load()
            output_limit = min(int(max_tokens), 1024)
            copied = [dict(message) for message in messages]
            # Leave room for chat-template tokens and the generated answer.
            budget = 8192 - output_limit - 256
            counts = [len(model.tokenize(m["content"].encode(), add_bos=False)) for m in copied]
            if sum(counts) > budget:
                longest = max(range(len(copied)), key=lambda i: counts[i])
                keep = budget - sum(counts) + counts[longest]
                if keep <= 0:
                    raise ValueError("System messages exceed the local context budget")
                tokens = model.tokenize(copied[longest]["content"].encode(), add_bos=False)
                copied[longest]["content"] = model.detokenize(tokens[:keep]).decode(errors="ignore")
            response = model.create_chat_completion(
                messages=copied, temperature=0, max_tokens=output_limit,
            )
            content = response["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise RuntimeError("Local model returned an empty answer")
            self.calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        except Exception:
            self.failures += 1
            raise
