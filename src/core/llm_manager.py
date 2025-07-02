from llama_cpp import Llama


class LLMManager:
    """
    Offline LLM manager using llama-cpp-python. Loads a local Llama model and
    generates responses.
    """
    def __init__(self, model_path=None):
        # TODO: Set the correct path to your local Llama model file (.gguf)
        self.model_path = model_path or "models/llama-2-7b-chat.gguf"
        self.llm = None
        self._load_model()

    def _load_model(self):
        try:
            self.llm = Llama(model_path=self.model_path, n_ctx=2048)
        except Exception as e:
            print(f"[LLMManager] Failed to load Llama model: {e}")
            self.llm = None

    def chat(self, prompt, history=None, max_tokens=256):
        if not self.llm:
            return "[LLM not available]"
        # TODO: Use history for multi-turn conversation
        full_prompt = prompt
        response = self.llm(full_prompt, max_tokens=max_tokens)
        return response["choices"][0]["text"].strip() 