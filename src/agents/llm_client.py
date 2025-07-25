import os
import requests
from typing import List, Dict, Any

class LLMClient:
    """
    LLM client for command parsing and workflow suggestion.
    Supports OpenAI API and local Llama.cpp server.
    """
    def __init__(self, api_type="openai", api_key=None, endpoint=None):
        self.api_type = api_type
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.endpoint = endpoint or "http://localhost:8000/v1/completions"

    def llm_parse(self, command: str, context: Dict[str, Any], history: List[str] = None) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(command, context, history)
        if self.api_type == "openai":
            return self._openai_parse(prompt)
        else:
            return self._llama_parse(prompt)

    def _build_prompt(self, command, context, history):
        prompt = "You are an AI workflow assistant. Given the user's command, context, and history, output a list of agent tasks in JSON.\n"
        if history:
            prompt += f"History: {history}\n"
        prompt += f"Context: {context}\n"
        prompt += f"Command: {command}\n"
        prompt += "Output: [ { 'agent': ..., 'params': ... }, ... ]\n"
        return prompt

    def _openai_parse(self, prompt):
        import openai
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=256,
            temperature=0.2,
        )
        # Extract JSON from response
        import json
        try:
            text = response.choices[0].message.content
            return json.loads(text[text.find("["):text.rfind("]")+1])
        except Exception:
            return []

    def _llama_parse(self, prompt):
        # Assumes local Llama.cpp server with OpenAI-compatible API
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama-2-7b-chat",
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.2,
        }
        try:
            r = requests.post(self.endpoint, headers=headers, json=data, timeout=30)
            r.raise_for_status()
            import json
            text = r.json()["choices"][0]["text"]
            return json.loads(text[text.find("["):text.rfind("]")+1])
        except Exception:
            return []

# Usage:
# llm = LLMClient(api_type="openai", api_key="sk-...")
# steps = llm.llm_parse("Restore and upscale this image", context, history) 