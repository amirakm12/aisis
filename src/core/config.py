import os
import json
from typing import Any, Dict
from pathlib import Path

class Config:
    """
    Configuration management for AISIS.
    Loads and saves configuration from a JSON file with dot notation support.
    """

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.data = json.load(f)
        else:
            self.data = self.default_config()
            self.save()

    def save(self):
        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        serializable_data = convert_paths(self.data)
        
        with open(self.config_path, "w") as f:
            json.dump(serializable_data, f, indent=4)

    def default_config(self) -> Dict[str, Any]:
        return {
            "voice": {
                "whisper_model": "small",
                "tts_engine": "bark",
                "sample_rate": 16000,
                "language": "en",
                "chunk_size": 30,
            },
            "llm": {
                "model_name": "mixtral-8x7b",
                "quantized": True,
            },
            "gpu": {
                "use_cuda": True,
                "device_id": 0,
            },
            "paths": {
                "models_dir": Path("models"),
                "cache_dir": Path("cache"),
                "textures_db": "textures/textures.sqlite",
            },
            "ui": {
                "theme": "dark",
                "window_size": [1280, 720],
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get value using dot notation (e.g., 'voice.sample_rate')"""
        keys = key.split('.')
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def set(self, key: str, value: Any):
        """Set value using dot notation"""
        keys = key.split('.')
        data = self.data
        
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        
        data[keys[-1]] = value
        self.save()

    def __getattr__(self, name: str) -> Any:
        """Support attribute-style access for nested config"""
        if name in self.data:
            return ConfigDict(self.data[name])
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

class ConfigDict:
    """Helper class for nested config access"""
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

# Global config instance
config = Config()
