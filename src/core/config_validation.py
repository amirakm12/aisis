from pydantic import BaseModel, ValidationError, Field
from typing import Optional

class UIConfig(BaseModel):
    theme: str = Field(default="dark")
    window_size: Optional[list[int]] = Field(default=[1280, 720])

class GPUConfig(BaseModel):
    use_cuda: bool = Field(default=True)
    device_id: int = Field(default=0)

class PathsConfig(BaseModel):
    models_dir: str = Field(default="models")
    cache_dir: str = Field(default="cache")
    textures_db: str = Field(default="textures/textures.sqlite")

class VoiceConfig(BaseModel):
    whisper_model: str = Field(default="small")
    tts_engine: str = Field(default="bark")
    sample_rate: int = Field(default=16000)
    language: str = Field(default="en")
    chunk_size: int = Field(default=30)

class LLMConfig(BaseModel):
    model_name: str = Field(default="mixtral-8x7b")
    quantized: bool = Field(default=True)

class AISISConfig(BaseModel):
    voice: VoiceConfig
    llm: LLMConfig
    gpu: GPUConfig
    paths: PathsConfig
    ui: UIConfig

# Usage:
# try:
#     config = AISISConfig(**your_dict)
# except ValidationError as e:
#     print(e)
