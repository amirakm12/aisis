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


class AISISConfig(BaseModel):
    ui: UIConfig
    gpu: GPUConfig
    paths: PathsConfig


# Usage:
# try:
#     config = AISISConfig(**your_dict)
# except ValidationError as e:
#     print(e)
