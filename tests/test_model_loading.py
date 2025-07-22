import pytest
from src.core.advanced_local_models import local_model_manager  # adjust if needed

@pytest.mark.asyncio
async def test_load_model():
    model = await local_model_manager.load_model("some_model")
    assert model is not None
