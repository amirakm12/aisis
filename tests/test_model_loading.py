import pytest
from src.core.advanced_local_models import LocalModelManager

@pytest.fixture
 def model_manager():
    return LocalModelManager()

@pytest.mark.asyncio
async def test_model_download(model_manager):
    await model_manager.download_model('test_model')
    # Assert model exists

@pytest.mark.asyncio
async def test_model_load(model_manager):
    model = await model_manager.load_model('test_model')
    assert model is not None