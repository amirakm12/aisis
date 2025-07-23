import pytest
import time
from src.agents import some_agent  # Adjust import

@pytest.mark.performance
def test_stress_multiple_images():
    start = time.time()
    for i in range(100):
        result = some_agent.process('test_image.jpg')
    duration = time.time() - start
    assert duration < 60  # Less than 1 min for 100 images