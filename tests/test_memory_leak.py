import tracemalloc
import pytest
from src.core import some_function  # Adjust

def test_memory_leak():
    tracemalloc.start()
    for _ in range(100):
        some_function()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    # Assert no significant leaks
    assert len(top_stats) < 10  # Example assertion