import re
import os
from pathlib import Path

def sanitize_string(input_str: str) -> str:
    """Remove potentially harmful characters from string"""
    return re.sub(r'[^a-zA-Z0-9_ -]', '', input_str)

def sanitize_path(path: str) -> Path:
    """Sanitize and normalize path"""
    safe_path = Path(path).resolve()
    if not safe_path.is_relative_to(os.getcwd()):
        raise ValueError("Path outside allowed directory")
    return safe_path
