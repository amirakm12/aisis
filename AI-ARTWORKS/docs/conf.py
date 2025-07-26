import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'AI-ARTWORK'
copyright = '2024, AI-ARTWORK Team'
author = 'AI-ARTWORK Team'
release = '0.1.0'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'alabaster' 