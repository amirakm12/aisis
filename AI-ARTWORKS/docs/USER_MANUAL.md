# AI-ARTWORK User Manual

## Overview
AI-ARTWORK is an AI-powered image restoration and enhancement platform with plugin and collaboration support.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-artwork.git
cd ai-artwork
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Copy `.env.example` to `.env` and configure as needed.

## Usage
- To launch the main application:
  ```
  python main.py
  ```
- For additional scripts and features, see the `scripts/` and `features/` folders.
- Use the UI to access agents, plugins, and collaboration tools.

## Plugins
- Place plugin files in the `plugins/` directory.
- Use the UI or CLI to enable/disable plugins.

## Troubleshooting
- Check logs in the `logs/` directory.
- Ensure all dependencies are installed.
- For model-related errors, verify the `MODEL_DIR` path in your `.env` file.
- For further help, see the documentation in the `docs/` folder or open an issue on GitHub. 