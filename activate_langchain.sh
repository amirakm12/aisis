#!/bin/bash
# Activation script for LangChain environment

echo "Activating LangChain virtual environment..."
source venv/bin/activate

echo "LangChain environment activated!"
echo "You can now use LangChain in your Python scripts."
echo ""
echo "To test the installation, run:"
echo "  python test_langchain.py"
echo ""
echo "To run the main example, run:"
echo "  python main.py"
echo ""
echo "Remember to set your API keys in a .env file if you want to use real LLMs:"
echo "  cp .env.example .env"
echo "  # Then edit .env with your actual API keys"