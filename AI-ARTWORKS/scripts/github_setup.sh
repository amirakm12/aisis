#!/bin/bash

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
fi

# Add all files
echo "Adding files to git..."
git add .

# Initial commit
echo "Creating initial commit..."
git commit -m "Initial commit of AI-ARTWORK - AI Creative Studio

- Project structure and architecture
- Core AI agent system
- Voice interaction stubs
- Modern UI framework
- Plugin system foundation
- Comprehensive documentation"

# Instructions for the user
echo "
🎉 Local repository is ready!

To push to GitHub:

1. Create a new repository at https://github.com/new
2. Then run these commands (replace YOUR-USERNAME with your GitHub username):

git remote add origin https://github.com/YOUR-USERNAME/ai-artwork.git
git branch -M main
git push -u origin main

For more information, check the README.md file.
" 