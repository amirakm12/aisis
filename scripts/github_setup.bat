@echo off
echo Initializing git repository...

if not exist .git (
    git init
)

echo Adding files to git...
git add .

echo Creating initial commit...
git commit -m "Initial commit of AISIS - AI Creative Studio

- Project structure and architecture
- Core AI agent system
- Voice interaction stubs
- Modern UI framework
- Plugin system foundation
- Comprehensive documentation"

echo.
echo 🎉 Local repository is ready!
echo.
echo To push to GitHub:
echo.
echo 1. Create a new repository at https://github.com/new
echo 2. Then run these commands (replace YOUR-USERNAME with your GitHub username):
echo.
echo git remote add origin https://github.com/YOUR-USERNAME/aisis.git
echo git branch -M main
echo git push -u origin main
echo.
echo For more information, check the README.md file.
pause 