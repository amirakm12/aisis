@echo off
REM AISIS Windows 64-bit Installer Build Script

REM 1. Create virtual environment (optional but recommended)
python -m venv venv
call venv\Scripts\activate

REM 2. Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

REM 3. Build the executable (main entry: main.py or src/main.py)
pyinstaller --noconfirm --onefile --windowed --add-data "src;src" --add-data "plugins;plugins" --add-data "config.json;." --add-data "models;models" --add-data "aisis.env;." --name AISIS src/main.py

REM 4. Output location
echo.
echo Build complete! Your Windows 64-bit executable is in the dist\AISIS.exe
pause 