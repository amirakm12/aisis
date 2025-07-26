@echo off
echo ========================================
echo AISIS Full Installation Test
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found. Running installation test...
echo.

REM Run the installation test
python scripts/installation_test.py

echo.
echo Test completed. Check the output above for results.
echo Detailed results saved to test_results.json
echo.
pause 