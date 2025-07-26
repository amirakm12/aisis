@echo off
setlocal enabledelayedexpansion

echo.
echo 🔥🔥🔥 WARLORD PERFORMANCE SYSTEM BUILD SCRIPT 🔥🔥🔥
echo 💀 Ultimate Windows System Domination Builder 💀
echo.

REM Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ⚠️ WARNING: Not running as Administrator
    echo Some optimizations may fail without admin privileges
    echo.
)

REM Set build configuration
set BUILD_TYPE=Release
set BUILD_DIR=build
set INSTALL_DIR=install

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :done_parsing
if /i "%~1"=="debug" set BUILD_TYPE=Debug
if /i "%~1"=="clean" set CLEAN_BUILD=1
if /i "%~1"=="optimize" set RUN_OPTIMIZE=1
if /i "%~1"=="deploy" set DEPLOY_MODE=1
shift
goto :parse_args
:done_parsing

echo 🎯 Build Configuration:
echo    Build Type: %BUILD_TYPE%
echo    Build Directory: %BUILD_DIR%
echo    Install Directory: %INSTALL_DIR%
echo.

REM Clean build if requested
if defined CLEAN_BUILD (
    echo 🗑️ Cleaning previous build...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
    if exist %INSTALL_DIR% rmdir /s /q %INSTALL_DIR%
    echo ✅ Clean complete
    echo.
)

REM Create build directory
if not exist %BUILD_DIR% mkdir %BUILD_DIR%
if not exist %INSTALL_DIR% mkdir %INSTALL_DIR%

REM Configure CMake
echo 🔧 Configuring CMake...
cd %BUILD_DIR%

cmake .. ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_INSTALL_PREFIX=../%INSTALL_DIR% ^
    -DCMAKE_GENERATOR_PLATFORM=x64

if %errorLevel% neq 0 (
    echo ❌ CMake configuration failed!
    cd ..
    pause
    exit /b 1
)

echo ✅ CMake configuration complete
echo.

REM Build the project
echo 🚀 Building Warlord Performance System...
cmake --build . --config %BUILD_TYPE% --parallel

if %errorLevel% neq 0 (
    echo ❌ Build failed!
    cd ..
    pause
    exit /b 1
)

echo ✅ Build complete
echo.

REM Install the project
echo 📦 Installing binaries...
cmake --install . --config %BUILD_TYPE%

if %errorLevel% neq 0 (
    echo ❌ Installation failed!
    cd ..
    pause
    exit /b 1
)

cd ..
echo ✅ Installation complete
echo.

REM Copy additional files
echo 📋 Copying additional files...
if not exist bin mkdir bin
copy %BUILD_DIR%\bin\*.exe bin\ >nul 2>&1
copy scripts\*.bat bin\ >nul 2>&1
copy scripts\*.ps1 bin\ >nul 2>&1

echo ✅ Files copied to bin directory
echo.

REM Run system optimizations if requested
if defined RUN_OPTIMIZE (
    echo 🔥 Running system optimizations...
    echo ⚠️ This will modify your system settings!
    choice /c YN /m "Continue with system optimization"
    if !errorlevel! equ 1 (
        call scripts\windows_warlord_setup.bat
        echo ✅ System optimization complete
    ) else (
        echo ❌ System optimization skipped
    )
    echo.
)

REM Deploy mode - run advanced tuning
if defined DEPLOY_MODE (
    echo ⚡ Running advanced deployment...
    powershell -ExecutionPolicy Bypass -File scripts\advanced_warlord_tuning.ps1 -OptimizeMemory -SetCPUAffinity
    echo ✅ Advanced deployment complete
    echo.
)

REM Display build results
echo 🎉 BUILD SUMMARY:
echo ================
echo.
echo 📁 Binaries built:
if exist bin\UltimateSystem.exe (
    echo    ✅ UltimateSystem.exe
) else (
    echo    ❌ UltimateSystem.exe - MISSING
)

if exist bin\warlord_performance.exe (
    echo    ✅ warlord_performance.exe
) else (
    echo    ❌ warlord_performance.exe - MISSING
)

echo.
echo 📁 Scripts available:
if exist bin\windows_warlord_setup.bat (
    echo    ✅ windows_warlord_setup.bat
)
if exist bin\advanced_warlord_tuning.ps1 (
    echo    ✅ advanced_warlord_tuning.ps1
)

echo.
echo 📖 Documentation:
if exist docs\WARLORD_PERFORMANCE_GUIDE.md (
    echo    ✅ WARLORD_PERFORMANCE_GUIDE.md
)

echo.
echo 🔥 WARLORD PERFORMANCE SYSTEM READY! 🔥
echo.
echo 🚀 Quick Start Commands:
echo    bin\warlord_performance.exe          - Run performance test
echo    bin\windows_warlord_setup.bat        - System optimization
echo    bin\advanced_warlord_tuning.ps1      - Advanced tuning
echo.
echo ⚠️ IMPORTANT WARNINGS:
echo    - Ensure adequate cooling before running performance tests
echo    - Monitor temperatures with HWiNFO64 or similar
echo    - System optimizations require Administrator privileges
echo    - Some optimizations disable security features
echo.

REM Option to run immediately
choice /c YN /m "Run Warlord Performance test now"
if %errorlevel% equ 1 (
    echo.
    echo 🔥 LAUNCHING WARLORD PERFORMANCE MODE...
    echo 💀 Ensure your cooling is adequate!
    echo.
    timeout /t 3 /nobreak
    
    if exist bin\warlord_performance.exe (
        bin\warlord_performance.exe
    ) else (
        echo ❌ warlord_performance.exe not found!
    )
) else (
    echo.
    echo 💡 You can run the performance test later with:
    echo    bin\warlord_performance.exe
)

echo.
echo 🔥 Build script complete! May your cores burn bright! 🔥
pause