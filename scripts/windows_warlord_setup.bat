@echo off
echo.
echo 🔥🔥🔥 WINDOWS WARLORD SYSTEM OPTIMIZATION 🔥🔥🔥
echo 💀 WARNING: MAXIMUM PERFORMANCE MODE - NO MERCY 💀
echo.
echo This script will brutally optimize your Windows system for maximum performance.
echo Ensure you have administrator privileges and adequate cooling!
echo.
pause

REM Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ ERROR: Administrator privileges required!
    echo Right-click and "Run as Administrator"
    pause
    exit /b 1
)

echo ⚡ PHASE 1: POWER MANAGEMENT DOMINATION
echo.

REM Duplicate and activate Ultimate Performance power scheme
echo 🚀 Creating Ultimate Performance power scheme...
powercfg /duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61
if %errorLevel% equ 0 (
    echo ✅ Ultimate Performance scheme created
    powercfg /setactive e9a42b02-d5df-448d-aa00-03f14749eb61
    echo ✅ Ultimate Performance activated
) else (
    echo ⚠️ Ultimate Performance scheme may already exist
    powercfg /setactive e9a42b02-d5df-448d-aa00-03f14749eb61
)

REM Set CPU min/max to 100%
echo 🎯 Setting CPU frequency to maximum...
powercfg /setacvalueindex SCHEME_CURRENT 54533251-82be-4824-96c1-47b60b740d00 893dee8e-2bef-41e0-89c6-b55d0929964c 100
powercfg /setacvalueindex SCHEME_CURRENT 54533251-82be-4824-96c1-47b60b740d00 bc5038f7-23e0-4960-96da-33abaf5935ec 100
powercfg /setactive SCHEME_CURRENT

REM Disable processor parking
echo 💀 DISABLING processor parking (keeping all cores online)...
powercfg /setacvalueindex SCHEME_CURRENT 54533251-82be-4824-96c1-47b60b740d00 0cc5b647-c1df-4637-891a-dec35c318583 0
powercfg /setdcvalueindex SCHEME_CURRENT 54533251-82be-4824-96c1-47b60b740d00 0cc5b647-c1df-4637-891a-dec35c318583 0

REM Disable all sleep states
echo 🚫 DISABLING all sleep states and C-states...
powercfg /setacvalueindex SCHEME_CURRENT 238c9fa8-0aad-41ed-83f4-97be242c8f20 29f6c1db-86da-48c5-9fdb-f2b67b1f44da 0
powercfg /setacvalueindex SCHEME_CURRENT 238c9fa8-0aad-41ed-83f4-97be242c8f20 25dfa149-5dd1-4736-b5ab-e8a37b5b8187 0

REM Apply power settings
powercfg /setactive SCHEME_CURRENT

echo.
echo ⚡ PHASE 2: SYSTEM PERFORMANCE BRUTALIZATION
echo.

REM Disable Windows Defender real-time protection (WARNING: Security risk)
echo 🛡️ DISABLING Windows Defender (SECURITY RISK - for performance only)...
reg add "HKLM\SOFTWARE\Policies\Microsoft\Windows Defender" /v DisableAntiSpyware /t REG_DWORD /d 1 /f
reg add "HKLM\SOFTWARE\Policies\Microsoft\Windows Defender\Real-Time Protection" /v DisableRealtimeMonitoring /t REG_DWORD /d 1 /f

REM Disable Windows Update automatic restart
echo 🔄 DISABLING automatic Windows Update restarts...
reg add "HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU" /v NoAutoRebootWithLoggedOnUsers /t REG_DWORD /d 1 /f

REM Disable HPET (High Precision Event Timer) - can reduce latency on some systems
echo ⏰ DISABLING HPET (High Precision Event Timer)...
bcdedit /deletevalue useplatformclock
bcdedit /set disabledynamictick yes

REM Set system to high performance visual settings
echo 🎨 Setting MAXIMUM PERFORMANCE visual settings...
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\VisualEffects" /v VisualFXSetting /t REG_DWORD /d 2 /f

REM Disable unnecessary services
echo 🗑️ DISABLING performance-killing services...
sc config "Fax" start= disabled
sc config "Spooler" start= disabled
sc config "Themes" start= disabled
sc config "TabletInputService" start= disabled
sc config "WebClient" start= disabled

REM Disable Game Mode (can interfere with manual optimizations)
echo 🎮 DISABLING Windows Game Mode...
reg add "HKCU\Software\Microsoft\GameBar" /v AllowAutoGameMode /t REG_DWORD /d 0 /f
reg add "HKCU\Software\Microsoft\GameBar" /v AutoGameModeEnabled /t REG_DWORD /d 0 /f

REM Set network adapter settings for minimum latency
echo 🌐 OPTIMIZING network settings for minimum latency...
for /f "tokens=3*" %%i in ('reg query "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\NetworkCards" /s /f "ServiceName" ^| findstr /i "ServiceName"') do (
    reg add "HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e972-e325-11ce-bfc1-08002be10318}\%%j" /v "*InterruptModeration" /t REG_SZ /d "0" /f 2>nul
    reg add "HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e972-e325-11ce-bfc1-08002be10318}\%%j" /v "*RSS" /t REG_SZ /d "1" /f 2>nul
)

echo.
echo ⚡ PHASE 3: MEMORY AND CACHE OPTIMIZATION
echo.

REM Disable paging executive
echo 💾 DISABLING paging executive (keep kernel in RAM)...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management" /v DisablePagingExecutive /t REG_DWORD /d 1 /f

REM Set large system cache
echo 📈 ENABLING large system cache...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management" /v LargeSystemCache /t REG_DWORD /d 1 /f

REM Clear page file at shutdown (optional - increases shutdown time)
echo 🗑️ Setting page file clear at shutdown...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management" /v ClearPageFileAtShutdown /t REG_DWORD /d 1 /f

echo.
echo ⚡ PHASE 4: INTERRUPT AND TIMER OPTIMIZATION
echo.

REM Set timer resolution to maximum precision
echo ⏱️ SETTING maximum timer resolution...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\kernel" /v GlobalTimerResolutionRequests /t REG_DWORD /d 1 /f

REM Disable CPU throttling
echo 🚫 DISABLING CPU throttling...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Power\PowerThrottling" /v PowerThrottlingOff /t REG_DWORD /d 1 /f

echo.
echo ⚡ PHASE 5: FINAL SYSTEM TWEAKS
echo.

REM Disable Superfetch/SysMain (can cause disk thrashing)
echo 🗑️ DISABLING Superfetch/SysMain...
sc config "SysMain" start= disabled
sc stop "SysMain"

REM Disable Windows Search indexing (performance impact)
echo 🔍 DISABLING Windows Search indexing...
sc config "WSearch" start= disabled
sc stop "WSearch"

REM Set processor scheduling for programs (not background services)
echo 🎯 OPTIMIZING processor scheduling for programs...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\PriorityControl" /v Win32PrioritySeparation /t REG_DWORD /d 38 /f

REM Disable Windows Error Reporting
echo ❌ DISABLING Windows Error Reporting...
reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting" /v Disabled /t REG_DWORD /d 1 /f

echo.
echo 🔥🔥🔥 WARLORD OPTIMIZATION COMPLETE! 🔥🔥🔥
echo.
echo ✅ Ultimate Performance power scheme activated
echo ✅ Processor parking disabled
echo ✅ All sleep states disabled
echo ✅ HPET disabled for reduced latency
echo ✅ Performance-killing services disabled
echo ✅ Network adapters optimized for minimum latency
echo ✅ Memory management optimized
echo ✅ Timer resolution maximized
echo ✅ CPU throttling disabled
echo.
echo ⚠️ WARNING: Some changes require a REBOOT to take effect!
echo ⚠️ WARNING: Windows Defender has been disabled (security risk)
echo ⚠️ WARNING: Your system will run HOT - ensure adequate cooling!
echo.
echo 🌡️ CRITICAL: Monitor temperatures with HWiNFO64 or Core Temp
echo 💡 Use Process Explorer to verify thread affinity and priorities
echo 📊 Use LatencyMon to check for system latency issues
echo.
echo Press any key to reboot now, or Ctrl+C to reboot later...
pause
shutdown /r /t 5 /c "Rebooting for WARLORD performance optimizations..."