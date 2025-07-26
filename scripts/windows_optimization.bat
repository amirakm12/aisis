@echo off
echo ========================================
echo HARDCORE WINDOWS PERFORMANCE OPTIMIZER
echo ========================================
echo.
echo WARNING: This script makes aggressive system changes!
echo Make sure you have administrator privileges.
echo Press any key to continue or Ctrl+C to abort...
pause >nul

echo.
echo [1/10] Setting Ultimate Performance Power Plan...
powercfg /duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61
powercfg /setactive e9a42b02-d5df-448d-aa00-03f14749eb61
powercfg /setacvalueindex scheme_current sub_processor PROCTHROTTLEMIN 100
powercfg /setacvalueindex scheme_current sub_processor PROCTHROTTLEMAX 100
powercfg /setdcvalueindex scheme_current sub_processor PROCTHROTTLEMIN 100
powercfg /setdcvalueindex scheme_current sub_processor PROCTHROTTLEMAX 100

echo [2/10] Disabling CPU Core Parking...
powercfg /setacvalueindex scheme_current sub_processor CPMINCORES 100
powercfg /setdcvalueindex scheme_current sub_processor CPMINCORES 100

echo [3/10] Disabling Power Throttling...
reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Power\PowerThrottling" /v PowerThrottlingOff /t REG_DWORD /d 1 /f

echo [4/10] Optimizing Process Scheduling...
reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\PriorityControl" /v Win32PrioritySeparation /t REG_DWORD /d 38 /f

echo [5/10] Disabling Windows Defender Real-time Protection...
echo NOTE: This requires manual intervention in Windows Security settings
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Policies\Microsoft\Windows Defender" /v DisableAntiSpyware /t REG_DWORD /d 1 /f
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Policies\Microsoft\Windows Defender\Real-Time Protection" /v DisableRealtimeMonitoring /t REG_DWORD /d 1 /f

echo [6/10] Disabling HPET (High Precision Event Timer)...
bcdedit /set useplatformclock false
bcdedit /set disabledynamictick yes

echo [7/10] Setting High Performance Timer Resolution...
bcdedit /set useplatformtick yes

echo [8/10] Disabling Unnecessary Services...
sc config "SysMain" start= disabled
sc config "WSearch" start= disabled
sc config "DiagTrack" start= disabled
sc config "dmwappushservice" start= disabled

echo [9/10] Optimizing Memory Management...
reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management" /v LargeSystemCache /t REG_DWORD /d 0 /f
reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management" /v DisablePagingExecutive /t REG_DWORD /d 1 /f

echo [10/10] Setting Network Adapter Optimizations...
echo You need to manually disable interrupt moderation in network adapter properties.

echo.
echo ========================================
echo OPTIMIZATION COMPLETE!
echo ========================================
echo.
echo IMPORTANT NEXT STEPS:
echo 1. Restart your computer for all changes to take effect
echo 2. Manually disable Windows Defender real-time protection
echo 3. Set your application to Realtime priority in Task Manager
echo 4. Disable interrupt moderation in network adapter settings
echo 5. Monitor temperatures during high-load operation
echo.
echo Additional Manual Optimizations:
echo - BIOS: Disable Turbo Boost for consistent performance
echo - BIOS: Set CPU to maximum performance mode
echo - BIOS: Disable C-states and sleep states
echo - Install latest chipset and CPU drivers
echo - Use high-performance cooling solution
echo.
pause