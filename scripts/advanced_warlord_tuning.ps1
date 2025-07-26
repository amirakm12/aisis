# Advanced Warlord Performance Tuning Script
# Run as Administrator in PowerShell
# Requires: Windows 10/11, PowerShell 5.1+

param(
    [switch]$DisableTurboBoost,
    [switch]$SetCPUAffinity,
    [switch]$OptimizeMemory,
    [switch]$MonitorPerformance,
    [switch]$RestoreDefaults
)

Write-Host @"
🔥🔥🔥 ADVANCED WARLORD PERFORMANCE TUNING 🔥🔥🔥
💀 PowerShell Edition - Maximum System Domination 💀
⚡ Advanced CPU, Memory, and System Optimizations ⚡
"@ -ForegroundColor Red

# Check for Administrator privileges
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Error "❌ Administrator privileges required! Run PowerShell as Administrator."
    exit 1
}

function Get-SystemInfo {
    Write-Host "`n📊 SYSTEM INFORMATION:" -ForegroundColor Cyan
    
    $cpu = Get-WmiObject -Class Win32_Processor
    $memory = Get-WmiObject -Class Win32_ComputerSystem
    $os = Get-WmiObject -Class Win32_OperatingSystem
    
    Write-Host "🔥 CPU: $($cpu.Name)" -ForegroundColor Green
    Write-Host "💾 Total RAM: $([math]::Round($memory.TotalPhysicalMemory / 1GB, 2)) GB" -ForegroundColor Green
    Write-Host "⚡ CPU Cores: $($cpu.NumberOfCores)" -ForegroundColor Green
    Write-Host "🧵 Logical Processors: $($cpu.NumberOfLogicalProcessors)" -ForegroundColor Green
    Write-Host "🖥️ OS: $($os.Caption) $($os.OSArchitecture)" -ForegroundColor Green
    
    # Check current power scheme
    $currentScheme = powercfg /getactivescheme
    Write-Host "⚡ Current Power Scheme: $currentScheme" -ForegroundColor Yellow
}

function Disable-TurboBoost {
    Write-Host "`n🚫 DISABLING Intel Turbo Boost..." -ForegroundColor Red
    
    # Disable Turbo Boost via registry (Intel)
    $regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Power\PowerSettings\54533251-82be-4824-96c1-47b60b740d00\be337238-0d82-4146-a960-4f3749d470c7"
    if (Test-Path $regPath) {
        Set-ItemProperty -Path $regPath -Name "Attributes" -Value 2
        Write-Host "✅ Intel Turbo Boost registry setting modified" -ForegroundColor Green
    }
    
    # Set maximum processor state to 99% (effectively disables Turbo Boost)
    powercfg /setacvalueindex SCHEME_CURRENT 54533251-82be-4824-96c1-47b60b740d00 bc5038f7-23e0-4960-96da-33abaf5935ec 99
    powercfg /setdcvalueindex SCHEME_CURRENT 54533251-82be-4824-96c1-47b60b740d00 bc5038f7-23e0-4960-96da-33abaf5935ec 99
    powercfg /setactive SCHEME_CURRENT
    
    Write-Host "✅ Turbo Boost disabled via power management" -ForegroundColor Green
}

function Set-CPUAffinity {
    param([string]$ProcessName = "warlord_performance")
    
    Write-Host "`n🎯 SETTING CPU AFFINITY for high-performance processes..." -ForegroundColor Red
    
    # Get all physical cores (avoid hyperthreading siblings)
    $coreCount = (Get-WmiObject -Class Win32_Processor).NumberOfCores
    $logicalCount = (Get-WmiObject -Class Win32_Processor).NumberOfLogicalProcessors
    
    Write-Host "🔥 Physical Cores: $coreCount, Logical Processors: $logicalCount" -ForegroundColor Yellow
    
    if ($logicalCount -gt $coreCount) {
        Write-Host "🧵 Hyperthreading detected - will prefer physical cores" -ForegroundColor Yellow
        
        # Create affinity mask for physical cores only (assuming even-numbered cores are physical)
        $physicalCoreMask = 0
        for ($i = 0; $i -lt $logicalCount; $i += 2) {
            $physicalCoreMask = $physicalCoreMask -bor (1 -shl $i)
        }
        
        Write-Host "🎯 Physical core affinity mask: 0x$($physicalCoreMask.ToString('X'))" -ForegroundColor Cyan
    }
    
    # Function to set affinity for a process when it starts
    $affinityScript = @"
`$processName = '$ProcessName'
while (`$true) {
    `$processes = Get-Process -Name `$processName -ErrorAction SilentlyContinue
    foreach (`$proc in `$processes) {
        if (`$proc.ProcessorAffinity -ne $physicalCoreMask) {
            `$proc.ProcessorAffinity = $physicalCoreMask
            Write-Host "🎯 Set affinity for `$(`$proc.ProcessName) (PID: `$(`$proc.Id))"
        }
    }
    Start-Sleep -Seconds 5
}
"@
    
    # Save affinity monitoring script
    $affinityScript | Out-File -FilePath ".\cpu_affinity_monitor.ps1" -Encoding UTF8
    Write-Host "💾 CPU affinity monitoring script saved to cpu_affinity_monitor.ps1" -ForegroundColor Green
    Write-Host "💡 Run: Start-Process PowerShell -ArgumentList '-File cpu_affinity_monitor.ps1' -WindowStyle Hidden" -ForegroundColor Cyan
}

function Optimize-Memory {
    Write-Host "`n💾 OPTIMIZING MEMORY SETTINGS..." -ForegroundColor Red
    
    # Set working set sizes
    $regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
    
    # Disable paging executive
    Set-ItemProperty -Path $regPath -Name "DisablePagingExecutive" -Value 1
    Write-Host "✅ Paging executive disabled" -ForegroundColor Green
    
    # Enable large system cache
    Set-ItemProperty -Path $regPath -Name "LargeSystemCache" -Value 1
    Write-Host "✅ Large system cache enabled" -ForegroundColor Green
    
    # Set I/O page lock limit (25% of RAM)
    $totalRAM = (Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory
    $ioPageLockLimit = [math]::Floor($totalRAM * 0.25)
    Set-ItemProperty -Path $regPath -Name "IoPageLockLimit" -Value $ioPageLockLimit
    Write-Host "✅ I/O page lock limit set to $([math]::Round($ioPageLockLimit / 1MB, 2)) MB" -ForegroundColor Green
    
    # Set system pages (for systems with >4GB RAM)
    if ($totalRAM -gt 4GB) {
        Set-ItemProperty -Path $regPath -Name "SystemPages" -Value 0
        Write-Host "✅ System pages optimized for large RAM" -ForegroundColor Green
    }
    
    # Optimize page file settings
    $pageFileSize = [math]::Floor($totalRAM / 1GB * 1.5) * 1024  # 1.5x RAM in MB
    Write-Host "🗃️ Recommended page file size: $pageFileSize MB" -ForegroundColor Yellow
    
    # Clear standby memory (requires RAMMap or similar tool)
    Write-Host "💡 To clear standby memory, use: RAMMap -Et" -ForegroundColor Cyan
}

function Start-PerformanceMonitoring {
    Write-Host "`n📈 STARTING PERFORMANCE MONITORING..." -ForegroundColor Red
    
    # Create performance monitoring script
    $monitorScript = @"
# Warlord Performance Monitor
while (`$true) {
    Clear-Host
    Write-Host "🔥🔥🔥 WARLORD PERFORMANCE MONITOR 🔥🔥🔥" -ForegroundColor Red
    Write-Host "Time: `$(Get-Date)" -ForegroundColor Cyan
    Write-Host ""
    
    # CPU Usage
    `$cpu = Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1
    `$cpuUsage = [math]::Round(100 - `$cpu.CounterSamples.CookedValue, 2)
    Write-Host "🔥 CPU Usage: `$cpuUsage%" -ForegroundColor $(if (`$cpuUsage -gt 80) { 'Red' } else { 'Green' })
    
    # Memory Usage
    `$memory = Get-Counter '\Memory\Available MBytes' -SampleInterval 1 -MaxSamples 1
    `$totalRAM = (Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1MB
    `$usedRAM = `$totalRAM - `$memory.CounterSamples.CookedValue
    `$memoryUsage = [math]::Round((`$usedRAM / `$totalRAM) * 100, 2)
    Write-Host "💾 Memory Usage: `$memoryUsage% (`$([math]::Round(`$usedRAM, 0)) MB / `$([math]::Round(`$totalRAM, 0)) MB)" -ForegroundColor $(if (`$memoryUsage -gt 80) { 'Red' } else { 'Green' })
    
    # Disk Usage
    `$disk = Get-Counter '\PhysicalDisk(_Total)\% Disk Time' -SampleInterval 1 -MaxSamples 1
    `$diskUsage = [math]::Round(`$disk.CounterSamples.CookedValue, 2)
    Write-Host "💿 Disk Usage: `$diskUsage%" -ForegroundColor $(if (`$diskUsage -gt 80) { 'Red' } else { 'Green' })
    
    # Network Usage
    `$network = Get-Counter '\Network Interface(*)\Bytes Total/sec' -SampleInterval 1 -MaxSamples 1 | Where-Object { `$_.CounterSamples.InstanceName -notlike '*Loopback*' -and `$_.CounterSamples.InstanceName -notlike '*isatap*' }
    if (`$network) {
        `$networkUsage = [math]::Round((`$network.CounterSamples | Measure-Object -Property CookedValue -Sum).Sum / 1MB, 2)
        Write-Host "🌐 Network Usage: `$networkUsage MB/s" -ForegroundColor Green
    }
    
    # Process Information
    Write-Host ""
    Write-Host "🎯 TOP CPU PROCESSES:" -ForegroundColor Yellow
    Get-Process | Sort-Object CPU -Descending | Select-Object -First 5 ProcessName, CPU, WorkingSet | Format-Table -AutoSize
    
    Write-Host "💀 HIGH PRIORITY PROCESSES:" -ForegroundColor Yellow
    Get-Process | Where-Object { `$_.PriorityClass -eq 'High' -or `$_.PriorityClass -eq 'RealTime' } | Select-Object ProcessName, PriorityClass, CPU | Format-Table -AutoSize
    
    Start-Sleep -Seconds 5
}
"@
    
    $monitorScript | Out-File -FilePath ".\performance_monitor.ps1" -Encoding UTF8
    Write-Host "💾 Performance monitor script saved to performance_monitor.ps1" -ForegroundColor Green
    Write-Host "🚀 Starting performance monitor..." -ForegroundColor Cyan
    
    # Start the monitor in a new window
    Start-Process PowerShell -ArgumentList "-File performance_monitor.ps1" -WindowStyle Normal
}

function Set-NetworkOptimizations {
    Write-Host "`n🌐 OPTIMIZING NETWORK SETTINGS..." -ForegroundColor Red
    
    # TCP settings for maximum performance
    netsh int tcp set global autotuninglevel=normal
    netsh int tcp set global chimney=enabled
    netsh int tcp set global rss=enabled
    netsh int tcp set global netdma=enabled
    netsh int tcp set global dca=enabled
    netsh int tcp set global ecncapability=enabled
    
    Write-Host "✅ TCP optimizations applied" -ForegroundColor Green
    
    # Disable Nagle's algorithm for low latency
    $regPath = "HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces"
    Get-ChildItem $regPath | ForEach-Object {
        $interfacePath = $_.PSPath
        try {
            Set-ItemProperty -Path $interfacePath -Name "TcpAckFrequency" -Value 1 -ErrorAction SilentlyContinue
            Set-ItemProperty -Path $interfacePath -Name "TCPNoDelay" -Value 1 -ErrorAction SilentlyContinue
        } catch {
            # Interface may not support these settings
        }
    }
    
    Write-Host "✅ Network latency optimizations applied" -ForegroundColor Green
}

function Restore-Defaults {
    Write-Host "`n🔄 RESTORING DEFAULT SETTINGS..." -ForegroundColor Yellow
    
    # Restore balanced power scheme
    powercfg /setactive SCHEME_BALANCED
    
    # Re-enable processor parking
    powercfg /setacvalueindex SCHEME_CURRENT 54533251-82be-4824-96c1-47b60b740d00 0cc5b647-c1df-4637-891a-dec35c318583 1
    
    # Restore default memory settings
    $regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
    Set-ItemProperty -Path $regPath -Name "DisablePagingExecutive" -Value 0
    Set-ItemProperty -Path $regPath -Name "LargeSystemCache" -Value 0
    
    Write-Host "✅ Default settings restored" -ForegroundColor Green
    Write-Host "⚠️ Reboot required for all changes to take effect" -ForegroundColor Yellow
}

# Main execution based on parameters
Get-SystemInfo

if ($DisableTurboBoost) {
    Disable-TurboBoost
}

if ($SetCPUAffinity) {
    Set-CPUAffinity
}

if ($OptimizeMemory) {
    Optimize-Memory
}

if ($MonitorPerformance) {
    Start-PerformanceMonitoring
}

if ($RestoreDefaults) {
    Restore-Defaults
    exit
}

# If no specific parameters, show menu
if (-not ($DisableTurboBoost -or $SetCPUAffinity -or $OptimizeMemory -or $MonitorPerformance -or $RestoreDefaults)) {
    Write-Host "`n🎮 WARLORD TUNING MENU:" -ForegroundColor Cyan
    Write-Host "1. Disable Turbo Boost" -ForegroundColor White
    Write-Host "2. Set CPU Affinity" -ForegroundColor White
    Write-Host "3. Optimize Memory" -ForegroundColor White
    Write-Host "4. Start Performance Monitor" -ForegroundColor White
    Write-Host "5. Optimize Network Settings" -ForegroundColor White
    Write-Host "6. ALL OPTIMIZATIONS" -ForegroundColor Red
    Write-Host "9. Restore Defaults" -ForegroundColor Yellow
    Write-Host "0. Exit" -ForegroundColor Gray
    
    $choice = Read-Host "`nEnter your choice"
    
    switch ($choice) {
        "1" { Disable-TurboBoost }
        "2" { Set-CPUAffinity }
        "3" { Optimize-Memory }
        "4" { Start-PerformanceMonitoring }
        "5" { Set-NetworkOptimizations }
        "6" { 
            Disable-TurboBoost
            Set-CPUAffinity
            Optimize-Memory
            Set-NetworkOptimizations
            Start-PerformanceMonitoring
        }
        "9" { Restore-Defaults }
        "0" { exit }
        default { Write-Host "Invalid choice" -ForegroundColor Red }
    }
}

Write-Host "`n🔥 WARLORD TUNING COMPLETE!" -ForegroundColor Red
Write-Host "⚠️ Some changes require a reboot to take effect" -ForegroundColor Yellow
Write-Host "🌡️ Monitor your temperatures closely!" -ForegroundColor Red