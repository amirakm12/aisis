#!/bin/bash

echo "========================================"
echo "HARDCORE LINUX PERFORMANCE OPTIMIZER"
echo "========================================"
echo
echo "WARNING: This script makes aggressive system changes!"
echo "Make sure you have root privileges (run with sudo)."
echo "Press Enter to continue or Ctrl+C to abort..."
read

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

echo
echo "[1/12] Setting CPU Governor to Performance..."
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
cpupower frequency-set -g performance 2>/dev/null || echo "cpupower not available, using sysfs"

echo "[2/12] Disabling CPU Frequency Scaling..."
echo 1 | tee /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo "Intel P-State not available"

echo "[3/12] Setting Maximum CPU Frequency..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
    if [ -f "$cpu" ]; then
        max_freq=$(cat "${cpu%/*}/cpuinfo_max_freq")
        echo $max_freq > "$cpu"
    fi
done

echo "[4/12] Disabling CPU Idle States..."
for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    if [ -f "$state" ]; then
        echo 1 > "$state"
    fi
done

echo "[5/12] Disabling Swap..."
swapoff -a
sed -i '/swap/d' /etc/fstab

echo "[6/12] Setting Kernel Parameters for Performance..."
sysctl -w kernel.sched_migration_cost_ns=5000000
sysctl -w kernel.sched_autogroup_enabled=0
sysctl -w vm.swappiness=1
sysctl -w vm.dirty_ratio=15
sysctl -w vm.dirty_background_ratio=5

echo "[7/12] Disabling Unnecessary Services..."
systemctl disable bluetooth.service 2>/dev/null || true
systemctl disable cups.service 2>/dev/null || true
systemctl disable avahi-daemon.service 2>/dev/null || true
systemctl disable NetworkManager-wait-online.service 2>/dev/null || true

echo "[8/12] Setting I/O Scheduler to Performance..."
for disk in /sys/block/sd*/queue/scheduler; do
    if [ -f "$disk" ]; then
        echo performance > "$disk" 2>/dev/null || echo mq-deadline > "$disk" 2>/dev/null || true
    fi
done

echo "[9/12] Disabling Transparent Huge Pages..."
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag

echo "[10/12] Setting Network Optimizations..."
sysctl -w net.core.busy_read=50
sysctl -w net.core.busy_poll=50
sysctl -w net.core.netdev_max_backlog=5000

echo "[11/12] Disabling Power Management..."
for pci in /sys/bus/pci/devices/*/power/control; do
    if [ -f "$pci" ]; then
        echo on > "$pci"
    fi
done

echo "[12/12] Setting Real-time Kernel Parameters..."
sysctl -w kernel.sched_rt_runtime_us=-1
sysctl -w kernel.sched_rt_period_us=1000000

# Make changes persistent
cat << EOF >> /etc/sysctl.conf

# Hardcore Performance Optimizations
kernel.sched_migration_cost_ns=5000000
kernel.sched_autogroup_enabled=0
vm.swappiness=1
vm.dirty_ratio=15
vm.dirty_background_ratio=5
net.core.busy_read=50
net.core.busy_poll=50
net.core.netdev_max_backlog=5000
kernel.sched_rt_runtime_us=-1
kernel.sched_rt_period_us=1000000
EOF

# Create systemd service for CPU optimizations
cat << EOF > /etc/systemd/system/hardcore-performance.service
[Unit]
Description=Hardcore Performance Optimizations
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
ExecStart=/bin/bash -c 'echo 1 | tee /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true'
ExecStart=/bin/bash -c 'for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do [ -f "\$state" ] && echo 1 > "\$state"; done'
ExecStart=/bin/bash -c 'echo never > /sys/kernel/mm/transparent_hugepage/enabled'
ExecStart=/bin/bash -c 'echo never > /sys/kernel/mm/transparent_hugepage/defrag'

[Install]
WantedBy=multi-user.target
EOF

systemctl enable hardcore-performance.service

echo
echo "========================================"
echo "OPTIMIZATION COMPLETE!"
echo "========================================"
echo
echo "IMPORTANT NEXT STEPS:"
echo "1. Reboot your system for all changes to take effect"
echo "2. Install a low-latency kernel: apt install linux-lowlatency"
echo "3. Run your application with: sudo nice -n -20 ./hardcore_performance"
echo "4. Monitor temperatures with: watch sensors"
echo "5. Use CPU isolation for dedicated cores (add isolcpus=1-N to kernel parameters)"
echo
echo "Additional Manual Optimizations:"
echo "- BIOS: Disable Turbo Boost for consistent performance"
echo "- BIOS: Set CPU to maximum performance mode"
echo "- BIOS: Disable C-states and sleep states"
echo "- Install latest CPU microcode updates"
echo "- Use high-performance cooling solution"
echo "- Consider PREEMPT_RT kernel for ultimate real-time performance"
echo
echo "To check current CPU governor: cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
echo "To check CPU frequencies: cat /proc/cpuinfo | grep MHz"
echo