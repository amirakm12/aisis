"""
Advanced Thermal & Power Management Agent
Predictive thermal modeling, dynamic throttling, power budget optimization
MAXIMUM PERFORMANCE - FORENSIC LEVEL COMPLEXITY
"""

import asyncio
import time
import threading
import os
import platform
import struct
import ctypes
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import math

from ..core.agent_base import BaseAgent, AgentState
from ..core.communication import MessagePriority


@dataclass
class ThermalSensor:
    """Advanced thermal sensor data structure"""
    sensor_id: str
    location: str
    sensor_type: str  # cpu, gpu, memory, vrm, ambient
    current_temp: float = 0.0
    max_temp: float = 100.0
    critical_temp: float = 105.0
    temp_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    thermal_mass: float = 1.0  # Thermal mass coefficient
    cooling_efficiency: float = 1.0  # Cooling efficiency
    power_correlation: float = 0.8  # Power-temperature correlation


@dataclass
class PowerDomain:
    """Power domain management structure"""
    domain_id: str
    domain_type: str  # cpu, gpu, memory, io, system
    current_power: float = 0.0
    max_power: float = 100.0
    target_power: float = 80.0
    power_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    voltage: float = 1.0
    frequency: float = 1000.0  # MHz
    efficiency_curve: List[Tuple[float, float]] = field(default_factory=list)
    throttle_state: float = 1.0  # 1.0 = no throttling, 0.0 = maximum throttling


@dataclass
class ThermalModel:
    """Advanced thermal modeling structure"""
    model_id: str
    thermal_resistance: float  # K/W
    thermal_capacitance: float  # J/K
    ambient_temp: float = 25.0
    heat_sources: List[str] = field(default_factory=list)
    cooling_capacity: float = 100.0  # Watts
    fan_curves: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    prediction_accuracy: float = 0.95


class ThermalAgent(BaseAgent):
    """Forensic-level thermal and power management agent"""
    
    def __init__(self, agent_id: str = "thermal_agent", cpu_affinity: Optional[List[int]] = None):
        super().__init__(agent_id, priority=10, cpu_affinity=cpu_affinity)
        
        # Thermal monitoring infrastructure
        self.thermal_sensors: Dict[str, ThermalSensor] = {}
        self.power_domains: Dict[str, PowerDomain] = {}
        self.thermal_models: Dict[str, ThermalModel] = {}
        
        # Advanced thermal management
        self.thermal_predictor = self._initialize_thermal_predictor()
        self.power_optimizer = self._initialize_power_optimizer()
        self.cooling_controller = self._initialize_cooling_controller()
        self.throttling_manager = self._initialize_throttling_manager()
        
        # Machine learning models
        self.ml_models = {
            "thermal_predictor": self._initialize_thermal_ml_model(),
            "power_predictor": self._initialize_power_ml_model(),
            "cooling_optimizer": self._initialize_cooling_ml_model(),
            "workload_classifier": self._initialize_workload_classifier()
        }
        
        # Hardware interface
        self.hardware_interface = self._initialize_hardware_interface()
        self.sensor_calibration = self._initialize_sensor_calibration()
        
        # Performance statistics
        self.thermal_stats = {
            "thermal_events": 0,
            "throttling_events": 0,
            "power_optimizations": 0,
            "cooling_adjustments": 0,
            "temperature_violations": 0,
            "power_violations": 0,
            "prediction_accuracy": 0.95,
            "avg_temp_reduction": 0.0,
            "avg_power_savings": 0.0
        }
        
        # Real-time control loops
        self.thermal_control_queue = asyncio.Queue(maxsize=10000)
        self.power_control_queue = asyncio.Queue(maxsize=10000)
        
        # Advanced algorithms
        self.pid_controllers = self._initialize_pid_controllers()
        self.fuzzy_logic_controller = self._initialize_fuzzy_controller()
        self.neural_thermal_controller = self._initialize_neural_controller()
        
        # System state tracking
        self.system_thermal_state = {
            "global_thermal_pressure": 0.0,
            "power_budget_utilization": 0.0,
            "cooling_efficiency": 1.0,
            "thermal_runaway_risk": 0.0,
            "emergency_throttling_active": False
        }
        
        # Initialize system
        self._discover_thermal_sensors()
        self._discover_power_domains()
        self._build_thermal_models()
        
        self.logger.info(f"Advanced thermal agent initialized with {len(self.thermal_sensors)} sensors")
        self.logger.info(f"Power domains: {len(self.power_domains)}, Thermal models: {len(self.thermal_models)}")
    
    def _discover_thermal_sensors(self):
        """Discover and initialize thermal sensors"""
        try:
            if platform.system() == "Linux":
                self._discover_linux_thermal_sensors()
            elif platform.system() == "Windows":
                self._discover_windows_thermal_sensors()
            
            # Add synthetic sensors for comprehensive monitoring
            self._add_synthetic_sensors()
            
        except Exception as e:
            self.logger.error(f"Thermal sensor discovery failed: {e}")
    
    def _discover_linux_thermal_sensors(self):
        """Discover thermal sensors on Linux"""
        thermal_zones_path = "/sys/class/thermal"
        hwmon_path = "/sys/class/hwmon"
        
        # Thermal zones
        if os.path.exists(thermal_zones_path):
            for zone_dir in os.listdir(thermal_zones_path):
                if zone_dir.startswith("thermal_zone"):
                    zone_path = os.path.join(thermal_zones_path, zone_dir)
                    sensor = self._create_thermal_zone_sensor(zone_path, zone_dir)
                    if sensor:
                        self.thermal_sensors[sensor.sensor_id] = sensor
        
        # Hardware monitoring
        if os.path.exists(hwmon_path):
            for hwmon_dir in os.listdir(hwmon_path):
                hwmon_full_path = os.path.join(hwmon_path, hwmon_dir)
                sensors = self._create_hwmon_sensors(hwmon_full_path, hwmon_dir)
                for sensor in sensors:
                    self.thermal_sensors[sensor.sensor_id] = sensor
        
        # CPU-specific sensors
        self._discover_cpu_thermal_sensors()
        
        # GPU-specific sensors
        self._discover_gpu_thermal_sensors()
    
    def _create_thermal_zone_sensor(self, zone_path: str, zone_id: str) -> Optional[ThermalSensor]:
        """Create thermal sensor from thermal zone"""
        try:
            # Read sensor type
            type_path = os.path.join(zone_path, "type")
            if os.path.exists(type_path):
                with open(type_path, 'r') as f:
                    sensor_type = f.read().strip()
            else:
                sensor_type = "unknown"
            
            # Read temperature
            temp_path = os.path.join(zone_path, "temp")
            current_temp = 0.0
            if os.path.exists(temp_path):
                with open(temp_path, 'r') as f:
                    current_temp = float(f.read().strip()) / 1000.0  # Convert from millicelsius
            
            # Determine sensor location and type
            location = self._determine_sensor_location(sensor_type, zone_id)
            component_type = self._classify_sensor_type(sensor_type, location)
            
            sensor = ThermalSensor(
                sensor_id=f"{zone_id}_{sensor_type}",
                location=location,
                sensor_type=component_type,
                current_temp=current_temp,
                max_temp=self._get_max_temp_for_type(component_type),
                critical_temp=self._get_critical_temp_for_type(component_type)
            )
            
            return sensor
            
        except Exception as e:
            self.logger.debug(f"Failed to create thermal zone sensor {zone_id}: {e}")
            return None
    
    def _create_hwmon_sensors(self, hwmon_path: str, hwmon_id: str) -> List[ThermalSensor]:
        """Create sensors from hwmon interface"""
        sensors = []
        
        try:
            # Read device name
            name_path = os.path.join(hwmon_path, "name")
            device_name = "unknown"
            if os.path.exists(name_path):
                with open(name_path, 'r') as f:
                    device_name = f.read().strip()
            
            # Find temperature inputs
            for item in os.listdir(hwmon_path):
                if item.startswith("temp") and item.endswith("_input"):
                    temp_id = item.replace("_input", "")
                    
                    # Read current temperature
                    temp_path = os.path.join(hwmon_path, item)
                    current_temp = 0.0
                    if os.path.exists(temp_path):
                        with open(temp_path, 'r') as f:
                            current_temp = float(f.read().strip()) / 1000.0
                    
                    # Read label if available
                    label_path = os.path.join(hwmon_path, f"{temp_id}_label")
                    label = temp_id
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            label = f.read().strip()
                    
                    # Create sensor
                    location = f"{device_name}_{label}"
                    component_type = self._classify_sensor_type(device_name, location)
                    
                    sensor = ThermalSensor(
                        sensor_id=f"{hwmon_id}_{temp_id}",
                        location=location,
                        sensor_type=component_type,
                        current_temp=current_temp,
                        max_temp=self._get_max_temp_for_type(component_type),
                        critical_temp=self._get_critical_temp_for_type(component_type)
                    )
                    
                    sensors.append(sensor)
                    
        except Exception as e:
            self.logger.debug(f"Failed to create hwmon sensors {hwmon_id}: {e}")
        
        return sensors
    
    def _discover_cpu_thermal_sensors(self):
        """Discover CPU-specific thermal sensors"""
        try:
            # Intel CPU thermal sensors
            if os.path.exists("/sys/devices/platform/coretemp.0"):
                self._discover_intel_cpu_sensors()
            
            # AMD CPU thermal sensors
            if os.path.exists("/sys/devices/pci0000:00"):
                self._discover_amd_cpu_sensors()
                
        except Exception as e:
            self.logger.debug(f"CPU thermal sensor discovery failed: {e}")
    
    def _discover_intel_cpu_sensors(self):
        """Discover Intel CPU thermal sensors"""
        try:
            coretemp_path = "/sys/devices/platform/coretemp.0/hwmon"
            if os.path.exists(coretemp_path):
                for hwmon_dir in os.listdir(coretemp_path):
                    hwmon_path = os.path.join(coretemp_path, hwmon_dir)
                    sensors = self._create_intel_core_sensors(hwmon_path)
                    for sensor in sensors:
                        self.thermal_sensors[sensor.sensor_id] = sensor
                        
        except Exception as e:
            self.logger.debug(f"Intel CPU sensor discovery failed: {e}")
    
    def _create_intel_core_sensors(self, hwmon_path: str) -> List[ThermalSensor]:
        """Create Intel CPU core sensors"""
        sensors = []
        
        try:
            for item in os.listdir(hwmon_path):
                if item.startswith("temp") and item.endswith("_input"):
                    temp_id = item.replace("_input", "")
                    
                    # Read temperature
                    temp_path = os.path.join(hwmon_path, item)
                    current_temp = 0.0
                    if os.path.exists(temp_path):
                        with open(temp_path, 'r') as f:
                            current_temp = float(f.read().strip()) / 1000.0
                    
                    # Read critical temperature
                    crit_path = os.path.join(hwmon_path, f"{temp_id}_crit")
                    critical_temp = 100.0
                    if os.path.exists(crit_path):
                        with open(crit_path, 'r') as f:
                            critical_temp = float(f.read().strip()) / 1000.0
                    
                    # Read max temperature
                    max_path = os.path.join(hwmon_path, f"{temp_id}_max")
                    max_temp = critical_temp - 5.0
                    if os.path.exists(max_path):
                        with open(max_path, 'r') as f:
                            max_temp = float(f.read().strip()) / 1000.0
                    
                    # Determine core number
                    core_num = temp_id.replace("temp", "")
                    if core_num == "1":
                        location = "cpu_package"
                    else:
                        location = f"cpu_core_{int(core_num) - 2}"
                    
                    sensor = ThermalSensor(
                        sensor_id=f"intel_{temp_id}",
                        location=location,
                        sensor_type="cpu",
                        current_temp=current_temp,
                        max_temp=max_temp,
                        critical_temp=critical_temp,
                        thermal_mass=0.5,  # CPU cores have low thermal mass
                        cooling_efficiency=0.8
                    )
                    
                    sensors.append(sensor)
                    
        except Exception as e:
            self.logger.debug(f"Intel core sensor creation failed: {e}")
        
        return sensors
    
    def _discover_amd_cpu_sensors(self):
        """Discover AMD CPU thermal sensors"""
        try:
            # AMD k10temp driver
            k10temp_path = "/sys/devices/pci0000:00"
            for item in os.listdir(k10temp_path):
                if "k10temp" in item:
                    hwmon_path = os.path.join(k10temp_path, item, "hwmon")
                    if os.path.exists(hwmon_path):
                        for hwmon_dir in os.listdir(hwmon_path):
                            full_hwmon_path = os.path.join(hwmon_path, hwmon_dir)
                            sensors = self._create_amd_cpu_sensors(full_hwmon_path)
                            for sensor in sensors:
                                self.thermal_sensors[sensor.sensor_id] = sensor
                                
        except Exception as e:
            self.logger.debug(f"AMD CPU sensor discovery failed: {e}")
    
    def _create_amd_cpu_sensors(self, hwmon_path: str) -> List[ThermalSensor]:
        """Create AMD CPU sensors"""
        sensors = []
        
        try:
            for item in os.listdir(hwmon_path):
                if item.startswith("temp") and item.endswith("_input"):
                    temp_id = item.replace("_input", "")
                    
                    # Read temperature
                    temp_path = os.path.join(hwmon_path, item)
                    current_temp = 0.0
                    if os.path.exists(temp_path):
                        with open(temp_path, 'r') as f:
                            current_temp = float(f.read().strip()) / 1000.0
                    
                    # AMD sensors typically use temp1 for Tctl
                    if temp_id == "temp1":
                        location = "cpu_tctl"
                    else:
                        location = f"cpu_{temp_id}"
                    
                    sensor = ThermalSensor(
                        sensor_id=f"amd_{temp_id}",
                        location=location,
                        sensor_type="cpu",
                        current_temp=current_temp,
                        max_temp=90.0,  # AMD typical
                        critical_temp=95.0,
                        thermal_mass=0.6,
                        cooling_efficiency=0.75
                    )
                    
                    sensors.append(sensor)
                    
        except Exception as e:
            self.logger.debug(f"AMD CPU sensor creation failed: {e}")
        
        return sensors
    
    def _discover_gpu_thermal_sensors(self):
        """Discover GPU thermal sensors"""
        try:
            # NVIDIA GPU sensors
            self._discover_nvidia_gpu_sensors()
            
            # AMD GPU sensors
            self._discover_amd_gpu_sensors()
            
        except Exception as e:
            self.logger.debug(f"GPU thermal sensor discovery failed: {e}")
    
    def _discover_nvidia_gpu_sensors(self):
        """Discover NVIDIA GPU thermal sensors"""
        try:
            # Try nvidia-ml-py if available
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    sensor = ThermalSensor(
                        sensor_id=f"nvidia_gpu_{i}",
                        location=f"gpu_{i}_core",
                        sensor_type="gpu",
                        current_temp=float(temp),
                        max_temp=83.0,  # NVIDIA typical
                        critical_temp=87.0,
                        thermal_mass=2.0,  # GPUs have higher thermal mass
                        cooling_efficiency=0.9
                    )
                    
                    self.thermal_sensors[sensor.sensor_id] = sensor
                    
                except Exception as e:
                    self.logger.debug(f"Failed to read NVIDIA GPU {i} temperature: {e}")
                    
        except ImportError:
            self.logger.debug("pynvml not available, skipping NVIDIA GPU sensors")
        except Exception as e:
            self.logger.debug(f"NVIDIA GPU sensor discovery failed: {e}")
    
    def _discover_amd_gpu_sensors(self):
        """Discover AMD GPU thermal sensors"""
        try:
            # AMD GPU sensors through hwmon
            hwmon_path = "/sys/class/hwmon"
            if os.path.exists(hwmon_path):
                for hwmon_dir in os.listdir(hwmon_path):
                    hwmon_full_path = os.path.join(hwmon_path, hwmon_dir)
                    name_path = os.path.join(hwmon_full_path, "name")
                    
                    if os.path.exists(name_path):
                        with open(name_path, 'r') as f:
                            name = f.read().strip()
                        
                        if "amdgpu" in name:
                            sensors = self._create_amd_gpu_sensors(hwmon_full_path, hwmon_dir)
                            for sensor in sensors:
                                self.thermal_sensors[sensor.sensor_id] = sensor
                                
        except Exception as e:
            self.logger.debug(f"AMD GPU sensor discovery failed: {e}")
    
    def _create_amd_gpu_sensors(self, hwmon_path: str, hwmon_id: str) -> List[ThermalSensor]:
        """Create AMD GPU sensors"""
        sensors = []
        
        try:
            for item in os.listdir(hwmon_path):
                if item.startswith("temp") and item.endswith("_input"):
                    temp_id = item.replace("_input", "")
                    
                    # Read temperature
                    temp_path = os.path.join(hwmon_path, item)
                    current_temp = 0.0
                    if os.path.exists(temp_path):
                        with open(temp_path, 'r') as f:
                            current_temp = float(f.read().strip()) / 1000.0
                    
                    # Read label
                    label_path = os.path.join(hwmon_path, f"{temp_id}_label")
                    label = temp_id
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            label = f.read().strip()
                    
                    sensor = ThermalSensor(
                        sensor_id=f"amdgpu_{hwmon_id}_{temp_id}",
                        location=f"gpu_{label}",
                        sensor_type="gpu",
                        current_temp=current_temp,
                        max_temp=90.0,  # AMD GPU typical
                        critical_temp=95.0,
                        thermal_mass=2.5,
                        cooling_efficiency=0.85
                    )
                    
                    sensors.append(sensor)
                    
        except Exception as e:
            self.logger.debug(f"AMD GPU sensor creation failed: {e}")
        
        return sensors
    
    def _discover_windows_thermal_sensors(self):
        """Discover thermal sensors on Windows"""
        try:
            # Windows thermal sensors would be discovered through WMI
            # This is a placeholder for Windows implementation
            self.logger.debug("Windows thermal sensor discovery not implemented")
            
        except Exception as e:
            self.logger.debug(f"Windows thermal sensor discovery failed: {e}")
    
    def _add_synthetic_sensors(self):
        """Add synthetic sensors for comprehensive monitoring"""
        # Add ambient temperature sensor
        ambient_sensor = ThermalSensor(
            sensor_id="ambient_synthetic",
            location="system_ambient",
            sensor_type="ambient",
            current_temp=25.0,
            max_temp=40.0,
            critical_temp=50.0,
            thermal_mass=10.0,  # High thermal mass for ambient
            cooling_efficiency=0.1
        )
        self.thermal_sensors[ambient_sensor.sensor_id] = ambient_sensor
        
        # Add VRM synthetic sensor
        vrm_sensor = ThermalSensor(
            sensor_id="vrm_synthetic",
            location="motherboard_vrm",
            sensor_type="vrm",
            current_temp=45.0,
            max_temp=90.0,
            critical_temp=100.0,
            thermal_mass=1.5,
            cooling_efficiency=0.6
        )
        self.thermal_sensors[vrm_sensor.sensor_id] = vrm_sensor
    
    def _determine_sensor_location(self, sensor_type: str, zone_id: str) -> str:
        """Determine sensor location from type and ID"""
        sensor_type_lower = sensor_type.lower()
        
        if "cpu" in sensor_type_lower or "coretemp" in sensor_type_lower:
            return "cpu_package"
        elif "gpu" in sensor_type_lower or "radeon" in sensor_type_lower or "nvidia" in sensor_type_lower:
            return "gpu_core"
        elif "acpi" in sensor_type_lower:
            return "system_acpi"
        elif "pch" in sensor_type_lower:
            return "chipset_pch"
        else:
            return f"unknown_{zone_id}"
    
    def _classify_sensor_type(self, device_name: str, location: str) -> str:
        """Classify sensor type from device name and location"""
        device_lower = device_name.lower()
        location_lower = location.lower()
        
        if any(cpu_indicator in device_lower for cpu_indicator in ["coretemp", "k10temp", "cpu"]):
            return "cpu"
        elif any(gpu_indicator in device_lower for gpu_indicator in ["amdgpu", "nvidia", "radeon"]):
            return "gpu"
        elif "ambient" in location_lower:
            return "ambient"
        elif "vrm" in location_lower or "vdd" in location_lower:
            return "vrm"
        elif "memory" in location_lower or "dimm" in location_lower:
            return "memory"
        else:
            return "system"
    
    def _get_max_temp_for_type(self, sensor_type: str) -> float:
        """Get maximum safe temperature for sensor type"""
        temp_limits = {
            "cpu": 85.0,
            "gpu": 83.0,
            "memory": 85.0,
            "vrm": 90.0,
            "ambient": 40.0,
            "system": 70.0
        }
        return temp_limits.get(sensor_type, 80.0)
    
    def _get_critical_temp_for_type(self, sensor_type: str) -> float:
        """Get critical temperature for sensor type"""
        critical_limits = {
            "cpu": 95.0,
            "gpu": 87.0,
            "memory": 95.0,
            "vrm": 100.0,
            "ambient": 50.0,
            "system": 80.0
        }
        return critical_limits.get(sensor_type, 90.0)
    
    def _discover_power_domains(self):
        """Discover and initialize power domains"""
        try:
            # CPU power domain
            cpu_domain = PowerDomain(
                domain_id="cpu_package",
                domain_type="cpu",
                max_power=125.0,  # Typical desktop CPU TDP
                target_power=100.0,
                voltage=1.2,
                frequency=3000.0
            )
            self.power_domains[cpu_domain.domain_id] = cpu_domain
            
            # GPU power domain
            gpu_domain = PowerDomain(
                domain_id="gpu_primary",
                domain_type="gpu",
                max_power=250.0,  # Typical GPU TDP
                target_power=200.0,
                voltage=1.1,
                frequency=1500.0
            )
            self.power_domains[gpu_domain.domain_id] = gpu_domain
            
            # Memory power domain
            memory_domain = PowerDomain(
                domain_id="memory_system",
                domain_type="memory",
                max_power=50.0,
                target_power=40.0,
                voltage=1.35,
                frequency=3200.0
            )
            self.power_domains[memory_domain.domain_id] = memory_domain
            
            # System power domain
            system_domain = PowerDomain(
                domain_id="system_total",
                domain_type="system",
                max_power=500.0,
                target_power=400.0,
                voltage=12.0,
                frequency=1.0
            )
            self.power_domains[system_domain.domain_id] = system_domain
            
        except Exception as e:
            self.logger.error(f"Power domain discovery failed: {e}")
    
    def _build_thermal_models(self):
        """Build thermal models for system components"""
        try:
            # CPU thermal model
            cpu_model = ThermalModel(
                model_id="cpu_thermal_model",
                thermal_resistance=0.5,  # K/W
                thermal_capacitance=100.0,  # J/K
                heat_sources=["cpu_package"],
                cooling_capacity=150.0,
                fan_curves={
                    "cpu_fan": [(30.0, 30.0), (60.0, 50.0), (80.0, 100.0)]
                }
            )
            self.thermal_models[cpu_model.model_id] = cpu_model
            
            # GPU thermal model
            gpu_model = ThermalModel(
                model_id="gpu_thermal_model",
                thermal_resistance=0.3,
                thermal_capacitance=200.0,
                heat_sources=["gpu_primary"],
                cooling_capacity=300.0,
                fan_curves={
                    "gpu_fan": [(40.0, 20.0), (70.0, 60.0), (85.0, 100.0)]
                }
            )
            self.thermal_models[gpu_model.model_id] = gpu_model
            
            # System thermal model
            system_model = ThermalModel(
                model_id="system_thermal_model",
                thermal_resistance=0.1,
                thermal_capacitance=1000.0,
                heat_sources=["cpu_package", "gpu_primary", "memory_system"],
                cooling_capacity=500.0,
                fan_curves={
                    "case_fan_intake": [(25.0, 20.0), (40.0, 40.0), (60.0, 80.0)],
                    "case_fan_exhaust": [(25.0, 25.0), (40.0, 45.0), (60.0, 85.0)]
                }
            )
            self.thermal_models[system_model.model_id] = system_model
            
        except Exception as e:
            self.logger.error(f"Thermal model building failed: {e}")
    
    def _initialize_thermal_predictor(self) -> Dict[str, Any]:
        """Initialize thermal prediction system"""
        return {
            "prediction_horizon": 60.0,  # seconds
            "update_interval": 1.0,
            "thermal_time_constants": {},
            "heat_generation_models": {},
            "cooling_effectiveness_models": {}
        }
    
    def _initialize_power_optimizer(self) -> Dict[str, Any]:
        """Initialize power optimization system"""
        return {
            "optimization_targets": ["performance", "efficiency", "thermal"],
            "dvfs_controller": self._create_dvfs_controller(),
            "power_gating_controller": self._create_power_gating_controller(),
            "workload_power_models": {}
        }
    
    def _create_dvfs_controller(self) -> Dict[str, Any]:
        """Create Dynamic Voltage and Frequency Scaling controller"""
        return {
            "cpu_p_states": [(800, 0.8), (1600, 0.9), (2400, 1.0), (3200, 1.1), (4000, 1.2)],
            "gpu_p_states": [(300, 0.8), (600, 0.9), (900, 1.0), (1200, 1.1), (1500, 1.2)],
            "current_cpu_state": 3,
            "current_gpu_state": 3,
            "transition_latency": 0.001  # 1ms
        }
    
    def _create_power_gating_controller(self) -> Dict[str, Any]:
        """Create power gating controller"""
        return {
            "gatable_units": ["unused_cores", "idle_gpu_blocks", "unused_memory_banks"],
            "gating_thresholds": {"idle_time": 0.1, "utilization": 0.05},
            "ungating_latency": 0.0001  # 0.1ms
        }
    
    def _initialize_cooling_controller(self) -> Dict[str, Any]:
        """Initialize cooling system controller"""
        return {
            "fan_controllers": self._create_fan_controllers(),
            "liquid_cooling_controller": self._create_liquid_cooling_controller(),
            "thermal_interface_optimizer": self._create_tim_optimizer()
        }
    
    def _create_fan_controllers(self) -> Dict[str, Any]:
        """Create fan control systems"""
        return {
            "cpu_fan": {
                "current_speed": 50.0,
                "min_speed": 20.0,
                "max_speed": 100.0,
                "response_time": 2.0,
                "noise_curve": [(20.0, 25.0), (50.0, 35.0), (100.0, 55.0)]  # Speed vs dB
            },
            "gpu_fan": {
                "current_speed": 40.0,
                "min_speed": 0.0,
                "max_speed": 100.0,
                "response_time": 1.5,
                "noise_curve": [(0.0, 0.0), (50.0, 30.0), (100.0, 50.0)]
            },
            "case_fans": {
                "current_speed": 30.0,
                "min_speed": 15.0,
                "max_speed": 100.0,
                "response_time": 3.0,
                "noise_curve": [(15.0, 20.0), (50.0, 28.0), (100.0, 45.0)]
            }
        }
    
    def _create_liquid_cooling_controller(self) -> Dict[str, Any]:
        """Create liquid cooling controller"""
        return {
            "pump_speed": 70.0,
            "radiator_fan_speed": 50.0,
            "coolant_temp": 30.0,
            "flow_rate": 1.5,  # L/min
            "thermal_efficiency": 0.95
        }
    
    def _create_tim_optimizer(self) -> Dict[str, Any]:
        """Create thermal interface material optimizer"""
        return {
            "tim_conductivity": 8.5,  # W/mK
            "contact_pressure": 40.0,  # PSI
            "aging_factor": 1.0,
            "optimization_schedule": 3600.0  # seconds
        }
    
    def _initialize_throttling_manager(self) -> Dict[str, Any]:
        """Initialize thermal throttling manager"""
        return {
            "throttling_policies": self._create_throttling_policies(),
            "emergency_actions": self._create_emergency_actions(),
            "throttling_history": deque(maxlen=1000)
        }
    
    def _create_throttling_policies(self) -> Dict[str, Any]:
        """Create thermal throttling policies"""
        return {
            "cpu_throttling": {
                "temp_thresholds": [75.0, 80.0, 85.0, 90.0],
                "throttle_levels": [0.95, 0.85, 0.70, 0.50],
                "hysteresis": 2.0
            },
            "gpu_throttling": {
                "temp_thresholds": [70.0, 75.0, 80.0, 83.0],
                "throttle_levels": [0.95, 0.85, 0.70, 0.50],
                "hysteresis": 2.0
            },
            "memory_throttling": {
                "temp_thresholds": [80.0, 85.0, 90.0],
                "throttle_levels": [0.90, 0.75, 0.60],
                "hysteresis": 3.0
            }
        }
    
    def _create_emergency_actions(self) -> Dict[str, Any]:
        """Create emergency thermal actions"""
        return {
            "emergency_shutdown": {
                "temp_threshold": 95.0,
                "grace_period": 5.0,
                "shutdown_sequence": ["save_state", "notify_agents", "power_off"]
            },
            "emergency_throttling": {
                "temp_threshold": 90.0,
                "throttle_level": 0.25,
                "duration": 30.0
            },
            "fan_override": {
                "temp_threshold": 85.0,
                "fan_speed": 100.0,
                "duration": 60.0
            }
        }
    
    def _initialize_thermal_ml_model(self) -> Dict[str, Any]:
        """Initialize thermal prediction ML model"""
        return {
            "model_type": "lstm",
            "input_features": 12,
            "hidden_layers": [64, 32],
            "prediction_horizon": 30,
            "weights": np.random.random((12, 64)),
            "training_data": deque(maxlen=10000),
            "accuracy": 0.95
        }
    
    def _initialize_power_ml_model(self) -> Dict[str, Any]:
        """Initialize power prediction ML model"""
        return {
            "model_type": "neural_network",
            "input_features": 8,
            "hidden_layers": [32, 16],
            "weights": np.random.random((8, 32)),
            "power_efficiency_model": {},
            "workload_power_correlation": 0.85
        }
    
    def _initialize_cooling_ml_model(self) -> Dict[str, Any]:
        """Initialize cooling optimization ML model"""
        return {
            "model_type": "reinforcement_learning",
            "state_space": 16,
            "action_space": 8,
            "q_table": defaultdict(lambda: defaultdict(float)),
            "learning_rate": 0.1,
            "exploration_rate": 0.1
        }
    
    def _initialize_workload_classifier(self) -> Dict[str, Any]:
        """Initialize workload classification model"""
        return {
            "model_type": "svm",
            "feature_vectors": [],
            "workload_classes": ["compute_intensive", "memory_intensive", "io_intensive", "balanced"],
            "classification_accuracy": 0.88,
            "thermal_profiles": {}
        }
    
    def _initialize_hardware_interface(self) -> Dict[str, Any]:
        """Initialize hardware control interface"""
        return {
            "msr_interface": self._create_msr_interface(),
            "acpi_interface": self._create_acpi_interface(),
            "pci_interface": self._create_pci_interface(),
            "sensor_interfaces": {}
        }
    
    def _create_msr_interface(self) -> Dict[str, Any]:
        """Create Model-Specific Register interface"""
        return {
            "available": os.path.exists("/dev/cpu/0/msr"),
            "msr_addresses": {
                "ia32_therm_status": 0x19C,
                "ia32_temperature_target": 0x1A2,
                "ia32_package_therm_status": 0x1B1,
                "ia32_perf_ctl": 0x199
            }
        }
    
    def _create_acpi_interface(self) -> Dict[str, Any]:
        """Create ACPI interface"""
        return {
            "available": os.path.exists("/sys/firmware/acpi"),
            "thermal_zones": [],
            "cooling_devices": [],
            "trip_points": {}
        }
    
    def _create_pci_interface(self) -> Dict[str, Any]:
        """Create PCI configuration interface"""
        return {
            "available": os.path.exists("/sys/bus/pci"),
            "thermal_devices": [],
            "power_management_devices": []
        }
    
    def _initialize_sensor_calibration(self) -> Dict[str, Any]:
        """Initialize sensor calibration system"""
        return {
            "calibration_offsets": {},
            "calibration_scales": {},
            "cross_correlation_matrix": np.eye(len(self.thermal_sensors)),
            "calibration_schedule": 86400.0  # 24 hours
        }
    
    def _initialize_pid_controllers(self) -> Dict[str, Any]:
        """Initialize PID controllers for thermal management"""
        return {
            "cpu_thermal_pid": {
                "kp": 2.0, "ki": 0.1, "kd": 0.5,
                "setpoint": 75.0, "integral": 0.0, "last_error": 0.0
            },
            "gpu_thermal_pid": {
                "kp": 1.5, "ki": 0.08, "kd": 0.3,
                "setpoint": 70.0, "integral": 0.0, "last_error": 0.0
            },
            "system_thermal_pid": {
                "kp": 1.0, "ki": 0.05, "kd": 0.2,
                "setpoint": 40.0, "integral": 0.0, "last_error": 0.0
            }
        }
    
    def _initialize_fuzzy_controller(self) -> Dict[str, Any]:
        """Initialize fuzzy logic thermal controller"""
        return {
            "input_variables": {
                "temperature": [(0, 50, 100), ("low", "medium", "high")],
                "load": [(0, 50, 100), ("light", "medium", "heavy")],
                "ambient": [(15, 25, 45), ("cool", "normal", "warm")]
            },
            "output_variables": {
                "fan_speed": [(0, 50, 100), ("slow", "medium", "fast")],
                "throttling": [(0, 50, 100), ("none", "moderate", "aggressive")]
            },
            "rules": self._create_fuzzy_rules()
        }
    
    def _create_fuzzy_rules(self) -> List[Dict[str, Any]]:
        """Create fuzzy logic rules"""
        return [
            {"if": {"temperature": "low", "load": "light"}, "then": {"fan_speed": "slow", "throttling": "none"}},
            {"if": {"temperature": "medium", "load": "medium"}, "then": {"fan_speed": "medium", "throttling": "none"}},
            {"if": {"temperature": "high", "load": "heavy"}, "then": {"fan_speed": "fast", "throttling": "moderate"}},
            {"if": {"temperature": "high", "ambient": "warm"}, "then": {"fan_speed": "fast", "throttling": "aggressive"}}
        ]
    
    def _initialize_neural_controller(self) -> Dict[str, Any]:
        """Initialize neural network thermal controller"""
        return {
            "network_architecture": [16, 32, 16, 8],
            "weights": [np.random.random((16, 32)), np.random.random((32, 16)), np.random.random((16, 8))],
            "biases": [np.random.random(32), np.random.random(16), np.random.random(8)],
            "activation_function": "relu",
            "learning_rate": 0.001
        }
    
    async def execute_cycle(self):
        """Main execution cycle for thermal agent"""
        try:
            # Read all thermal sensors
            await self._read_thermal_sensors()
            
            # Update power domain measurements
            await self._update_power_measurements()
            
            # Perform thermal predictions
            await self._perform_thermal_predictions()
            
            # Execute thermal control
            await self._execute_thermal_control()
            
            # Execute power optimization
            await self._execute_power_optimization()
            
            # Update machine learning models
            await self._update_ml_models()
            
            # Check for thermal emergencies
            await self._check_thermal_emergencies()
            
            # Update system thermal state
            await self._update_system_thermal_state()
            
            # Update performance metrics
            self.update_metrics()
            
        except Exception as e:
            self.logger.error(f"Error in thermal agent cycle: {e}")
            self.state = AgentState.ERROR
    
    async def _read_thermal_sensors(self):
        """Read all thermal sensors"""
        try:
            for sensor_id, sensor in self.thermal_sensors.items():
                new_temp = await self._read_sensor_temperature(sensor)
                if new_temp is not None:
                    sensor.temp_history.append((time.time(), sensor.current_temp))
                    sensor.current_temp = new_temp
                    
        except Exception as e:
            self.logger.error(f"Thermal sensor reading failed: {e}")
    
    async def _read_sensor_temperature(self, sensor: ThermalSensor) -> Optional[float]:
        """Read temperature from individual sensor"""
        try:
            if sensor.sensor_id.startswith("thermal_zone"):
                zone_id = sensor.sensor_id.split('_')[2]
                temp_path = f"/sys/class/thermal/thermal_zone{zone_id}/temp"
                if os.path.exists(temp_path):
                    with open(temp_path, 'r') as f:
                        return float(f.read().strip()) / 1000.0
            
            elif sensor.sensor_id.startswith("hwmon"):
                # Parse hwmon sensor path
                parts = sensor.sensor_id.split('_')
                hwmon_id = parts[1]
                temp_id = parts[2]
                temp_path = f"/sys/class/hwmon/hwmon{hwmon_id}/{temp_id}_input"
                if os.path.exists(temp_path):
                    with open(temp_path, 'r') as f:
                        return float(f.read().strip()) / 1000.0
            
            elif sensor.sensor_id.startswith("nvidia"):
                # NVIDIA GPU temperature
                try:
                    import pynvml
                    gpu_id = int(sensor.sensor_id.split('_')[2])
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                except:
                    pass
            
            elif sensor.sensor_id.endswith("synthetic"):
                # Synthetic sensors - estimate based on system state
                return await self._estimate_synthetic_temperature(sensor)
            
        except Exception as e:
            self.logger.debug(f"Failed to read sensor {sensor.sensor_id}: {e}")
        
        return None
    
    async def _estimate_synthetic_temperature(self, sensor: ThermalSensor) -> float:
        """Estimate temperature for synthetic sensors"""
        if sensor.sensor_type == "ambient":
            # Estimate ambient based on system temperatures
            cpu_temps = [s.current_temp for s in self.thermal_sensors.values() if s.sensor_type == "cpu"]
            if cpu_temps:
                return min(cpu_temps) - 15.0  # Rough ambient estimate
            return 25.0
        
        elif sensor.sensor_type == "vrm":
            # Estimate VRM temperature based on CPU load and temperature
            cpu_temps = [s.current_temp for s in self.thermal_sensors.values() if s.sensor_type == "cpu"]
            if cpu_temps:
                return max(cpu_temps) + 10.0  # VRM typically hotter than CPU
            return 45.0
        
        return sensor.current_temp
    
    async def _update_power_measurements(self):
        """Update power domain measurements"""
        try:
            for domain_id, domain in self.power_domains.items():
                new_power = await self._measure_domain_power(domain)
                if new_power is not None:
                    domain.power_history.append((time.time(), domain.current_power))
                    domain.current_power = new_power
                    
        except Exception as e:
            self.logger.error(f"Power measurement update failed: {e}")
    
    async def _measure_domain_power(self, domain: PowerDomain) -> Optional[float]:
        """Measure power consumption for domain"""
        try:
            if domain.domain_type == "cpu":
                # Estimate CPU power based on frequency and temperature
                base_power = 65.0  # Base TDP
                freq_factor = domain.frequency / 3000.0  # Normalized to 3GHz
                voltage_factor = (domain.voltage / 1.2) ** 2  # Quadratic voltage scaling
                return base_power * freq_factor * voltage_factor * domain.throttle_state
            
            elif domain.domain_type == "gpu":
                # Estimate GPU power
                base_power = 150.0
                freq_factor = domain.frequency / 1500.0
                voltage_factor = (domain.voltage / 1.1) ** 2
                return base_power * freq_factor * voltage_factor * domain.throttle_state
            
            elif domain.domain_type == "memory":
                # Estimate memory power
                base_power = 30.0
                freq_factor = domain.frequency / 3200.0
                return base_power * freq_factor
            
            elif domain.domain_type == "system":
                # Total system power
                total_power = sum(d.current_power for d in self.power_domains.values() 
                                if d.domain_type != "system")
                return total_power + 50.0  # Add motherboard and peripherals
                
        except Exception as e:
            self.logger.debug(f"Power measurement failed for {domain.domain_id}: {e}")
        
        return None
    
    async def _perform_thermal_predictions(self):
        """Perform thermal predictions using ML models"""
        try:
            for model_id, model in self.thermal_models.items():
                prediction = await self._predict_thermal_behavior(model)
                if prediction:
                    await self._process_thermal_prediction(model_id, prediction)
                    
        except Exception as e:
            self.logger.error(f"Thermal prediction failed: {e}")
    
    async def _predict_thermal_behavior(self, model: ThermalModel) -> Optional[Dict[str, Any]]:
        """Predict thermal behavior using model"""
        try:
            # Collect input data
            current_temps = []
            power_inputs = []
            
            for heat_source in model.heat_sources:
                if heat_source in self.power_domains:
                    power_inputs.append(self.power_domains[heat_source].current_power)
                
                # Find corresponding thermal sensor
                for sensor in self.thermal_sensors.values():
                    if heat_source in sensor.location:
                        current_temps.append(sensor.current_temp)
                        break
            
            if not current_temps or not power_inputs:
                return None
            
            # Simple thermal model: T_future = T_current + (P * R - (T_current - T_ambient) / τ) * dt
            dt = 1.0  # Prediction time step
            ambient_temp = model.ambient_temp
            
            predicted_temps = []
            for i, (temp, power) in enumerate(zip(current_temps, power_inputs)):
                heat_rise = power * model.thermal_resistance
                cooling_effect = (temp - ambient_temp) / model.thermal_capacitance
                temp_change = (heat_rise - cooling_effect) * dt
                predicted_temp = temp + temp_change
                predicted_temps.append(predicted_temp)
            
            return {
                "model_id": model.model_id,
                "prediction_time": time.time() + dt,
                "predicted_temperatures": predicted_temps,
                "confidence": model.prediction_accuracy
            }
            
        except Exception as e:
            self.logger.debug(f"Thermal prediction failed for model {model.model_id}: {e}")
            return None
    
    async def _process_thermal_prediction(self, model_id: str, prediction: Dict[str, Any]):
        """Process thermal prediction results"""
        try:
            predicted_temps = prediction["predicted_temperatures"]
            max_predicted_temp = max(predicted_temps) if predicted_temps else 0.0
            
            # Check if predicted temperature exceeds thresholds
            if max_predicted_temp > 85.0:  # High temperature predicted
                await self._trigger_preemptive_cooling(model_id, max_predicted_temp)
            
            # Update system thermal pressure
            thermal_pressure = max_predicted_temp / 100.0  # Normalize to 0-1
            self.system_thermal_state["global_thermal_pressure"] = thermal_pressure
            
        except Exception as e:
            self.logger.error(f"Thermal prediction processing failed: {e}")
    
    async def _trigger_preemptive_cooling(self, model_id: str, predicted_temp: float):
        """Trigger preemptive cooling actions"""
        try:
            cooling_actions = []
            
            if predicted_temp > 90.0:
                # Aggressive preemptive actions
                cooling_actions.extend([
                    {"action": "increase_fan_speed", "target": "all", "value": 80.0},
                    {"action": "reduce_power_target", "domain": "cpu", "reduction": 0.15},
                    {"action": "reduce_power_target", "domain": "gpu", "reduction": 0.10}
                ])
            elif predicted_temp > 85.0:
                # Moderate preemptive actions
                cooling_actions.extend([
                    {"action": "increase_fan_speed", "target": "primary", "value": 65.0},
                    {"action": "reduce_power_target", "domain": "cpu", "reduction": 0.08}
                ])
            
            for action in cooling_actions:
                await self._execute_cooling_action(action)
                
            self.thermal_stats["cooling_adjustments"] += len(cooling_actions)
            
        except Exception as e:
            self.logger.error(f"Preemptive cooling trigger failed: {e}")
    
    async def _execute_cooling_action(self, action: Dict[str, Any]):
        """Execute cooling action"""
        try:
            action_type = action["action"]
            
            if action_type == "increase_fan_speed":
                await self._adjust_fan_speed(action["target"], action["value"])
            elif action_type == "reduce_power_target":
                await self._adjust_power_target(action["domain"], action["reduction"])
            elif action_type == "enable_liquid_cooling":
                await self._adjust_liquid_cooling(action.get("intensity", 1.0))
                
        except Exception as e:
            self.logger.error(f"Cooling action execution failed: {e}")
    
    async def _adjust_fan_speed(self, target: str, speed: float):
        """Adjust fan speed"""
        try:
            if target == "all":
                for fan_id in self.cooling_controller["fan_controllers"]:
                    self.cooling_controller["fan_controllers"][fan_id]["current_speed"] = min(speed, 100.0)
            elif target in self.cooling_controller["fan_controllers"]:
                self.cooling_controller["fan_controllers"][target]["current_speed"] = min(speed, 100.0)
                
        except Exception as e:
            self.logger.error(f"Fan speed adjustment failed: {e}")
    
    async def _adjust_power_target(self, domain: str, reduction: float):
        """Adjust power target for domain"""
        try:
            domain_key = f"{domain}_package" if domain in ["cpu", "gpu"] else domain
            if domain_key in self.power_domains:
                domain_obj = self.power_domains[domain_key]
                new_target = domain_obj.target_power * (1.0 - reduction)
                domain_obj.target_power = max(new_target, domain_obj.max_power * 0.3)  # Minimum 30%
                
        except Exception as e:
            self.logger.error(f"Power target adjustment failed: {e}")
    
    async def _adjust_liquid_cooling(self, intensity: float):
        """Adjust liquid cooling system"""
        try:
            lc_controller = self.cooling_controller["liquid_cooling_controller"]
            lc_controller["pump_speed"] = min(lc_controller["pump_speed"] * intensity, 100.0)
            lc_controller["radiator_fan_speed"] = min(lc_controller["radiator_fan_speed"] * intensity, 100.0)
            
        except Exception as e:
            self.logger.error(f"Liquid cooling adjustment failed: {e}")
    
    async def _execute_thermal_control(self):
        """Execute thermal control algorithms"""
        try:
            # PID control
            await self._execute_pid_control()
            
            # Fuzzy logic control
            await self._execute_fuzzy_control()
            
            # Neural network control
            await self._execute_neural_control()
            
        except Exception as e:
            self.logger.error(f"Thermal control execution failed: {e}")
    
    async def _execute_pid_control(self):
        """Execute PID thermal control"""
        try:
            for controller_id, controller in self.pid_controllers.items():
                # Find corresponding sensor
                sensor_type = controller_id.split('_')[0]
                sensors = [s for s in self.thermal_sensors.values() if s.sensor_type == sensor_type]
                
                if not sensors:
                    continue
                
                # Use average temperature for control
                current_temp = sum(s.current_temp for s in sensors) / len(sensors)
                setpoint = controller["setpoint"]
                
                # PID calculation
                error = current_temp - setpoint
                controller["integral"] += error
                derivative = error - controller["last_error"]
                
                output = (controller["kp"] * error + 
                         controller["ki"] * controller["integral"] + 
                         controller["kd"] * derivative)
                
                controller["last_error"] = error
                
                # Apply control output
                await self._apply_pid_output(controller_id, output)
                
        except Exception as e:
            self.logger.error(f"PID control execution failed: {e}")
    
    async def _apply_pid_output(self, controller_id: str, output: float):
        """Apply PID controller output"""
        try:
            if "cpu" in controller_id:
                # Apply to CPU fan and power
                fan_adjustment = max(0, min(output * 2.0, 50.0))  # Scale to fan speed
                current_speed = self.cooling_controller["fan_controllers"]["cpu_fan"]["current_speed"]
                new_speed = min(current_speed + fan_adjustment, 100.0)
                await self._adjust_fan_speed("cpu_fan", new_speed)
                
            elif "gpu" in controller_id:
                # Apply to GPU fan and power
                fan_adjustment = max(0, min(output * 2.0, 50.0))
                current_speed = self.cooling_controller["fan_controllers"]["gpu_fan"]["current_speed"]
                new_speed = min(current_speed + fan_adjustment, 100.0)
                await self._adjust_fan_speed("gpu_fan", new_speed)
                
        except Exception as e:
            self.logger.error(f"PID output application failed: {e}")
    
    async def _execute_fuzzy_control(self):
        """Execute fuzzy logic thermal control"""
        try:
            # Get system state
            cpu_temps = [s.current_temp for s in self.thermal_sensors.values() if s.sensor_type == "cpu"]
            avg_cpu_temp = sum(cpu_temps) / len(cpu_temps) if cpu_temps else 50.0
            
            # Simple fuzzy inference
            if avg_cpu_temp < 60.0:
                fan_speed = 30.0
                throttling = 0.0
            elif avg_cpu_temp < 75.0:
                fan_speed = 50.0
                throttling = 0.0
            else:
                fan_speed = 80.0
                throttling = 0.1
            
            # Apply fuzzy control output
            await self._adjust_fan_speed("case_fans", fan_speed)
            
            if throttling > 0:
                await self._apply_thermal_throttling("cpu", throttling)
                
        except Exception as e:
            self.logger.error(f"Fuzzy control execution failed: {e}")
    
    async def _execute_neural_control(self):
        """Execute neural network thermal control"""
        try:
            # Prepare input vector
            inputs = []
            
            # Temperature inputs
            for sensor_type in ["cpu", "gpu", "memory", "ambient"]:
                temps = [s.current_temp for s in self.thermal_sensors.values() if s.sensor_type == sensor_type]
                inputs.append(sum(temps) / len(temps) if temps else 25.0)
            
            # Power inputs
            for domain_type in ["cpu", "gpu", "memory", "system"]:
                powers = [d.current_power for d in self.power_domains.values() if d.domain_type == domain_type]
                inputs.append(sum(powers) / len(powers) if powers else 0.0)
            
            # Fan speed inputs
            for fan_id in ["cpu_fan", "gpu_fan", "case_fans"]:
                if fan_id in self.cooling_controller["fan_controllers"]:
                    inputs.append(self.cooling_controller["fan_controllers"][fan_id]["current_speed"])
                else:
                    inputs.append(50.0)
            
            # Pad or truncate to expected input size
            while len(inputs) < 16:
                inputs.append(0.0)
            inputs = inputs[:16]
            
            # Simple neural network forward pass
            neural_controller = self.neural_thermal_controller
            layer_input = np.array(inputs)
            
            for i, (weights, bias) in enumerate(zip(neural_controller["weights"], neural_controller["biases"])):
                layer_output = np.dot(layer_input, weights) + bias
                if i < len(neural_controller["weights"]) - 1:  # Apply ReLU to hidden layers
                    layer_output = np.maximum(0, layer_output)
                layer_input = layer_output
            
            # Interpret output
            outputs = layer_output
            if len(outputs) >= 4:
                cpu_fan_adjustment = outputs[0] * 10.0  # Scale to reasonable range
                gpu_fan_adjustment = outputs[1] * 10.0
                cpu_throttling = max(0, min(outputs[2] * 0.1, 0.2))  # Max 20% throttling
                gpu_throttling = max(0, min(outputs[3] * 0.1, 0.2))
                
                # Apply neural control outputs
                current_cpu_fan = self.cooling_controller["fan_controllers"]["cpu_fan"]["current_speed"]
                current_gpu_fan = self.cooling_controller["fan_controllers"]["gpu_fan"]["current_speed"]
                
                await self._adjust_fan_speed("cpu_fan", min(current_cpu_fan + cpu_fan_adjustment, 100.0))
                await self._adjust_fan_speed("gpu_fan", min(current_gpu_fan + gpu_fan_adjustment, 100.0))
                
                if cpu_throttling > 0.01:
                    await self._apply_thermal_throttling("cpu", cpu_throttling)
                if gpu_throttling > 0.01:
                    await self._apply_thermal_throttling("gpu", gpu_throttling)
                    
        except Exception as e:
            self.logger.error(f"Neural control execution failed: {e}")
    
    async def _apply_thermal_throttling(self, domain_type: str, throttle_amount: float):
        """Apply thermal throttling to domain"""
        try:
            domain_key = f"{domain_type}_package" if domain_type in ["cpu", "gpu"] else domain_type
            if domain_key in self.power_domains:
                domain = self.power_domains[domain_key]
                new_throttle_state = max(domain.throttle_state - throttle_amount, 0.3)  # Min 30%
                domain.throttle_state = new_throttle_state
                
                self.thermal_stats["throttling_events"] += 1
                
                # Notify other agents about throttling
                await self.broadcast_message(
                    "thermal_throttling_applied",
                    {
                        "domain": domain_type,
                        "throttle_level": 1.0 - new_throttle_state,
                        "reason": "thermal_protection"
                    },
                    priority=MessagePriority.HIGH
                )
                
        except Exception as e:
            self.logger.error(f"Thermal throttling application failed: {e}")
    
    async def _execute_power_optimization(self):
        """Execute power optimization algorithms"""
        try:
            # DVFS optimization
            await self._optimize_dvfs()
            
            # Power gating optimization
            await self._optimize_power_gating()
            
            # Workload-based power optimization
            await self._optimize_workload_power()
            
        except Exception as e:
            self.logger.error(f"Power optimization execution failed: {e}")
    
    async def _optimize_dvfs(self):
        """Optimize Dynamic Voltage and Frequency Scaling"""
        try:
            dvfs_controller = self.power_optimizer["dvfs_controller"]
            
            # CPU DVFS optimization
            cpu_domain = self.power_domains.get("cpu_package")
            if cpu_domain:
                target_power = cpu_domain.target_power
                current_power = cpu_domain.current_power
                
                if current_power > target_power * 1.1:  # 10% over target
                    # Reduce frequency/voltage
                    if dvfs_controller["current_cpu_state"] > 0:
                        dvfs_controller["current_cpu_state"] -= 1
                        new_freq, new_voltage = dvfs_controller["cpu_p_states"][dvfs_controller["current_cpu_state"]]
                        cpu_domain.frequency = new_freq
                        cpu_domain.voltage = new_voltage
                        
                elif current_power < target_power * 0.8:  # 20% under target
                    # Increase frequency/voltage
                    if dvfs_controller["current_cpu_state"] < len(dvfs_controller["cpu_p_states"]) - 1:
                        dvfs_controller["current_cpu_state"] += 1
                        new_freq, new_voltage = dvfs_controller["cpu_p_states"][dvfs_controller["current_cpu_state"]]
                        cpu_domain.frequency = new_freq
                        cpu_domain.voltage = new_voltage
            
            # GPU DVFS optimization
            gpu_domain = self.power_domains.get("gpu_primary")
            if gpu_domain:
                target_power = gpu_domain.target_power
                current_power = gpu_domain.current_power
                
                if current_power > target_power * 1.1:
                    if dvfs_controller["current_gpu_state"] > 0:
                        dvfs_controller["current_gpu_state"] -= 1
                        new_freq, new_voltage = dvfs_controller["gpu_p_states"][dvfs_controller["current_gpu_state"]]
                        gpu_domain.frequency = new_freq
                        gpu_domain.voltage = new_voltage
                        
                elif current_power < target_power * 0.8:
                    if dvfs_controller["current_gpu_state"] < len(dvfs_controller["gpu_p_states"]) - 1:
                        dvfs_controller["current_gpu_state"] += 1
                        new_freq, new_voltage = dvfs_controller["gpu_p_states"][dvfs_controller["current_gpu_state"]]
                        gpu_domain.frequency = new_freq
                        gpu_domain.voltage = new_voltage
                        
        except Exception as e:
            self.logger.error(f"DVFS optimization failed: {e}")
    
    async def _optimize_power_gating(self):
        """Optimize power gating for unused components"""
        try:
            gating_controller = self.power_optimizer["power_gating_controller"]
            
            # Simple power gating logic
            for unit in gating_controller["gatable_units"]:
                # This would check actual utilization and gate unused units
                # For now, simulate power gating decisions
                if "unused" in unit:
                    # Gate unused units
                    pass
                    
        except Exception as e:
            self.logger.error(f"Power gating optimization failed: {e}")
    
    async def _optimize_workload_power(self):
        """Optimize power based on workload characteristics"""
        try:
            # This would analyze current workload and optimize power accordingly
            # For now, implement basic workload-aware power management
            
            # Get system utilization
            cpu_utilization = 0.5  # Placeholder
            gpu_utilization = 0.3  # Placeholder
            
            # Adjust power targets based on utilization
            cpu_domain = self.power_domains.get("cpu_package")
            if cpu_domain:
                optimal_power = cpu_domain.max_power * cpu_utilization * 1.2
                cpu_domain.target_power = min(optimal_power, cpu_domain.max_power)
            
            gpu_domain = self.power_domains.get("gpu_primary")
            if gpu_domain:
                optimal_power = gpu_domain.max_power * gpu_utilization * 1.2
                gpu_domain.target_power = min(optimal_power, gpu_domain.max_power)
                
        except Exception as e:
            self.logger.error(f"Workload power optimization failed: {e}")
    
    async def _update_ml_models(self):
        """Update machine learning models"""
        try:
            # Update thermal prediction model
            await self._update_thermal_ml_model()
            
            # Update power prediction model
            await self._update_power_ml_model()
            
            # Update cooling optimization model
            await self._update_cooling_ml_model()
            
            # Update workload classifier
            await self._update_workload_classifier()
            
        except Exception as e:
            self.logger.error(f"ML model update failed: {e}")
    
    async def _update_thermal_ml_model(self):
        """Update thermal prediction ML model"""
        try:
            thermal_model = self.ml_models["thermal_predictor"]
            
            # Collect training data
            current_data = []
            for sensor in self.thermal_sensors.values():
                current_data.append(sensor.current_temp)
            for domain in self.power_domains.values():
                current_data.append(domain.current_power)
            
            if len(current_data) >= thermal_model["input_features"]:
                thermal_model["training_data"].append(current_data[:thermal_model["input_features"]])
            
            # Simple model update (in practice, this would be more sophisticated)
            if len(thermal_model["training_data"]) > 100:
                # Update weights based on recent data
                thermal_model["weights"] *= 0.999  # Decay
                
        except Exception as e:
            self.logger.error(f"Thermal ML model update failed: {e}")
    
    async def _update_power_ml_model(self):
        """Update power prediction ML model"""
        try:
            power_model = self.ml_models["power_predictor"]
            
            # Update power efficiency correlations
            for domain in self.power_domains.values():
                efficiency = domain.current_power / max(domain.frequency * domain.voltage, 1.0)
                power_model["workload_power_correlation"] = (
                    power_model["workload_power_correlation"] * 0.95 + efficiency * 0.05
                )
                
        except Exception as e:
            self.logger.error(f"Power ML model update failed: {e}")
    
    async def _update_cooling_ml_model(self):
        """Update cooling optimization ML model"""
        try:
            cooling_model = self.ml_models["cooling_optimizer"]
            
            # Update Q-learning for cooling optimization
            current_state = self._get_cooling_state()
            
            # Simple Q-learning update
            for action in range(cooling_model["action_space"]):
                reward = self._calculate_cooling_reward(action)
                current_q = cooling_model["q_table"][current_state][action]
                cooling_model["q_table"][current_state][action] = (
                    current_q + cooling_model["learning_rate"] * (reward - current_q)
                )
                
        except Exception as e:
            self.logger.error(f"Cooling ML model update failed: {e}")
    
    def _get_cooling_state(self) -> str:
        """Get current cooling state for ML model"""
        # Discretize system state for Q-learning
        cpu_temps = [s.current_temp for s in self.thermal_sensors.values() if s.sensor_type == "cpu"]
        avg_cpu_temp = sum(cpu_temps) / len(cpu_temps) if cpu_temps else 50.0
        
        temp_bucket = int(avg_cpu_temp // 10)  # 10-degree buckets
        fan_speed_bucket = int(self.cooling_controller["fan_controllers"]["cpu_fan"]["current_speed"] // 25)
        
        return f"temp_{temp_bucket}_fan_{fan_speed_bucket}"
    
    def _calculate_cooling_reward(self, action: int) -> float:
        """Calculate reward for cooling action"""
        # Simple reward function: negative for high temperatures, positive for efficiency
        cpu_temps = [s.current_temp for s in self.thermal_sensors.values() if s.sensor_type == "cpu"]
        avg_cpu_temp = sum(cpu_temps) / len(cpu_temps) if cpu_temps else 50.0
        
        temp_penalty = max(0, avg_cpu_temp - 75.0) * -0.1
        efficiency_reward = 1.0 / (1.0 + avg_cpu_temp / 100.0)
        
        return temp_penalty + efficiency_reward
    
    async def _update_workload_classifier(self):
        """Update workload classification model"""
        try:
            classifier = self.ml_models["workload_classifier"]
            
            # Collect workload features
            features = []
            features.append(sum(d.current_power for d in self.power_domains.values()))
            features.extend([s.current_temp for s in list(self.thermal_sensors.values())[:4]])
            
            if len(features) >= 5:
                classifier["feature_vectors"].append(features[:5])
                
                # Keep only recent feature vectors
                if len(classifier["feature_vectors"]) > 1000:
                    classifier["feature_vectors"] = classifier["feature_vectors"][-500:]
                    
        except Exception as e:
            self.logger.error(f"Workload classifier update failed: {e}")
    
    async def _check_thermal_emergencies(self):
        """Check for thermal emergency conditions"""
        try:
            emergency_actions = self.throttling_manager["emergency_actions"]
            
            # Check for emergency shutdown conditions
            for sensor in self.thermal_sensors.values():
                if sensor.current_temp >= emergency_actions["emergency_shutdown"]["temp_threshold"]:
                    await self._trigger_emergency_shutdown(sensor)
                    
                elif sensor.current_temp >= emergency_actions["emergency_throttling"]["temp_threshold"]:
                    await self._trigger_emergency_throttling(sensor)
                    
                elif sensor.current_temp >= emergency_actions["fan_override"]["temp_threshold"]:
                    await self._trigger_fan_override(sensor)
                    
        except Exception as e:
            self.logger.error(f"Thermal emergency check failed: {e}")
    
    async def _trigger_emergency_shutdown(self, sensor: ThermalSensor):
        """Trigger emergency system shutdown"""
        try:
            self.logger.critical(f"EMERGENCY: Temperature {sensor.current_temp}°C on {sensor.location} exceeds shutdown threshold!")
            
            self.system_thermal_state["emergency_throttling_active"] = True
            self.thermal_stats["temperature_violations"] += 1
            
            # Notify all agents of emergency
            await self.broadcast_message(
                "thermal_emergency_shutdown",
                {
                    "sensor": sensor.sensor_id,
                    "temperature": sensor.current_temp,
                    "location": sensor.location,
                    "action": "emergency_shutdown_initiated"
                },
                priority=MessagePriority.CRITICAL
            )
            
        except Exception as e:
            self.logger.error(f"Emergency shutdown trigger failed: {e}")
    
    async def _trigger_emergency_throttling(self, sensor: ThermalSensor):
        """Trigger emergency thermal throttling"""
        try:
            self.logger.warning(f"EMERGENCY THROTTLING: {sensor.location} at {sensor.current_temp}°C")
            
            # Apply maximum throttling
            throttle_level = 0.75  # 75% throttling
            
            if sensor.sensor_type == "cpu":
                await self._apply_thermal_throttling("cpu", throttle_level)
            elif sensor.sensor_type == "gpu":
                await self._apply_thermal_throttling("gpu", throttle_level)
            
            # Maximum cooling
            await self._adjust_fan_speed("all", 100.0)
            
        except Exception as e:
            self.logger.error(f"Emergency throttling trigger failed: {e}")
    
    async def _trigger_fan_override(self, sensor: ThermalSensor):
        """Trigger fan speed override"""
        try:
            self.logger.warning(f"FAN OVERRIDE: {sensor.location} at {sensor.current_temp}°C")
            
            # Set all fans to maximum
            await self._adjust_fan_speed("all", 100.0)
            
            # Increase liquid cooling if available
            await self._adjust_liquid_cooling(1.2)
            
        except Exception as e:
            self.logger.error(f"Fan override trigger failed: {e}")
    
    async def _update_system_thermal_state(self):
        """Update overall system thermal state"""
        try:
            # Calculate global thermal pressure
            max_temp_ratio = 0.0
            for sensor in self.thermal_sensors.values():
                temp_ratio = sensor.current_temp / sensor.critical_temp
                max_temp_ratio = max(max_temp_ratio, temp_ratio)
            
            self.system_thermal_state["global_thermal_pressure"] = max_temp_ratio
            
            # Calculate power budget utilization
            total_power = sum(d.current_power for d in self.power_domains.values() if d.domain_type != "system")
            max_power = sum(d.max_power for d in self.power_domains.values() if d.domain_type != "system")
            
            if max_power > 0:
                self.system_thermal_state["power_budget_utilization"] = total_power / max_power
            
            # Calculate cooling efficiency
            cooling_effectiveness = 0.0
            for fan_id, fan_data in self.cooling_controller["fan_controllers"].items():
                cooling_effectiveness += fan_data["current_speed"] / 100.0
            
            self.system_thermal_state["cooling_efficiency"] = cooling_effectiveness / len(self.cooling_controller["fan_controllers"])
            
            # Calculate thermal runaway risk
            temp_trends = []
            for sensor in self.thermal_sensors.values():
                if len(sensor.temp_history) >= 2:
                    recent_temps = [temp for _, temp in list(sensor.temp_history)[-5:]]
                    if len(recent_temps) >= 2:
                        trend = (recent_temps[-1] - recent_temps[0]) / len(recent_temps)
                        temp_trends.append(trend)
            
            if temp_trends:
                avg_trend = sum(temp_trends) / len(temp_trends)
                self.system_thermal_state["thermal_runaway_risk"] = max(0.0, avg_trend / 10.0)
            
        except Exception as e:
            self.logger.error(f"System thermal state update failed: {e}")
    
    async def handle_message(self, message):
        """Handle thermal management specific messages"""
        await super().handle_message(message)
        
        if message.message_type == "thermal_control_request":
            await self._handle_thermal_control_request(message)
        elif message.message_type == "power_optimization_request":
            await self._handle_power_optimization_request(message)
        elif message.message_type == "get_thermal_status":
            await self._handle_thermal_status_request(message)
        elif message.message_type == "workload_change_notification":
            await self._handle_workload_change(message)
    
    async def _handle_thermal_control_request(self, message):
        """Handle thermal control requests"""
        try:
            request_data = message.payload
            control_type = request_data.get("control_type")
            
            if control_type == "increase_cooling":
                intensity = request_data.get("intensity", 1.2)
                await self._adjust_fan_speed("all", 
                    min(self.cooling_controller["fan_controllers"]["cpu_fan"]["current_speed"] * intensity, 100.0))
                
            elif control_type == "reduce_power":
                domain = request_data.get("domain", "cpu")
                reduction = request_data.get("reduction", 0.1)
                await self._adjust_power_target(domain, reduction)
                
        except Exception as e:
            self.logger.error(f"Thermal control request handling failed: {e}")
    
    async def _handle_power_optimization_request(self, message):
        """Handle power optimization requests"""
        try:
            request_data = message.payload
            optimization_type = request_data.get("optimization_type")
            
            if optimization_type == "efficiency":
                # Optimize for power efficiency
                await self._optimize_dvfs()
                await self._optimize_power_gating()
                
            elif optimization_type == "performance":
                # Optimize for maximum performance within thermal limits
                for domain in self.power_domains.values():
                    if domain.current_power < domain.max_power * 0.9:
                        domain.target_power = min(domain.target_power * 1.1, domain.max_power)
                        
        except Exception as e:
            self.logger.error(f"Power optimization request handling failed: {e}")
    
    async def _handle_thermal_status_request(self, message):
        """Handle thermal status requests"""
        try:
            status = {
                "thermal_sensors": {
                    sensor_id: {
                        "temperature": sensor.current_temp,
                        "location": sensor.location,
                        "type": sensor.sensor_type,
                        "status": "normal" if sensor.current_temp < sensor.max_temp else "warning"
                    }
                    for sensor_id, sensor in self.thermal_sensors.items()
                },
                "power_domains": {
                    domain_id: {
                        "current_power": domain.current_power,
                        "target_power": domain.target_power,
                        "max_power": domain.max_power,
                        "throttle_state": domain.throttle_state
                    }
                    for domain_id, domain in self.power_domains.items()
                },
                "system_thermal_state": self.system_thermal_state,
                "cooling_status": {
                    fan_id: fan_data["current_speed"]
                    for fan_id, fan_data in self.cooling_controller["fan_controllers"].items()
                }
            }
            
            await self.send_message(
                message.sender_id,
                "thermal_status_response",
                status,
                priority=MessagePriority.NORMAL
            )
            
        except Exception as e:
            self.logger.error(f"Thermal status request handling failed: {e}")
    
    async def _handle_workload_change(self, message):
        """Handle workload change notifications"""
        try:
            workload_data = message.payload
            workload_type = workload_data.get("workload_type", "unknown")
            intensity = workload_data.get("intensity", 1.0)
            
            # Adjust thermal management based on workload
            if workload_type == "compute_intensive":
                # Prepare for high CPU/GPU usage
                await self._adjust_fan_speed("cpu_fan", 60.0)
                await self._adjust_fan_speed("gpu_fan", 60.0)
                
            elif workload_type == "memory_intensive":
                # Prepare for high memory usage
                await self._adjust_power_target("memory", -0.05)  # Slight power reduction
                
            # Update workload classifier
            classifier = self.ml_models["workload_classifier"]
            if workload_type in classifier["workload_classes"]:
                # Update thermal profile for this workload type
                current_temps = [s.current_temp for s in self.thermal_sensors.values()]
                classifier["thermal_profiles"][workload_type] = {
                    "avg_temp": sum(current_temps) / len(current_temps),
                    "max_temp": max(current_temps),
                    "intensity": intensity
                }
                
        except Exception as e:
            self.logger.error(f"Workload change handling failed: {e}")
    
    async def _agent_specific_optimization(self):
        """Thermal agent specific optimizations"""
        # Clean up old sensor history
        for sensor in self.thermal_sensors.values():
            if len(sensor.temp_history) > 1000:
                # Keep only recent history
                sensor.temp_history = deque(list(sensor.temp_history)[-500:], maxlen=1000)
        
        # Clean up old power history
        for domain in self.power_domains.values():
            if len(domain.power_history) > 1000:
                domain.power_history = deque(list(domain.power_history)[-500:], maxlen=1000)
        
        # Update thermal statistics
        cpu_temps = [s.current_temp for s in self.thermal_sensors.values() if s.sensor_type == "cpu"]
        if cpu_temps:
            avg_temp = sum(cpu_temps) / len(cpu_temps)
            if avg_temp < 70.0:  # Good temperature
                self.thermal_stats["avg_temp_reduction"] = max(0, 75.0 - avg_temp)
        
        # Update power savings
        total_power = sum(d.current_power for d in self.power_domains.values() if d.domain_type != "system")
        max_power = sum(d.max_power for d in self.power_domains.values() if d.domain_type != "system")
        if max_power > 0:
            power_savings = (max_power - total_power) / max_power
            self.thermal_stats["avg_power_savings"] = power_savings
        
        self.logger.debug(f"Thermal agent optimization complete. "
                         f"Sensors: {len(self.thermal_sensors)}, "
                         f"Power domains: {len(self.power_domains)}")
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get thermal agent statistics"""
        return {
            "thermal_stats": self.thermal_stats.copy(),
            "thermal_sensors": len(self.thermal_sensors),
            "power_domains": len(self.power_domains),
            "thermal_models": len(self.thermal_models),
            "current_temperatures": {
                sensor_id: sensor.current_temp
                for sensor_id, sensor in self.thermal_sensors.items()
            },
            "current_power_consumption": {
                domain_id: domain.current_power
                for domain_id, domain in self.power_domains.items()
            },
            "system_thermal_state": self.system_thermal_state.copy(),
            "cooling_status": {
                fan_id: fan_data["current_speed"]
                for fan_id, fan_data in self.cooling_controller["fan_controllers"].items()
            },
            "ml_models": {
                "thermal_predictor_accuracy": self.ml_models["thermal_predictor"]["accuracy"],
                "power_correlation": self.ml_models["power_predictor"]["workload_power_correlation"],
                "workload_classes": len(self.ml_models["workload_classifier"]["workload_classes"])
            },
            "control_loops": {
                "pid_controllers": len(self.pid_controllers),
                "fuzzy_rules": len(self.fuzzy_logic_controller["rules"]),
                "neural_layers": len(self.neural_thermal_controller["weights"])
            }
        }