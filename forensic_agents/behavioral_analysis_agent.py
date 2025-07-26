#!/usr/bin/env python3
"""
Behavioral Analysis Agent
ML-powered behavioral analysis for anomaly detection and threat hunting
"""

import asyncio
import json
import math
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, deque
import uuid

from forensic_framework import ForensicAgent, Evidence, Finding

class BehavioralAnalysisAgent(ForensicAgent):
    """Advanced behavioral analysis agent with ML capabilities"""
    
    def __init__(self):
        super().__init__("behavioral_analysis", "Behavioral Analysis Agent")
        self.baseline_behaviors: Dict[str, Dict[str, Any]] = {}
        self.anomaly_thresholds: Dict[str, float] = {}
        self.behavior_patterns: Dict[str, List[Dict]] = {}
        self.temporal_analysis_window = 3600  # 1 hour window
        self.ml_models: Dict[str, Any] = {}
        self.feature_extractors: Dict[str, callable] = {}
        
    async def initialize(self) -> bool:
        """Initialize the behavioral analysis agent"""
        try:
            # Initialize behavior baselines
            await self.initialize_baselines()
            
            # Load ML models and feature extractors
            await self.load_ml_models()
            
            # Initialize anomaly detection thresholds
            await self.initialize_thresholds()
            
            self.logger.info("Behavioral Analysis Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Behavioral Analysis Agent: {e}")
            return False
    
    async def initialize_baselines(self):
        """Initialize behavioral baselines"""
        self.baseline_behaviors = {
            "process_creation": {
                "normal_rate": 10.0,  # processes per minute
                "common_parents": ["explorer.exe", "cmd.exe", "powershell.exe"],
                "typical_children": ["notepad.exe", "calc.exe", "chrome.exe"]
            },
            "network_activity": {
                "normal_connections_per_hour": 50,
                "common_ports": [80, 443, 53, 22],
                "typical_protocols": ["HTTP", "HTTPS", "DNS", "SSH"]
            },
            "file_operations": {
                "normal_read_rate": 100,  # files per hour
                "normal_write_rate": 20,  # files per hour
                "common_extensions": [".txt", ".doc", ".pdf", ".jpg"]
            },
            "registry_operations": {
                "normal_modification_rate": 5,  # per hour
                "common_keys": ["HKCU\\Software", "HKLM\\Software"]
            },
            "user_behavior": {
                "typical_login_hours": list(range(8, 18)),  # 8 AM to 6 PM
                "normal_session_duration": 480,  # 8 hours in minutes
                "common_applications": ["chrome.exe", "outlook.exe", "word.exe"]
            }
        }
        
        self.logger.info("Initialized behavioral baselines")
    
    async def load_ml_models(self):
        """Load ML models for behavioral analysis"""
        # Simulated ML models - in real implementation, these would be
        # trained models loaded from files
        self.ml_models = {
            "anomaly_detector": self.simple_anomaly_detector,
            "sequence_analyzer": self.sequence_pattern_analyzer,
            "clustering_model": self.behavioral_clustering,
            "time_series_analyzer": self.time_series_anomaly_detection
        }
        
        # Feature extractors for different evidence types
        self.feature_extractors = {
            "process": self.extract_process_features,
            "network_connection": self.extract_network_features,
            "file": self.extract_file_features,
            "registry_entry": self.extract_registry_features,
            "user_activity": self.extract_user_features
        }
        
        self.logger.info(f"Loaded {len(self.ml_models)} ML models")
    
    async def initialize_thresholds(self):
        """Initialize anomaly detection thresholds"""
        self.anomaly_thresholds = {
            "process_creation_rate": 2.0,  # Standard deviations from normal
            "network_connection_rate": 2.5,
            "file_operation_rate": 2.0,
            "unusual_time_activity": 1.5,
            "sequence_anomaly_score": 0.8,
            "clustering_distance": 0.7
        }
        
        self.logger.info("Initialized anomaly detection thresholds")
    
    async def analyze(self, evidence: Evidence) -> List[Finding]:
        """Analyze evidence for behavioral anomalies"""
        findings = []
        
        try:
            # Extract features from evidence
            features = await self.extract_features(evidence)
            if not features:
                return findings
            
            # Run anomaly detection algorithms
            anomaly_findings = await self.detect_anomalies(evidence, features)
            findings.extend(anomaly_findings)
            
            # Perform temporal analysis
            temporal_findings = await self.temporal_analysis(evidence, features)
            findings.extend(temporal_findings)
            
            # Sequence pattern analysis
            sequence_findings = await self.sequence_analysis(evidence, features)
            findings.extend(sequence_findings)
            
            # Behavioral clustering analysis
            clustering_findings = await self.clustering_analysis(evidence, features)
            findings.extend(clustering_findings)
            
            # Update behavior patterns
            await self.update_behavior_patterns(evidence, features)
            
        except Exception as e:
            self.logger.error(f"Error in behavioral analysis: {e}")
        
        return findings
    
    async def extract_features(self, evidence: Evidence) -> Dict[str, Any]:
        """Extract features from evidence for ML analysis"""
        evidence_type = evidence.type
        
        if evidence_type in self.feature_extractors:
            return await self.feature_extractors[evidence_type](evidence)
        
        return {}
    
    async def extract_process_features(self, evidence: Evidence) -> Dict[str, Any]:
        """Extract features from process evidence"""
        data = evidence.data
        
        features = {
            "process_name": data.get("name", ""),
            "parent_process": data.get("parent_process", ""),
            "command_line_length": len(data.get("command_line", "")),
            "creation_time": evidence.timestamp.hour,
            "user": data.get("user", ""),
            "cpu_usage": data.get("cpu_usage", 0.0),
            "memory_usage": data.get("memory_usage", 0.0),
            "network_connections": len(data.get("network_connections", [])),
            "file_handles": len(data.get("file_handles", [])),
            "registry_accesses": len(data.get("registry_accesses", []))
        }
        
        return features
    
    async def extract_network_features(self, evidence: Evidence) -> Dict[str, Any]:
        """Extract features from network evidence"""
        data = evidence.data
        
        features = {
            "src_ip": data.get("src_ip", ""),
            "dst_ip": data.get("dst_ip", ""),
            "dst_port": data.get("dst_port", 0),
            "protocol": data.get("protocol", ""),
            "bytes_transferred": data.get("bytes_out", 0) + data.get("bytes_in", 0),
            "connection_duration": data.get("duration", 0),
            "connection_time": evidence.timestamp.hour,
            "is_external": self.is_external_ip(data.get("dst_ip", "")),
            "port_category": self.categorize_port(data.get("dst_port", 0))
        }
        
        return features
    
    async def extract_file_features(self, evidence: Evidence) -> Dict[str, Any]:
        """Extract features from file evidence"""
        data = evidence.data
        file_path = data.get("path", "")
        
        features = {
            "file_path": file_path,
            "file_extension": self.get_file_extension(file_path),
            "file_size": data.get("size", 0),
            "creation_time": evidence.timestamp.hour,
            "is_executable": file_path.lower().endswith(('.exe', '.dll', '.bat', '.ps1')),
            "is_system_directory": self.is_system_directory(file_path),
            "entropy": data.get("entropy", 0.0),
            "digital_signature": data.get("signed", False)
        }
        
        return features
    
    async def extract_registry_features(self, evidence: Evidence) -> Dict[str, Any]:
        """Extract features from registry evidence"""
        data = evidence.data
        
        features = {
            "registry_key": data.get("key", ""),
            "registry_value": data.get("value", ""),
            "operation": data.get("operation", ""),
            "modification_time": evidence.timestamp.hour,
            "is_startup_key": self.is_startup_registry_key(data.get("key", "")),
            "is_security_key": self.is_security_registry_key(data.get("key", "")),
            "value_length": len(str(data.get("value", "")))
        }
        
        return features
    
    async def extract_user_features(self, evidence: Evidence) -> Dict[str, Any]:
        """Extract features from user activity evidence"""
        data = evidence.data
        
        features = {
            "user": data.get("user", ""),
            "activity_type": data.get("activity_type", ""),
            "activity_time": evidence.timestamp.hour,
            "day_of_week": evidence.timestamp.weekday(),
            "session_duration": data.get("session_duration", 0),
            "applications_used": len(data.get("applications", [])),
            "failed_logins": data.get("failed_logins", 0),
            "privilege_escalation": data.get("privilege_escalation", False)
        }
        
        return features
    
    def is_external_ip(self, ip: str) -> bool:
        """Check if IP address is external"""
        if not ip:
            return False
        
        # Simple check for private IP ranges
        private_ranges = ["192.168.", "10.", "172.16.", "127."]
        return not any(ip.startswith(range_) for range_ in private_ranges)
    
    def categorize_port(self, port: int) -> str:
        """Categorize port number"""
        if port < 1024:
            return "system"
        elif port < 49152:
            return "registered"
        else:
            return "dynamic"
    
    def get_file_extension(self, file_path: str) -> str:
        """Get file extension"""
        return file_path.split('.')[-1].lower() if '.' in file_path else ""
    
    def is_system_directory(self, file_path: str) -> bool:
        """Check if file is in system directory"""
        system_dirs = ["\\windows\\", "\\system32\\", "\\syswow64\\", "\\program files\\"]
        return any(dir_name in file_path.lower() for dir_name in system_dirs)
    
    def is_startup_registry_key(self, key: str) -> bool:
        """Check if registry key is related to startup"""
        startup_keys = ["\\run", "\\runonce", "\\runservices"]
        return any(startup_key in key.lower() for startup_key in startup_keys)
    
    def is_security_registry_key(self, key: str) -> bool:
        """Check if registry key is security-related"""
        security_keys = ["\\policies\\", "\\security\\", "\\sam\\"]
        return any(sec_key in key.lower() for sec_key in security_keys)
    
    async def detect_anomalies(self, evidence: Evidence, features: Dict[str, Any]) -> List[Finding]:
        """Detect anomalies using various ML techniques"""
        findings = []
        
        # Statistical anomaly detection
        statistical_findings = await self.statistical_anomaly_detection(evidence, features)
        findings.extend(statistical_findings)
        
        # ML-based anomaly detection
        ml_findings = await self.ml_anomaly_detection(evidence, features)
        findings.extend(ml_findings)
        
        return findings
    
    async def statistical_anomaly_detection(self, evidence: Evidence, features: Dict[str, Any]) -> List[Finding]:
        """Perform statistical anomaly detection"""
        findings = []
        
        # Process creation rate anomaly
        if evidence.type == "process":
            process_rate = await self.calculate_process_creation_rate(evidence.timestamp)
            baseline_rate = self.baseline_behaviors["process_creation"]["normal_rate"]
            
            if process_rate > baseline_rate * 3:  # 3x normal rate
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity="MEDIUM",
                    category="Behavioral Anomaly",
                    title="Abnormal Process Creation Rate",
                    description=f"Process creation rate ({process_rate:.1f}/min) is "
                               f"{process_rate/baseline_rate:.1f}x higher than baseline",
                    evidence_ids=[evidence.id],
                    confidence=0.75,
                    recommendations=[
                        "Investigate cause of high process creation rate",
                        "Check for automated tools or scripts",
                        "Review process creation timeline",
                        "Analyze parent-child process relationships"
                    ]
                )
                findings.append(finding)
        
        # Network connection rate anomaly
        if evidence.type == "network_connection":
            conn_rate = await self.calculate_network_connection_rate(evidence.timestamp)
            baseline_rate = self.baseline_behaviors["network_activity"]["normal_connections_per_hour"]
            
            if conn_rate > baseline_rate * 2:  # 2x normal rate
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity="MEDIUM",
                    category="Behavioral Anomaly",
                    title="Abnormal Network Connection Rate",
                    description=f"Network connection rate ({conn_rate:.1f}/hour) is "
                               f"{conn_rate/baseline_rate:.1f}x higher than baseline",
                    evidence_ids=[evidence.id],
                    confidence=0.70,
                    recommendations=[
                        "Investigate network activity patterns",
                        "Check for data exfiltration or C2 communication",
                        "Review destination IPs and ports",
                        "Analyze connection timing patterns"
                    ]
                )
                findings.append(finding)
        
        return findings
    
    async def ml_anomaly_detection(self, evidence: Evidence, features: Dict[str, Any]) -> List[Finding]:
        """Perform ML-based anomaly detection"""
        findings = []
        
        # Use simple anomaly detector
        anomaly_score = await self.ml_models["anomaly_detector"](features)
        
        if anomaly_score > self.anomaly_thresholds["sequence_anomaly_score"]:
            finding = Finding(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                agent_id=self.agent_id,
                severity="MEDIUM",
                category="ML Anomaly Detection",
                title="Machine Learning Anomaly Detected",
                description=f"ML model detected anomalous behavior with score {anomaly_score:.2f}",
                evidence_ids=[evidence.id],
                confidence=anomaly_score,
                recommendations=[
                    "Investigate detected anomaly in detail",
                    "Review related evidence and context",
                    "Check for similar patterns in historical data",
                    "Consider manual analysis for confirmation"
                ]
            )
            findings.append(finding)
        
        return findings
    
    async def temporal_analysis(self, evidence: Evidence, features: Dict[str, Any]) -> List[Finding]:
        """Perform temporal analysis for time-based anomalies"""
        findings = []
        
        # Check for unusual time activity
        current_hour = evidence.timestamp.hour
        
        if evidence.type == "user_activity":
            typical_hours = self.baseline_behaviors["user_behavior"]["typical_login_hours"]
            
            if current_hour not in typical_hours:
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity="LOW",
                    category="Temporal Anomaly",
                    title="Unusual Time Activity",
                    description=f"User activity detected at {current_hour}:00, "
                               f"outside typical hours ({typical_hours[0]}:00-{typical_hours[-1]}:00)",
                    evidence_ids=[evidence.id],
                    confidence=0.60,
                    recommendations=[
                        "Verify if user activity is legitimate",
                        "Check for compromised accounts",
                        "Review authentication logs",
                        "Analyze activity patterns during unusual hours"
                    ]
                )
                findings.append(finding)
        
        return findings
    
    async def sequence_analysis(self, evidence: Evidence, features: Dict[str, Any]) -> List[Finding]:
        """Analyze sequences of events for patterns"""
        findings = []
        
        # Track event sequences by source
        source = evidence.source
        if source not in self.behavior_patterns:
            self.behavior_patterns[source] = []
        
        # Add current event to sequence
        self.behavior_patterns[source].append({
            "timestamp": evidence.timestamp,
            "type": evidence.type,
            "features": features
        })
        
        # Keep only recent events (sliding window)
        cutoff_time = evidence.timestamp - timedelta(seconds=self.temporal_analysis_window)
        self.behavior_patterns[source] = [
            event for event in self.behavior_patterns[source]
            if event["timestamp"] > cutoff_time
        ]
        
        # Analyze sequence patterns
        if len(self.behavior_patterns[source]) >= 5:
            sequence_score = await self.ml_models["sequence_analyzer"](
                self.behavior_patterns[source]
            )
            
            if sequence_score > 0.8:
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity="MEDIUM",
                    category="Sequence Anomaly",
                    title="Unusual Event Sequence Pattern",
                    description=f"Detected unusual sequence pattern from {source} "
                               f"with anomaly score {sequence_score:.2f}",
                    evidence_ids=[evidence.id],
                    confidence=sequence_score,
                    recommendations=[
                        "Analyze complete event sequence",
                        "Check for attack pattern indicators",
                        "Review timeline for related events",
                        "Consider correlation with other sources"
                    ]
                )
                findings.append(finding)
        
        return findings
    
    async def clustering_analysis(self, evidence: Evidence, features: Dict[str, Any]) -> List[Finding]:
        """Perform behavioral clustering analysis"""
        findings = []
        
        # Simple clustering based on feature similarity
        cluster_distance = await self.ml_models["clustering_model"](features)
        
        if cluster_distance > self.anomaly_thresholds["clustering_distance"]:
            finding = Finding(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                agent_id=self.agent_id,
                severity="LOW",
                category="Clustering Anomaly",
                title="Behavioral Clustering Anomaly",
                description=f"Behavior significantly different from established clusters "
                           f"(distance: {cluster_distance:.2f})",
                evidence_ids=[evidence.id],
                confidence=cluster_distance,
                recommendations=[
                    "Investigate unusual behavioral patterns",
                    "Compare with similar evidence types",
                    "Check for new attack techniques",
                    "Update behavioral baselines if legitimate"
                ]
            )
            findings.append(finding)
        
        return findings
    
    async def calculate_process_creation_rate(self, timestamp: datetime) -> float:
        """Calculate process creation rate"""
        # Simulated calculation - in real implementation, this would
        # query historical data
        return 15.0  # processes per minute
    
    async def calculate_network_connection_rate(self, timestamp: datetime) -> float:
        """Calculate network connection rate"""
        # Simulated calculation
        return 120.0  # connections per hour
    
    async def simple_anomaly_detector(self, features: Dict[str, Any]) -> float:
        """Simple anomaly detection algorithm"""
        # Calculate anomaly score based on feature deviations
        score = 0.0
        
        # Check for unusual feature values
        if features.get("command_line_length", 0) > 500:
            score += 0.3
        
        if features.get("creation_time", 12) in [0, 1, 2, 3, 4, 5, 23]:
            score += 0.2
        
        if features.get("is_external", False) and features.get("dst_port", 0) > 10000:
            score += 0.4
        
        if features.get("entropy", 0) > 7.5:
            score += 0.3
        
        return min(score, 1.0)
    
    async def sequence_pattern_analyzer(self, sequence: List[Dict[str, Any]]) -> float:
        """Analyze sequence patterns for anomalies"""
        if len(sequence) < 3:
            return 0.0
        
        # Simple pattern analysis
        score = 0.0
        
        # Check for rapid succession of events
        time_diffs = []
        for i in range(1, len(sequence)):
            diff = (sequence[i]["timestamp"] - sequence[i-1]["timestamp"]).total_seconds()
            time_diffs.append(diff)
        
        if time_diffs and statistics.mean(time_diffs) < 1.0:  # Very rapid events
            score += 0.5
        
        # Check for unusual type combinations
        types = [event["type"] for event in sequence]
        if "process" in types and "network_connection" in types and "file" in types:
            score += 0.4
        
        return min(score, 1.0)
    
    async def behavioral_clustering(self, features: Dict[str, Any]) -> float:
        """Simple behavioral clustering algorithm"""
        # Calculate distance from "normal" behavior cluster center
        
        # Define normal cluster centers for different evidence types
        normal_centers = {
            "process": {"command_line_length": 50, "creation_time": 12, "cpu_usage": 5.0},
            "network_connection": {"dst_port": 443, "bytes_transferred": 1000, "connection_time": 12},
            "file": {"file_size": 10000, "creation_time": 12, "entropy": 4.0}
        }
        
        # Calculate Euclidean distance (simplified)
        distance = 0.0
        feature_count = 0
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                # Normalize and calculate distance
                normalized_value = value / 100.0  # Simple normalization
                distance += normalized_value ** 2
                feature_count += 1
        
        if feature_count > 0:
            distance = math.sqrt(distance / feature_count)
        
        return min(distance, 1.0)
    
    async def time_series_anomaly_detection(self, features: Dict[str, Any]) -> float:
        """Time series anomaly detection"""
        # Simulated time series analysis
        return 0.3
    
    async def update_behavior_patterns(self, evidence: Evidence, features: Dict[str, Any]):
        """Update behavior patterns with new evidence"""
        # Update baseline behaviors based on observed patterns
        # This would implement adaptive learning in a real system
        pass