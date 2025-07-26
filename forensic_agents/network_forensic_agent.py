#!/usr/bin/env python3
"""
Network Forensic Agent
Advanced network traffic analysis and communication pattern detection
"""

import asyncio
import re
import ipaddress
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Set, Tuple
import uuid

from forensic_framework import ForensicAgent, Evidence, Finding

class NetworkForensicAgent(ForensicAgent):
    """Advanced network forensic analysis agent"""
    
    def __init__(self):
        super().__init__("net_forensic", "Network Forensic Agent")
        self.suspicious_ips: Set[str] = set()
        self.known_malicious_domains: Set[str] = set()
        self.communication_patterns: Dict[str, List[Dict]] = {}
        self.port_scan_threshold = 10
        self.data_exfiltration_threshold = 1000000  # 1MB
        
    async def initialize(self) -> bool:
        """Initialize the network forensic agent"""
        try:
            # Load threat intelligence feeds
            await self.load_threat_intelligence()
            
            # Initialize pattern detection algorithms
            self.init_pattern_detectors()
            
            self.logger.info("Network Forensic Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Network Forensic Agent: {e}")
            return False
    
    async def load_threat_intelligence(self):
        """Load threat intelligence data"""
        # Simulated threat intelligence - in real implementation, 
        # this would fetch from external threat feeds
        self.suspicious_ips.update([
            "192.168.1.100",  # Internal suspicious host
            "10.0.0.50",      # Another internal host
            "203.0.113.1",    # External malicious IP
            "198.51.100.1"    # Another external malicious IP
        ])
        
        self.known_malicious_domains.update([
            "malware-c2.evil.com",
            "phishing-site.bad.org",
            "data-exfil.attacker.net",
            "botnet-command.malicious.io"
        ])
        
        self.logger.info(f"Loaded {len(self.suspicious_ips)} suspicious IPs and "
                        f"{len(self.known_malicious_domains)} malicious domains")
    
    def init_pattern_detectors(self):
        """Initialize pattern detection algorithms"""
        self.patterns = {
            'port_scan': self.detect_port_scan,
            'data_exfiltration': self.detect_data_exfiltration,
            'lateral_movement': self.detect_lateral_movement,
            'dns_tunneling': self.detect_dns_tunneling,
            'beaconing': self.detect_beaconing,
            'anomalous_traffic': self.detect_anomalous_traffic
        }
    
    async def analyze(self, evidence: Evidence) -> List[Finding]:
        """Analyze network evidence"""
        findings = []
        
        if evidence.type not in ['network_traffic', 'network_connection', 'dns_query', 'packet_capture']:
            return findings
        
        try:
            # Run all pattern detection algorithms
            for pattern_name, detector in self.patterns.items():
                pattern_findings = await detector(evidence)
                findings.extend(pattern_findings)
            
            # Analyze communication patterns
            comm_findings = await self.analyze_communication_patterns(evidence)
            findings.extend(comm_findings)
            
            # Check against threat intelligence
            ti_findings = await self.check_threat_intelligence(evidence)
            findings.extend(ti_findings)
            
        except Exception as e:
            self.logger.error(f"Error analyzing network evidence: {e}")
        
        return findings
    
    async def detect_port_scan(self, evidence: Evidence) -> List[Finding]:
        """Detect port scanning activities"""
        findings = []
        
        if evidence.type != 'network_connection':
            return findings
        
        data = evidence.data
        src_ip = data.get('src_ip')
        dst_ip = data.get('dst_ip')
        dst_port = data.get('dst_port')
        
        if not all([src_ip, dst_ip, dst_port]):
            return findings
        
        # Track port access patterns
        key = f"{src_ip}->{dst_ip}"
        if key not in self.communication_patterns:
            self.communication_patterns[key] = []
        
        self.communication_patterns[key].append({
            'timestamp': evidence.timestamp.isoformat(),
            'port': dst_port,
            'protocol': data.get('protocol', 'unknown')
        })
        
        # Check if threshold exceeded
        unique_ports = len(set(conn['port'] for conn in self.communication_patterns[key]))
        
        if unique_ports >= self.port_scan_threshold:
            finding = Finding(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                agent_id=self.agent_id,
                severity="HIGH",
                category="Network Reconnaissance",
                title="Port Scanning Activity Detected",
                description=f"Host {src_ip} has accessed {unique_ports} different ports on {dst_ip}, "
                           f"indicating potential port scanning activity.",
                evidence_ids=[evidence.id],
                confidence=0.85,
                recommendations=[
                    f"Investigate host {src_ip} for malicious activity",
                    f"Review firewall logs for {src_ip}",
                    "Consider blocking or isolating the scanning host",
                    "Check for successful connections after scan attempts"
                ]
            )
            findings.append(finding)
        
        return findings
    
    async def detect_data_exfiltration(self, evidence: Evidence) -> List[Finding]:
        """Detect potential data exfiltration"""
        findings = []
        
        if evidence.type != 'network_traffic':
            return findings
        
        data = evidence.data
        src_ip = data.get('src_ip')
        dst_ip = data.get('dst_ip')
        bytes_transferred = data.get('bytes_out', 0)
        
        if bytes_transferred > self.data_exfiltration_threshold:
            # Check if destination is external
            try:
                dst_addr = ipaddress.ip_address(dst_ip)
                if not dst_addr.is_private:
                    finding = Finding(
                        id=str(uuid.uuid4()),
                        timestamp=datetime.now(timezone.utc),
                        agent_id=self.agent_id,
                        severity="CRITICAL",
                        category="Data Exfiltration",
                        title="Large Data Transfer to External Host",
                        description=f"Host {src_ip} transferred {bytes_transferred} bytes "
                                   f"to external host {dst_ip}, indicating potential data exfiltration.",
                        evidence_ids=[evidence.id],
                        confidence=0.75,
                        recommendations=[
                            f"Immediately investigate host {src_ip}",
                            f"Analyze content of data transferred to {dst_ip}",
                            "Check for similar patterns in historical data",
                            "Consider isolating the source host",
                            "Review DLP policies and controls"
                        ]
                    )
                    findings.append(finding)
            except ValueError:
                pass  # Invalid IP address
        
        return findings
    
    async def detect_lateral_movement(self, evidence: Evidence) -> List[Finding]:
        """Detect lateral movement patterns"""
        findings = []
        
        if evidence.type != 'network_connection':
            return findings
        
        data = evidence.data
        src_ip = data.get('src_ip')
        dst_ip = data.get('dst_ip')
        dst_port = data.get('dst_port')
        protocol = data.get('protocol', '')
        
        # Check for administrative protocol usage
        admin_ports = {22, 23, 135, 139, 445, 3389, 5985, 5986}  # SSH, Telnet, RPC, SMB, RDP, WinRM
        
        if dst_port in admin_ports:
            # Track lateral movement attempts
            key = f"lateral_{src_ip}"
            if key not in self.communication_patterns:
                self.communication_patterns[key] = []
            
            self.communication_patterns[key].append({
                'timestamp': evidence.timestamp.isoformat(),
                'dst_ip': dst_ip,
                'port': dst_port,
                'protocol': protocol
            })
            
            # Check for multiple targets
            unique_targets = len(set(conn['dst_ip'] for conn in self.communication_patterns[key]))
            
            if unique_targets >= 3:  # Threshold for lateral movement
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity="HIGH",
                    category="Lateral Movement",
                    title="Potential Lateral Movement Detected",
                    description=f"Host {src_ip} has initiated administrative connections "
                               f"to {unique_targets} different internal hosts, indicating "
                               f"potential lateral movement.",
                    evidence_ids=[evidence.id],
                    confidence=0.80,
                    recommendations=[
                        f"Investigate host {src_ip} for compromise",
                        "Review authentication logs for unusual activity",
                        "Check privilege escalation attempts",
                        "Analyze process execution on target hosts",
                        "Consider network segmentation improvements"
                    ]
                )
                findings.append(finding)
        
        return findings
    
    async def detect_dns_tunneling(self, evidence: Evidence) -> List[Finding]:
        """Detect DNS tunneling activities"""
        findings = []
        
        if evidence.type != 'dns_query':
            return findings
        
        data = evidence.data
        query = data.get('query', '')
        query_type = data.get('type', '')
        response_size = data.get('response_size', 0)
        
        # Check for suspicious DNS patterns
        suspicious_patterns = [
            len(query) > 100,  # Unusually long queries
            query_type in ['TXT', 'NULL'],  # Unusual record types
            response_size > 1000,  # Large responses
            re.search(r'[a-f0-9]{32,}', query),  # Hex-encoded data
            query.count('.') > 10  # Too many subdomains
        ]
        
        if any(suspicious_patterns):
            finding = Finding(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                agent_id=self.agent_id,
                severity="MEDIUM",
                category="DNS Tunneling",
                title="Suspicious DNS Query Pattern",
                description=f"DNS query '{query[:50]}...' shows patterns consistent "
                           f"with DNS tunneling or data exfiltration.",
                evidence_ids=[evidence.id],
                confidence=0.70,
                recommendations=[
                    "Analyze DNS query patterns for this domain",
                    "Check for data encoding in DNS queries",
                    "Review DNS server logs for similar patterns",
                    "Consider blocking suspicious domains",
                    "Implement DNS monitoring and filtering"
                ]
            )
            findings.append(finding)
        
        return findings
    
    async def detect_beaconing(self, evidence: Evidence) -> List[Finding]:
        """Detect C2 beaconing patterns"""
        findings = []
        
        if evidence.type != 'network_connection':
            return findings
        
        data = evidence.data
        src_ip = data.get('src_ip')
        dst_ip = data.get('dst_ip')
        
        # Track connection timing patterns
        key = f"beacon_{src_ip}_{dst_ip}"
        if key not in self.communication_patterns:
            self.communication_patterns[key] = []
        
        self.communication_patterns[key].append({
            'timestamp': evidence.timestamp.timestamp(),
            'bytes_in': data.get('bytes_in', 0),
            'bytes_out': data.get('bytes_out', 0)
        })
        
        # Analyze for regular intervals (simple heuristic)
        if len(self.communication_patterns[key]) >= 5:
            timestamps = [conn['timestamp'] for conn in self.communication_patterns[key][-5:]]
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            
            # Check for regular intervals (within 10% variance)
            if len(set(round(interval, 1) for interval in intervals)) <= 2:
                avg_interval = sum(intervals) / len(intervals)
                
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity="HIGH",
                    category="Command and Control",
                    title="Regular Beaconing Pattern Detected",
                    description=f"Host {src_ip} is communicating with {dst_ip} at regular "
                               f"intervals (avg: {avg_interval:.1f}s), indicating potential "
                               f"C2 beaconing activity.",
                    evidence_ids=[evidence.id],
                    confidence=0.75,
                    recommendations=[
                        f"Block communication between {src_ip} and {dst_ip}",
                        f"Investigate host {src_ip} for malware infection",
                        "Analyze payload content of communications",
                        "Check for similar patterns with other external hosts",
                        "Review process activity on the beaconing host"
                    ]
                )
                findings.append(finding)
        
        return findings
    
    async def detect_anomalous_traffic(self, evidence: Evidence) -> List[Finding]:
        """Detect anomalous network traffic patterns"""
        findings = []
        
        if evidence.type != 'network_traffic':
            return findings
        
        data = evidence.data
        protocol = data.get('protocol', '')
        dst_port = data.get('dst_port', 0)
        bytes_transferred = data.get('bytes_out', 0) + data.get('bytes_in', 0)
        
        # Check for unusual protocol/port combinations
        unusual_combinations = [
            (protocol == 'HTTP' and dst_port not in [80, 8080, 8000]),
            (protocol == 'HTTPS' and dst_port not in [443, 8443]),
            (protocol == 'SSH' and dst_port != 22),
            (protocol == 'FTP' and dst_port not in [20, 21])
        ]
        
        if any(unusual_combinations):
            finding = Finding(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                agent_id=self.agent_id,
                severity="MEDIUM",
                category="Anomalous Traffic",
                title="Unusual Protocol/Port Combination",
                description=f"Traffic using {protocol} protocol on port {dst_port} "
                           f"is unusual and may indicate tunneling or evasion.",
                evidence_ids=[evidence.id],
                confidence=0.60,
                recommendations=[
                    "Investigate the application using this port",
                    "Check for protocol tunneling or evasion",
                    "Review firewall rules for this port",
                    "Analyze packet contents for protocol violations"
                ]
            )
            findings.append(finding)
        
        return findings
    
    async def analyze_communication_patterns(self, evidence: Evidence) -> List[Finding]:
        """Analyze overall communication patterns"""
        findings = []
        
        # This would implement more sophisticated pattern analysis
        # such as graph analysis, clustering, and behavioral modeling
        
        return findings
    
    async def check_threat_intelligence(self, evidence: Evidence) -> List[Finding]:
        """Check evidence against threat intelligence feeds"""
        findings = []
        
        data = evidence.data
        
        # Check IPs against threat intelligence
        for ip_field in ['src_ip', 'dst_ip']:
            ip = data.get(ip_field)
            if ip and ip in self.suspicious_ips:
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity="HIGH",
                    category="Threat Intelligence",
                    title="Communication with Known Malicious IP",
                    description=f"Communication detected with known malicious IP {ip} "
                               f"flagged in threat intelligence feeds.",
                    evidence_ids=[evidence.id],
                    confidence=0.90,
                    recommendations=[
                        f"Immediately block IP {ip}",
                        "Investigate all hosts that communicated with this IP",
                        "Check for indicators of compromise",
                        "Review historical communications with this IP"
                    ]
                )
                findings.append(finding)
        
        # Check domains against threat intelligence
        domain = data.get('domain') or data.get('query')
        if domain and domain in self.known_malicious_domains:
            finding = Finding(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                agent_id=self.agent_id,
                severity="CRITICAL",
                category="Threat Intelligence",
                title="Communication with Known Malicious Domain",
                description=f"Communication detected with known malicious domain {domain} "
                           f"flagged in threat intelligence feeds.",
                evidence_ids=[evidence.id],
                confidence=0.95,
                recommendations=[
                    f"Immediately block domain {domain}",
                    "Investigate all hosts that accessed this domain",
                    "Check for malware downloads or C2 communication",
                    "Review DNS logs for similar domains"
                ]
            )
            findings.append(finding)
        
        return findings