"""
Network Forensics Agent
======================

Specialized agent for analyzing network captures, detecting intrusions,
and reconstructing network activities
"""

import asyncio
import ipaddress
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pyshark
import dpkt
import socket
import struct

from .base_agent import BaseAgent, AgentCapabilities
from ..utils.network_utils import PacketAnalyzer, FlowReconstructor
from ..utils.threat_intel import ThreatIntelligence


class NetworkForensicsAgent(BaseAgent):
    """
    Advanced network forensics agent with capabilities:
    - Packet capture analysis (PCAP/PCAPNG)
    - Network flow reconstruction
    - Intrusion detection
    - Protocol analysis
    - Data exfiltration detection
    - C2 communication identification
    - SSL/TLS analysis
    """
    
    def __init__(self):
        super().__init__("NetworkForensicsAgent", "2.0.0")
        self.packet_analyzer = None
        self.flow_reconstructor = None
        self.threat_intel = None
        
    def _define_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            evidence_types=["network_capture", "pcap", "pcapng", "netflow"],
            analysis_types=[
                "packet_analysis",
                "flow_reconstruction",
                "intrusion_detection",
                "protocol_analysis",
                "data_exfiltration",
                "c2_detection",
                "ssl_analysis"
            ],
            required_tools=["tshark", "tcpdump", "suricata"],
            parallel_capable=True,
            gpu_accelerated=False
        )
    
    async def _setup(self):
        """Initialize network forensics tools"""
        self.packet_analyzer = PacketAnalyzer()
        self.flow_reconstructor = FlowReconstructor()
        self.threat_intel = ThreatIntelligence()
        
        # Load threat intelligence feeds
        await self.threat_intel.load_feeds()
    
    async def analyze(self, evidence_data: Dict[str, Any], 
                     case_context: Any) -> Dict[str, Any]:
        """Perform comprehensive network analysis"""
        if not await self.validate_evidence(evidence_data):
            return {"error": "Invalid evidence type for network analysis"}
        
        self.logger.info(f"Starting network analysis for {evidence_data['id']}")
        
        capture_path = Path(evidence_data['path'])
        results = {
            'agent': self.name,
            'version': self.version,
            'analysis_start': datetime.now().isoformat(),
            'evidence_id': evidence_data['id'],
            'findings': {}
        }
        
        # Run analyses in parallel
        analysis_tasks = [
            self._analyze_packets(capture_path),
            self._reconstruct_flows(capture_path),
            self._detect_intrusions(capture_path),
            self._analyze_protocols(capture_path),
            self._detect_data_exfiltration(capture_path),
            self._detect_c2_communication(capture_path),
            self._analyze_ssl_tls(capture_path)
        ]
        
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Process results
        analysis_names = [
            'packet_analysis',
            'flow_reconstruction',
            'intrusion_detection',
            'protocol_analysis',
            'data_exfiltration',
            'c2_detection',
            'ssl_analysis'
        ]
        
        for name, result in zip(analysis_names, analysis_results):
            if isinstance(result, Exception):
                self.logger.error(f"{name} failed: {result}")
                results['findings'][name] = {'error': str(result)}
            else:
                results['findings'][name] = result
        
        # Generate timeline events
        results['timeline_events'] = await self._generate_timeline_events(results['findings'])
        
        # Calculate severity
        results['severity'] = self._calculate_severity(results['findings'])
        
        results['analysis_end'] = datetime.now().isoformat()
        
        return results
    
    async def _analyze_packets(self, capture_path: Path) -> Dict[str, Any]:
        """Analyze individual packets for anomalies"""
        packet_stats = {
            'total_packets': 0,
            'protocols': defaultdict(int),
            'suspicious_packets': [],
            'malformed_packets': [],
            'large_packets': [],
            'fragmented_packets': []
        }
        
        try:
            cap = pyshark.FileCapture(str(capture_path))
            
            for packet in cap:
                packet_stats['total_packets'] += 1
                
                # Protocol statistics
                if hasattr(packet, 'highest_layer'):
                    packet_stats['protocols'][packet.highest_layer] += 1
                
                # Check for suspicious characteristics
                suspicion_reasons = []
                
                # Check for malformed packets
                if hasattr(packet, 'malformed'):
                    packet_stats['malformed_packets'].append({
                        'number': packet.number,
                        'time': packet.sniff_time.isoformat() if hasattr(packet, 'sniff_time') else None,
                        'info': str(packet.malformed)
                    })
                    suspicion_reasons.append("Malformed packet")
                
                # Check for large packets
                if hasattr(packet, 'length') and int(packet.length) > 1500:
                    packet_stats['large_packets'].append({
                        'number': packet.number,
                        'size': int(packet.length),
                        'protocol': packet.highest_layer if hasattr(packet, 'highest_layer') else 'unknown'
                    })
                    if int(packet.length) > 9000:
                        suspicion_reasons.append(f"Unusually large packet: {packet.length} bytes")
                
                # Check for fragmentation
                if hasattr(packet, 'ip'):
                    if hasattr(packet.ip, 'flags_mf') and packet.ip.flags_mf == '1':
                        packet_stats['fragmented_packets'].append({
                            'number': packet.number,
                            'src': packet.ip.src,
                            'dst': packet.ip.dst
                        })
                
                # Check for suspicious ports
                suspicious_ports = [445, 3389, 23, 21, 1433, 3306, 5900, 22]
                if hasattr(packet, 'tcp'):
                    if hasattr(packet.tcp, 'dstport'):
                        if int(packet.tcp.dstport) in suspicious_ports:
                            suspicion_reasons.append(f"Connection to suspicious port: {packet.tcp.dstport}")
                
                if suspicion_reasons:
                    packet_stats['suspicious_packets'].append({
                        'number': packet.number,
                        'time': packet.sniff_time.isoformat() if hasattr(packet, 'sniff_time') else None,
                        'src': self._get_packet_src(packet),
                        'dst': self._get_packet_dst(packet),
                        'reasons': suspicion_reasons
                    })
                
                # Limit processing for large captures
                if packet_stats['total_packets'] > 100000:
                    self.logger.warning("Large capture file, limiting packet analysis")
                    break
            
            cap.close()
            
        except Exception as e:
            self.logger.error(f"Packet analysis error: {e}")
            packet_stats['error'] = str(e)
        
        # Convert defaultdict to regular dict
        packet_stats['protocols'] = dict(packet_stats['protocols'])
        
        return packet_stats
    
    def _get_packet_src(self, packet) -> str:
        """Extract source address from packet"""
        if hasattr(packet, 'ip'):
            return packet.ip.src
        elif hasattr(packet, 'ipv6'):
            return packet.ipv6.src
        elif hasattr(packet, 'eth'):
            return packet.eth.src
        return 'unknown'
    
    def _get_packet_dst(self, packet) -> str:
        """Extract destination address from packet"""
        if hasattr(packet, 'ip'):
            return packet.ip.dst
        elif hasattr(packet, 'ipv6'):
            return packet.ipv6.dst
        elif hasattr(packet, 'eth'):
            return packet.eth.dst
        return 'unknown'
    
    async def _reconstruct_flows(self, capture_path: Path) -> Dict[str, Any]:
        """Reconstruct network flows and conversations"""
        flows = await self.flow_reconstructor.reconstruct(capture_path)
        
        flow_stats = {
            'total_flows': len(flows),
            'top_talkers': [],
            'long_connections': [],
            'data_transfers': [],
            'suspicious_flows': []
        }
        
        # Analyze flows
        src_bytes = defaultdict(int)
        dst_bytes = defaultdict(int)
        
        for flow in flows:
            # Track top talkers
            src_bytes[flow['src_ip']] += flow['bytes_sent']
            dst_bytes[flow['dst_ip']] += flow['bytes_received']
            
            # Long connections (potential tunnels or C2)
            if flow['duration'] > 3600:  # 1 hour
                flow_stats['long_connections'].append({
                    'src': f"{flow['src_ip']}:{flow['src_port']}",
                    'dst': f"{flow['dst_ip']}:{flow['dst_port']}",
                    'duration': flow['duration'],
                    'bytes': flow['bytes_sent'] + flow['bytes_received']
                })
            
            # Large data transfers
            total_bytes = flow['bytes_sent'] + flow['bytes_received']
            if total_bytes > 100 * 1024 * 1024:  # 100MB
                flow_stats['data_transfers'].append({
                    'src': f"{flow['src_ip']}:{flow['src_port']}",
                    'dst': f"{flow['dst_ip']}:{flow['dst_port']}",
                    'bytes_sent': flow['bytes_sent'],
                    'bytes_received': flow['bytes_received'],
                    'total_bytes': total_bytes
                })
            
            # Check for suspicious patterns
            suspicion_score = 0
            reasons = []
            
            # Check against threat intelligence
            if await self.threat_intel.is_malicious_ip(flow['dst_ip']):
                suspicion_score += 100
                reasons.append(f"Destination IP {flow['dst_ip']} is known malicious")
            
            # Suspicious port combinations
            if flow['dst_port'] in [4444, 5555, 6666, 7777, 8888, 9999]:
                suspicion_score += 50
                reasons.append(f"Common backdoor port: {flow['dst_port']}")
            
            # Periodic connections (beaconing)
            if flow.get('periodic_score', 0) > 0.8:
                suspicion_score += 70
                reasons.append("Periodic connection pattern (possible beaconing)")
            
            if suspicion_score > 50:
                flow_stats['suspicious_flows'].append({
                    'flow': f"{flow['src_ip']}:{flow['src_port']} -> {flow['dst_ip']}:{flow['dst_port']}",
                    'suspicion_score': suspicion_score,
                    'reasons': reasons,
                    'bytes': total_bytes,
                    'packets': flow['packet_count']
                })
        
        # Get top talkers
        top_sources = sorted(src_bytes.items(), key=lambda x: x[1], reverse=True)[:10]
        top_destinations = sorted(dst_bytes.items(), key=lambda x: x[1], reverse=True)[:10]
        
        flow_stats['top_talkers'] = {
            'sources': [{'ip': ip, 'bytes': bytes_val} for ip, bytes_val in top_sources],
            'destinations': [{'ip': ip, 'bytes': bytes_val} for ip, bytes_val in top_destinations]
        }
        
        return flow_stats
    
    async def _detect_intrusions(self, capture_path: Path) -> Dict[str, Any]:
        """Detect intrusion attempts and attacks"""
        intrusions = {
            'port_scans': [],
            'brute_force_attempts': [],
            'dos_attacks': [],
            'exploit_attempts': [],
            'anomalous_traffic': []
        }
        
        # Port scan detection
        port_scans = await self._detect_port_scans(capture_path)
        intrusions['port_scans'] = port_scans
        
        # Brute force detection
        brute_force = await self._detect_brute_force(capture_path)
        intrusions['brute_force_attempts'] = brute_force
        
        # DoS attack detection
        dos_attacks = await self._detect_dos_attacks(capture_path)
        intrusions['dos_attacks'] = dos_attacks
        
        # Exploit attempt detection
        exploits = await self._detect_exploits(capture_path)
        intrusions['exploit_attempts'] = exploits
        
        return intrusions
    
    async def _detect_port_scans(self, capture_path: Path) -> List[Dict]:
        """Detect port scanning activity"""
        scans = []
        src_port_count = defaultdict(set)
        
        try:
            with open(capture_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                
                for timestamp, buf in pcap:
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        if isinstance(eth.data, dpkt.ip.IP):
                            ip = eth.data
                            if isinstance(ip.data, dpkt.tcp.TCP):
                                tcp = ip.data
                                
                                # Track unique destination ports per source IP
                                src_ip = socket.inet_ntoa(ip.src)
                                dst_port = tcp.dport
                                
                                # SYN flag set, ACK not set (typical scan)
                                if (tcp.flags & dpkt.tcp.TH_SYN) and not (tcp.flags & dpkt.tcp.TH_ACK):
                                    src_port_count[src_ip].add(dst_port)
                    except:
                        continue
        except Exception as e:
            self.logger.error(f"Port scan detection error: {e}")
        
        # Identify port scanners
        for src_ip, ports in src_port_count.items():
            if len(ports) > 20:  # More than 20 different ports
                scans.append({
                    'source_ip': src_ip,
                    'port_count': len(ports),
                    'sample_ports': list(ports)[:10],
                    'scan_type': 'SYN scan' if len(ports) > 100 else 'Targeted scan'
                })
        
        return scans
    
    async def _detect_brute_force(self, capture_path: Path) -> List[Dict]:
        """Detect brute force login attempts"""
        brute_force_attempts = []
        auth_attempts = defaultdict(list)
        
        # Common authentication ports
        auth_ports = {
            21: 'FTP',
            22: 'SSH',
            23: 'Telnet',
            3389: 'RDP',
            5900: 'VNC',
            3306: 'MySQL',
            1433: 'MSSQL',
            5432: 'PostgreSQL'
        }
        
        try:
            cap = pyshark.FileCapture(str(capture_path), 
                                     display_filter='tcp.port==22 or tcp.port==3389 or tcp.port==21')
            
            for packet in cap:
                if hasattr(packet, 'tcp') and hasattr(packet, 'ip'):
                    dst_port = int(packet.tcp.dstport)
                    if dst_port in auth_ports:
                        key = f"{packet.ip.src}->{packet.ip.dst}:{dst_port}"
                        auth_attempts[key].append(packet.sniff_time)
            
            cap.close()
            
            # Analyze attempts
            for connection, timestamps in auth_attempts.items():
                if len(timestamps) > 10:  # More than 10 attempts
                    src, dst = connection.split('->')
                    dst_ip, dst_port = dst.split(':')
                    
                    # Calculate attempt rate
                    if len(timestamps) > 1:
                        time_span = (timestamps[-1] - timestamps[0]).total_seconds()
                        rate = len(timestamps) / time_span if time_span > 0 else 0
                        
                        if rate > 0.5:  # More than 1 attempt per 2 seconds
                            brute_force_attempts.append({
                                'source_ip': src,
                                'target': dst,
                                'service': auth_ports[int(dst_port)],
                                'attempt_count': len(timestamps),
                                'duration_seconds': time_span,
                                'rate_per_second': rate
                            })
            
        except Exception as e:
            self.logger.error(f"Brute force detection error: {e}")
        
        return brute_force_attempts
    
    async def _detect_dos_attacks(self, capture_path: Path) -> List[Dict]:
        """Detect DoS/DDoS attacks"""
        dos_indicators = []
        
        # Track packet rates
        packet_rates = defaultdict(lambda: defaultdict(int))
        syn_flood_candidates = defaultdict(int)
        
        try:
            with open(capture_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                
                start_time = None
                current_second = 0
                
                for timestamp, buf in pcap:
                    if start_time is None:
                        start_time = timestamp
                    
                    # Calculate current second
                    second = int(timestamp - start_time)
                    
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        if isinstance(eth.data, dpkt.ip.IP):
                            ip = eth.data
                            src_ip = socket.inet_ntoa(ip.src)
                            dst_ip = socket.inet_ntoa(ip.dst)
                            
                            # Track packet rates per second
                            packet_rates[second][f"{src_ip}->{dst_ip}"] += 1
                            
                            # Check for SYN flood
                            if isinstance(ip.data, dpkt.tcp.TCP):
                                tcp = ip.data
                                if (tcp.flags & dpkt.tcp.TH_SYN) and not (tcp.flags & dpkt.tcp.TH_ACK):
                                    syn_flood_candidates[dst_ip] += 1
                    except:
                        continue
        except Exception as e:
            self.logger.error(f"DoS detection error: {e}")
        
        # Analyze for DoS patterns
        for second, connections in packet_rates.items():
            for conn, count in connections.items():
                if count > 1000:  # More than 1000 packets per second
                    src, dst = conn.split('->')
                    dos_indicators.append({
                        'type': 'High packet rate',
                        'source': src,
                        'target': dst,
                        'packets_per_second': count,
                        'timestamp_offset': second
                    })
        
        # Check SYN flood candidates
        for dst_ip, syn_count in syn_flood_candidates.items():
            if syn_count > 1000:
                dos_indicators.append({
                    'type': 'SYN flood',
                    'target': dst_ip,
                    'syn_packets': syn_count
                })
        
        return dos_indicators
    
    async def _detect_exploits(self, capture_path: Path) -> List[Dict]:
        """Detect known exploit patterns"""
        exploits = []
        
        # Common exploit signatures
        exploit_signatures = {
            b'\x90\x90\x90\x90': 'NOP sled (buffer overflow)',
            b'../../': 'Directory traversal',
            b'<script>': 'XSS attempt',
            b'SELECT * FROM': 'SQL injection',
            b'cmd.exe': 'Command injection',
            b'/etc/passwd': 'Unix file access',
            b'UNION SELECT': 'SQL injection',
            b'\x41\x41\x41\x41': 'Buffer overflow pattern'
        }
        
        try:
            with open(capture_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                
                packet_num = 0
                for timestamp, buf in pcap:
                    packet_num += 1
                    
                    # Check payload for exploit signatures
                    for signature, description in exploit_signatures.items():
                        if signature in buf:
                            try:
                                eth = dpkt.ethernet.Ethernet(buf)
                                if isinstance(eth.data, dpkt.ip.IP):
                                    ip = eth.data
                                    src_ip = socket.inet_ntoa(ip.src)
                                    dst_ip = socket.inet_ntoa(ip.dst)
                                    
                                    exploits.append({
                                        'packet_number': packet_num,
                                        'type': description,
                                        'source': src_ip,
                                        'destination': dst_ip,
                                        'signature': signature.hex(),
                                        'timestamp': datetime.fromtimestamp(timestamp).isoformat()
                                    })
                            except:
                                pass
                    
                    # Limit processing
                    if packet_num > 50000:
                        break
        
        except Exception as e:
            self.logger.error(f"Exploit detection error: {e}")
        
        return exploits
    
    async def _analyze_protocols(self, capture_path: Path) -> Dict[str, Any]:
        """Analyze protocol usage and anomalies"""
        protocol_stats = {
            'http_analysis': await self._analyze_http(capture_path),
            'dns_analysis': await self._analyze_dns(capture_path),
            'smtp_analysis': await self._analyze_smtp(capture_path),
            'unusual_protocols': []
        }
        
        return protocol_stats
    
    async def _analyze_http(self, capture_path: Path) -> Dict[str, Any]:
        """Analyze HTTP traffic"""
        http_data = {
            'requests': [],
            'suspicious_user_agents': [],
            'suspicious_requests': [],
            'file_downloads': []
        }
        
        suspicious_user_agents = [
            'sqlmap', 'nikto', 'nmap', 'masscan', 'metasploit',
            'havij', 'acunetix', 'burp', 'python-requests'
        ]
        
        suspicious_paths = [
            'admin', 'administrator', 'wp-admin', 'phpmyadmin',
            '.git', '.env', 'config.php', 'web.config'
        ]
        
        try:
            cap = pyshark.FileCapture(str(capture_path), display_filter='http')
            
            for packet in cap:
                if hasattr(packet, 'http'):
                    # Extract HTTP request data
                    if hasattr(packet.http, 'request_method'):
                        request = {
                            'method': packet.http.request_method,
                            'uri': packet.http.request_uri if hasattr(packet.http, 'request_uri') else '',
                            'host': packet.http.host if hasattr(packet.http, 'host') else '',
                            'user_agent': packet.http.user_agent if hasattr(packet.http, 'user_agent') else '',
                            'src_ip': packet.ip.src if hasattr(packet, 'ip') else '',
                            'dst_ip': packet.ip.dst if hasattr(packet, 'ip') else ''
                        }
                        
                        http_data['requests'].append(request)
                        
                        # Check for suspicious user agents
                        if request['user_agent']:
                            for sus_agent in suspicious_user_agents:
                                if sus_agent.lower() in request['user_agent'].lower():
                                    http_data['suspicious_user_agents'].append({
                                        'user_agent': request['user_agent'],
                                        'source_ip': request['src_ip'],
                                        'target': request['host']
                                    })
                                    break
                        
                        # Check for suspicious paths
                        for sus_path in suspicious_paths:
                            if sus_path in request['uri'].lower():
                                http_data['suspicious_requests'].append({
                                    'uri': request['uri'],
                                    'method': request['method'],
                                    'source_ip': request['src_ip'],
                                    'suspicious_element': sus_path
                                })
                                break
                    
                    # Check for file downloads
                    if hasattr(packet.http, 'response_code') and hasattr(packet.http, 'content_type'):
                        if packet.http.response_code == '200':
                            content_type = packet.http.content_type
                            if any(ct in content_type for ct in ['application/', 'binary', 'executable']):
                                http_data['file_downloads'].append({
                                    'content_type': content_type,
                                    'server': packet.http.server if hasattr(packet.http, 'server') else '',
                                    'size': packet.http.content_length if hasattr(packet.http, 'content_length') else 'unknown'
                                })
            
            cap.close()
            
        except Exception as e:
            self.logger.error(f"HTTP analysis error: {e}")
        
        return http_data
    
    async def _analyze_dns(self, capture_path: Path) -> Dict[str, Any]:
        """Analyze DNS traffic for suspicious queries"""
        dns_data = {
            'total_queries': 0,
            'unique_domains': set(),
            'suspicious_domains': [],
            'dga_candidates': [],
            'tunneling_suspects': []
        }
        
        try:
            cap = pyshark.FileCapture(str(capture_path), display_filter='dns')
            
            domain_lengths = []
            
            for packet in cap:
                if hasattr(packet, 'dns'):
                    dns_data['total_queries'] += 1
                    
                    # Extract queried domain
                    if hasattr(packet.dns, 'qry_name'):
                        domain = packet.dns.qry_name
                        dns_data['unique_domains'].add(domain)
                        domain_lengths.append(len(domain))
                        
                        # Check for suspicious patterns
                        
                        # DGA detection (high entropy, unusual length)
                        if len(domain) > 20 and self._calculate_entropy(domain) > 4.0:
                            dns_data['dga_candidates'].append({
                                'domain': domain,
                                'entropy': self._calculate_entropy(domain),
                                'length': len(domain)
                            })
                        
                        # DNS tunneling detection (long subdomains)
                        parts = domain.split('.')
                        if any(len(part) > 50 for part in parts):
                            dns_data['tunneling_suspects'].append({
                                'domain': domain,
                                'max_label_length': max(len(part) for part in parts)
                            })
                        
                        # Check against threat intelligence
                        if await self.threat_intel.is_malicious_domain(domain):
                            dns_data['suspicious_domains'].append({
                                'domain': domain,
                                'reason': 'Known malicious domain'
                            })
            
            cap.close()
            
        except Exception as e:
            self.logger.error(f"DNS analysis error: {e}")
        
        dns_data['unique_domains'] = list(dns_data['unique_domains'])[:100]  # Limit output
        
        return dns_data
    
    def _calculate_entropy(self, string: str) -> float:
        """Calculate Shannon entropy of a string"""
        import math
        from collections import Counter
        
        if not string:
            return 0
        
        # Calculate frequency of each character
        counts = Counter(string)
        length = len(string)
        
        # Calculate entropy
        entropy = 0
        for count in counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    async def _analyze_smtp(self, capture_path: Path) -> Dict[str, Any]:
        """Analyze SMTP traffic for spam/phishing"""
        smtp_data = {
            'emails_sent': 0,
            'suspicious_emails': [],
            'attachments': []
        }
        
        # Implementation would analyze SMTP traffic
        # Looking for spam patterns, suspicious attachments, etc.
        
        return smtp_data
    
    async def _detect_data_exfiltration(self, capture_path: Path) -> Dict[str, Any]:
        """Detect potential data exfiltration"""
        exfiltration = {
            'large_uploads': [],
            'unusual_destinations': [],
            'encrypted_channels': [],
            'steganography_suspects': []
        }
        
        # Track data transfers
        flows = await self.flow_reconstructor.reconstruct(capture_path)
        
        for flow in flows:
            # Large outbound transfers
            if flow['bytes_sent'] > 50 * 1024 * 1024:  # 50MB
                # Check if destination is external
                if not self._is_internal_ip(flow['dst_ip']):
                    exfiltration['large_uploads'].append({
                        'source': f"{flow['src_ip']}:{flow['src_port']}",
                        'destination': f"{flow['dst_ip']}:{flow['dst_port']}",
                        'bytes': flow['bytes_sent'],
                        'duration': flow['duration']
                    })
            
            # Unusual destinations
            if flow['dst_port'] not in [80, 443, 22, 21, 25]:
                if flow['bytes_sent'] > 10 * 1024 * 1024:  # 10MB
                    exfiltration['unusual_destinations'].append({
                        'destination': f"{flow['dst_ip']}:{flow['dst_port']}",
                        'bytes_sent': flow['bytes_sent'],
                        'protocol': flow.get('protocol', 'unknown')
                    })
        
        return exfiltration
    
    def _is_internal_ip(self, ip: str) -> bool:
        """Check if IP is internal/private"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private
        except:
            return False
    
    async def _detect_c2_communication(self, capture_path: Path) -> Dict[str, Any]:
        """Detect command and control communication patterns"""
        c2_indicators = {
            'beaconing': [],
            'known_c2_servers': [],
            'suspicious_patterns': []
        }
        
        # Analyze flows for beaconing behavior
        flows = await self.flow_reconstructor.reconstruct(capture_path)
        
        # Group flows by source and destination
        connection_groups = defaultdict(list)
        for flow in flows:
            key = f"{flow['src_ip']}->{flow['dst_ip']}:{flow['dst_port']}"
            connection_groups[key].append(flow)
        
        # Look for beaconing patterns
        for connection, flows in connection_groups.items():
            if len(flows) > 5:  # At least 5 connections
                # Calculate time intervals
                intervals = []
                for i in range(1, len(flows)):
                    interval = flows[i]['start_time'] - flows[i-1]['start_time']
                    intervals.append(interval)
                
                if intervals:
                    # Check for regular intervals (beaconing)
                    avg_interval = sum(intervals) / len(intervals)
                    variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
                    
                    # Low variance indicates regular intervals
                    if variance < (avg_interval * 0.1) ** 2:  # Less than 10% variation
                        c2_indicators['beaconing'].append({
                            'connection': connection,
                            'beacon_count': len(flows),
                            'average_interval': avg_interval,
                            'variance': variance
                        })
        
        # Check against known C2 servers
        for flow in flows:
            if await self.threat_intel.is_c2_server(flow['dst_ip']):
                c2_indicators['known_c2_servers'].append({
                    'server_ip': flow['dst_ip'],
                    'source_ip': flow['src_ip'],
                    'bytes_exchanged': flow['bytes_sent'] + flow['bytes_received']
                })
        
        return c2_indicators
    
    async def _analyze_ssl_tls(self, capture_path: Path) -> Dict[str, Any]:
        """Analyze SSL/TLS traffic"""
        ssl_data = {
            'certificates': [],
            'cipher_suites': [],
            'suspicious_certificates': [],
            'weak_ciphers': []
        }
        
        weak_ciphers = [
            'DES', 'RC4', 'MD5', 'EXPORT', 'NULL', 'anon'
        ]
        
        try:
            cap = pyshark.FileCapture(str(capture_path), display_filter='ssl or tls')
            
            for packet in cap:
                if hasattr(packet, 'ssl') or hasattr(packet, 'tls'):
                    layer = packet.ssl if hasattr(packet, 'ssl') else packet.tls
                    
                    # Extract certificate information
                    if hasattr(layer, 'handshake_certificate'):
                        cert_info = {
                            'subject': layer.x509ce_attributevalue if hasattr(layer, 'x509ce_attributevalue') else '',
                            'issuer': layer.x509ce_issuer if hasattr(layer, 'x509ce_issuer') else '',
                            'serial': layer.x509ce_serialnumber if hasattr(layer, 'x509ce_serialnumber') else ''
                        }
                        ssl_data['certificates'].append(cert_info)
                        
                        # Check for self-signed or suspicious certificates
                        if cert_info['subject'] == cert_info['issuer']:
                            ssl_data['suspicious_certificates'].append({
                                'type': 'self-signed',
                                'certificate': cert_info
                            })
                    
                    # Extract cipher suite
                    if hasattr(layer, 'handshake_ciphersuite'):
                        cipher = layer.handshake_ciphersuite
                        ssl_data['cipher_suites'].append(cipher)
                        
                        # Check for weak ciphers
                        for weak in weak_ciphers:
                            if weak in str(cipher).upper():
                                ssl_data['weak_ciphers'].append({
                                    'cipher': cipher,
                                    'weakness': weak
                                })
                                break
            
            cap.close()
            
        except Exception as e:
            self.logger.error(f"SSL/TLS analysis error: {e}")
        
        return ssl_data
    
    async def _generate_timeline_events(self, findings: Dict) -> List[Dict]:
        """Generate timeline events from network findings"""
        events = []
        
        # Add intrusion events
        if 'intrusion_detection' in findings:
            for scan in findings['intrusion_detection'].get('port_scans', []):
                events.append({
                    'timestamp': datetime.now().isoformat(),  # Would use actual packet time
                    'type': 'port_scan',
                    'description': f"Port scan from {scan['source_ip']} ({scan['port_count']} ports)",
                    'severity': 'high',
                    'data': scan
                })
            
            for brute in findings['intrusion_detection'].get('brute_force_attempts', []):
                events.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'brute_force',
                    'description': f"Brute force attempt on {brute['service']} from {brute['source_ip']}",
                    'severity': 'critical',
                    'data': brute
                })
        
        # Add C2 communication events
        if 'c2_detection' in findings:
            for beacon in findings['c2_detection'].get('beaconing', []):
                events.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'c2_beacon',
                    'description': f"Potential C2 beaconing: {beacon['connection']}",
                    'severity': 'critical',
                    'data': beacon
                })
        
        return sorted(events, key=lambda x: x['timestamp'])
    
    def _calculate_severity(self, findings: Dict) -> str:
        """Calculate overall severity based on findings"""
        severity_score = 0
        
        # Check for intrusions
        if findings.get('intrusion_detection', {}):
            if findings['intrusion_detection'].get('brute_force_attempts'):
                severity_score += 80
            if findings['intrusion_detection'].get('exploit_attempts'):
                severity_score += 100
            if findings['intrusion_detection'].get('dos_attacks'):
                severity_score += 90
        
        # Check for C2 communication
        if findings.get('c2_detection', {}).get('beaconing'):
            severity_score += 100
        if findings.get('c2_detection', {}).get('known_c2_servers'):
            severity_score += 120
        
        # Check for data exfiltration
        if findings.get('data_exfiltration', {}).get('large_uploads'):
            severity_score += 60
        
        # Determine severity level
        if severity_score >= 100:
            return 'critical'
        elif severity_score >= 60:
            return 'high'
        elif severity_score >= 30:
            return 'medium'
        else:
            return 'low'
    
    async def _cleanup(self):
        """Cleanup agent resources"""
        if self.packet_analyzer:
            await self.packet_analyzer.cleanup()
        if self.flow_reconstructor:
            await self.flow_reconstructor.cleanup()
        if self.threat_intel:
            await self.threat_intel.cleanup()