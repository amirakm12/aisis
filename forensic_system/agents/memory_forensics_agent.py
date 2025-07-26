"""
Memory Forensics Agent
=====================

Specialized agent for analyzing memory dumps, detecting malware,
and extracting volatile artifacts
"""

import asyncio
import re
import struct
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yara
import pefile
from datetime import datetime

from .base_agent import BaseAgent, AgentCapabilities
from ..utils.memory_utils import MemoryParser, ProcessInfo
from ..utils.pattern_matcher import PatternMatcher


class MemoryForensicsAgent(BaseAgent):
    """
    Advanced memory forensics agent with capabilities:
    - Process analysis and anomaly detection
    - Malware detection using YARA rules
    - Network connection extraction
    - Registry key extraction from memory
    - Cryptographic key recovery
    - Hidden process detection
    """
    
    def __init__(self):
        super().__init__("MemoryForensicsAgent", "2.0.0")
        self.yara_rules = None
        self.memory_parser = None
        self.pattern_matcher = None
        
    def _define_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            evidence_types=["memory_dump", "hibernation_file", "crash_dump"],
            analysis_types=[
                "process_analysis",
                "malware_detection", 
                "network_connections",
                "registry_extraction",
                "crypto_key_recovery",
                "rootkit_detection"
            ],
            required_tools=["volatility3", "yara", "strings"],
            parallel_capable=True,
            gpu_accelerated=True
        )
    
    async def _setup(self):
        """Initialize memory forensics tools"""
        # Load YARA rules
        self.yara_rules = await self._load_yara_rules()
        
        # Initialize memory parser
        self.memory_parser = MemoryParser()
        
        # Initialize pattern matcher for various artifacts
        self.pattern_matcher = PatternMatcher()
        await self.pattern_matcher.load_patterns({
            'ipv4': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'ipv6': r'(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4})',
            'url': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'bitcoin': r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',
            'private_key': r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC )?PRIVATE KEY-----',
            'password_hash': r'\$[0-9a-z]+\$[0-9]+\$[a-zA-Z0-9./]+',
            'registry_key': r'(?:HKEY_[A-Z_]+\\[\\A-Za-z0-9_]+)+',
            'mutex': r'(?:Global\\|Local\\)[A-Za-z0-9_\-]+',
            'command_line': r'(?:cmd\.exe|powershell\.exe|bash|sh)\s+.*'
        })
        
    async def _load_yara_rules(self) -> yara.Rules:
        """Load YARA rules for malware detection"""
        rules_path = Path(__file__).parent.parent / "rules" / "yara"
        
        # Compile all YARA rules
        rule_files = list(rules_path.glob("*.yar"))
        if not rule_files:
            # Create default rules if none exist
            await self._create_default_yara_rules(rules_path)
            rule_files = list(rules_path.glob("*.yar"))
        
        filepaths = {str(f.stem): str(f) for f in rule_files}
        return yara.compile(filepaths=filepaths)
    
    async def _create_default_yara_rules(self, rules_path: Path):
        """Create default YARA rules for common malware"""
        rules_path.mkdir(parents=True, exist_ok=True)
        
        # Basic malware detection rules
        default_rules = """
rule Suspicious_API_Calls
{
    meta:
        description = "Detects suspicious API call patterns"
        severity = "medium"
        
    strings:
        $api1 = "VirtualAllocEx"
        $api2 = "WriteProcessMemory"
        $api3 = "CreateRemoteThread"
        $api4 = "SetWindowsHookEx"
        $api5 = "NtQuerySystemInformation"
        
    condition:
        3 of them
}

rule Ransomware_Indicators
{
    meta:
        description = "Detects potential ransomware indicators"
        severity = "critical"
        
    strings:
        $s1 = "Your files have been encrypted"
        $s2 = "bitcoin"
        $s3 = "decrypt"
        $s4 = ".locked"
        $s5 = "AES-256"
        $api1 = "CryptEncrypt"
        $api2 = "CryptGenKey"
        
    condition:
        2 of ($s*) and 1 of ($api*)
}

rule Rootkit_Behavior
{
    meta:
        description = "Detects rootkit-like behavior"
        severity = "critical"
        
    strings:
        $s1 = "\\Device\\PhysicalMemory"
        $s2 = "ZwQuerySystemInformation"
        $s3 = "KeServiceDescriptorTable"
        $s4 = "NtOpenProcess"
        $s5 = "SSDT"
        
    condition:
        3 of them
}
"""
        
        with open(rules_path / "default_malware.yar", "w") as f:
            f.write(default_rules)
    
    async def analyze(self, evidence_data: Dict[str, Any], 
                     case_context: Any) -> Dict[str, Any]:
        """Perform comprehensive memory analysis"""
        if not await self.validate_evidence(evidence_data):
            return {"error": "Invalid evidence type for memory analysis"}
        
        self.logger.info(f"Starting memory analysis for {evidence_data['id']}")
        
        memory_path = Path(evidence_data['path'])
        results = {
            'agent': self.name,
            'version': self.version,
            'analysis_start': datetime.now().isoformat(),
            'evidence_id': evidence_data['id'],
            'findings': {}
        }
        
        # Run analyses in parallel
        analysis_tasks = [
            self._analyze_processes(memory_path),
            self._detect_malware(memory_path),
            self._extract_network_artifacts(memory_path),
            self._extract_registry_artifacts(memory_path),
            self._detect_hidden_processes(memory_path),
            self._extract_crypto_artifacts(memory_path)
        ]
        
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Process results
        analysis_names = [
            'process_analysis',
            'malware_detection',
            'network_artifacts',
            'registry_artifacts', 
            'hidden_processes',
            'crypto_artifacts'
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
    
    async def _analyze_processes(self, memory_path: Path) -> Dict[str, Any]:
        """Analyze running processes from memory"""
        processes = await self.memory_parser.extract_processes(memory_path)
        
        suspicious_processes = []
        process_tree = {}
        
        for proc in processes:
            # Check for suspicious characteristics
            suspicion_score = 0
            suspicion_reasons = []
            
            # Check for suspicious process names
            suspicious_names = [
                'svchost.exe', 'csrss.exe', 'lsass.exe', 'services.exe'
            ]
            
            if proc.name.lower() in suspicious_names:
                # Verify if it's in the correct path
                if not proc.path.lower().startswith('c:\\windows\\system32'):
                    suspicion_score += 50
                    suspicion_reasons.append("Process in unexpected location")
            
            # Check for processes without a path
            if not proc.path:
                suspicion_score += 30
                suspicion_reasons.append("No process path")
            
            # Check for high thread count
            if proc.thread_count > 100:
                suspicion_score += 20
                suspicion_reasons.append(f"High thread count: {proc.thread_count}")
            
            # Check for suspicious parent-child relationships
            if proc.parent_pid == 4 and proc.name.lower() != 'system':
                suspicion_score += 40
                suspicion_reasons.append("Suspicious parent process")
            
            if suspicion_score > 30:
                suspicious_processes.append({
                    'pid': proc.pid,
                    'name': proc.name,
                    'path': proc.path,
                    'parent_pid': proc.parent_pid,
                    'threads': proc.thread_count,
                    'suspicion_score': suspicion_score,
                    'reasons': suspicion_reasons
                })
            
            # Build process tree
            if proc.parent_pid not in process_tree:
                process_tree[proc.parent_pid] = []
            process_tree[proc.parent_pid].append(proc.pid)
        
        return {
            'total_processes': len(processes),
            'suspicious_processes': suspicious_processes,
            'process_tree': process_tree,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def _detect_malware(self, memory_path: Path) -> Dict[str, Any]:
        """Detect malware using YARA rules"""
        matches = []
        
        # Scan memory with YARA rules
        try:
            matches_raw = self.yara_rules.match(str(memory_path))
            
            for match in matches_raw:
                match_info = {
                    'rule': match.rule,
                    'namespace': match.namespace,
                    'tags': match.tags,
                    'meta': match.meta,
                    'strings': []
                }
                
                for string in match.strings:
                    match_info['strings'].append({
                        'offset': string[0],
                        'identifier': string[1],
                        'data': string[2].decode('utf-8', errors='ignore')[:100]
                    })
                
                matches.append(match_info)
                
        except Exception as e:
            self.logger.error(f"YARA scan failed: {e}")
        
        # Additional heuristic detection
        heuristic_detections = await self._heuristic_malware_detection(memory_path)
        
        return {
            'yara_matches': matches,
            'heuristic_detections': heuristic_detections,
            'malware_detected': len(matches) > 0 or len(heuristic_detections) > 0
        }
    
    async def _heuristic_malware_detection(self, memory_path: Path) -> List[Dict]:
        """Perform heuristic malware detection"""
        detections = []
        
        # Check for code injection patterns
        injection_patterns = [
            b'\x50\x53\x51\x52\x56\x57\x55',  # Push all registers
            b'\x60\x61',  # PUSHAD/POPAD
            b'\xE8\x00\x00\x00\x00',  # CALL $+5
            b'\x64\xA1\x30\x00\x00\x00'  # MOV EAX, FS:[30h] - PEB access
        ]
        
        with open(memory_path, 'rb') as f:
            # Read in chunks to avoid memory issues
            chunk_size = 1024 * 1024  # 1MB chunks
            offset = 0
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                for pattern in injection_patterns:
                    if pattern in chunk:
                        detections.append({
                            'type': 'code_injection_pattern',
                            'offset': offset + chunk.find(pattern),
                            'pattern': pattern.hex(),
                            'confidence': 'medium'
                        })
                
                offset += len(chunk)
        
        return detections
    
    async def _extract_network_artifacts(self, memory_path: Path) -> Dict[str, Any]:
        """Extract network connections and artifacts"""
        network_data = {
            'connections': [],
            'dns_cache': [],
            'urls': [],
            'ip_addresses': []
        }
        
        # Extract network connections
        connections = await self.memory_parser.extract_network_connections(memory_path)
        
        for conn in connections:
            network_data['connections'].append({
                'local_addr': f"{conn.local_ip}:{conn.local_port}",
                'remote_addr': f"{conn.remote_ip}:{conn.remote_port}",
                'state': conn.state,
                'pid': conn.pid,
                'protocol': conn.protocol
            })
        
        # Extract URLs and IPs using pattern matching
        with open(memory_path, 'rb') as f:
            content = f.read(100 * 1024 * 1024)  # Read first 100MB
            text_content = content.decode('utf-8', errors='ignore')
            
            # Find URLs
            urls = self.pattern_matcher.find_all('url', text_content)
            network_data['urls'] = list(set(urls))[:100]  # Limit to 100 unique URLs
            
            # Find IP addresses
            ipv4s = self.pattern_matcher.find_all('ipv4', text_content)
            ipv6s = self.pattern_matcher.find_all('ipv6', text_content)
            
            # Filter out common local IPs
            network_data['ip_addresses'] = [
                ip for ip in set(ipv4s + ipv6s)
                if not ip.startswith(('127.', '192.168.', '10.', '172.'))
            ][:100]
        
        return network_data
    
    async def _extract_registry_artifacts(self, memory_path: Path) -> Dict[str, Any]:
        """Extract registry artifacts from memory"""
        registry_data = {
            'keys': [],
            'suspicious_keys': [],
            'autostart_locations': []
        }
        
        # Common autostart registry locations
        autostart_locations = [
            r'HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Run',
            r'HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run',
            r'HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\RunOnce',
            r'HKEY_LOCAL_MACHINE\System\CurrentControlSet\Services',
            r'HKEY_LOCAL_MACHINE\Software\Microsoft\Windows NT\CurrentVersion\Winlogon'
        ]
        
        with open(memory_path, 'rb') as f:
            content = f.read(50 * 1024 * 1024)  # Read first 50MB
            text_content = content.decode('utf-8', errors='ignore')
            
            # Find registry keys
            registry_keys = self.pattern_matcher.find_all('registry_key', text_content)
            
            for key in set(registry_keys):
                registry_data['keys'].append(key)
                
                # Check if it's an autostart location
                for autostart in autostart_locations:
                    if autostart.lower() in key.lower():
                        registry_data['autostart_locations'].append({
                            'key': key,
                            'type': 'autostart',
                            'risk': 'high'
                        })
                
                # Check for suspicious patterns
                suspicious_patterns = [
                    'currentversion\\run',
                    'image file execution options',
                    'appinit_dlls',
                    'winlogon\\shell',
                    'winlogon\\userinit'
                ]
                
                for pattern in suspicious_patterns:
                    if pattern in key.lower():
                        registry_data['suspicious_keys'].append({
                            'key': key,
                            'pattern': pattern,
                            'risk': 'medium'
                        })
        
        return registry_data
    
    async def _detect_hidden_processes(self, memory_path: Path) -> Dict[str, Any]:
        """Detect hidden processes using various techniques"""
        hidden_processes = []
        
        # Get process list using different methods
        pslist_procs = await self.memory_parser.extract_processes(memory_path, method='pslist')
        psscan_procs = await self.memory_parser.extract_processes(memory_path, method='psscan')
        
        # Find processes that appear in psscan but not pslist (potentially hidden)
        pslist_pids = {p.pid for p in pslist_procs}
        
        for proc in psscan_procs:
            if proc.pid not in pslist_pids:
                hidden_processes.append({
                    'pid': proc.pid,
                    'name': proc.name,
                    'path': proc.path,
                    'detection_method': 'psscan_only',
                    'confidence': 'high'
                })
        
        # Check for DKOM (Direct Kernel Object Manipulation)
        dkom_indicators = await self._check_dkom_indicators(memory_path)
        
        return {
            'hidden_process_count': len(hidden_processes),
            'hidden_processes': hidden_processes,
            'dkom_indicators': dkom_indicators
        }
    
    async def _check_dkom_indicators(self, memory_path: Path) -> List[Dict]:
        """Check for DKOM rootkit indicators"""
        indicators = []
        
        # Check for manipulated EPROCESS structures
        # This is a simplified check - real implementation would be more complex
        
        return indicators
    
    async def _extract_crypto_artifacts(self, memory_path: Path) -> Dict[str, Any]:
        """Extract cryptographic artifacts"""
        crypto_data = {
            'private_keys': [],
            'certificates': [],
            'bitcoin_addresses': [],
            'password_hashes': []
        }
        
        with open(memory_path, 'rb') as f:
            content = f.read(50 * 1024 * 1024)  # Read first 50MB
            text_content = content.decode('utf-8', errors='ignore')
            
            # Find private keys
            private_keys = self.pattern_matcher.find_all('private_key', text_content)
            crypto_data['private_keys'] = [
                {'type': 'private_key', 'preview': key[:50] + '...'}
                for key in private_keys[:10]
            ]
            
            # Find bitcoin addresses
            bitcoin_addrs = self.pattern_matcher.find_all('bitcoin', text_content)
            crypto_data['bitcoin_addresses'] = list(set(bitcoin_addrs))[:20]
            
            # Find password hashes
            password_hashes = self.pattern_matcher.find_all('password_hash', text_content)
            crypto_data['password_hashes'] = [
                {'hash': h, 'type': self._identify_hash_type(h)}
                for h in set(password_hashes)[:20]
            ]
        
        return crypto_data
    
    def _identify_hash_type(self, hash_string: str) -> str:
        """Identify the type of password hash"""
        if hash_string.startswith('$2a$') or hash_string.startswith('$2b$'):
            return 'bcrypt'
        elif hash_string.startswith('$6$'):
            return 'sha512crypt'
        elif hash_string.startswith('$5$'):
            return 'sha256crypt'
        elif hash_string.startswith('$1$'):
            return 'md5crypt'
        else:
            return 'unknown'
    
    async def _generate_timeline_events(self, findings: Dict) -> List[Dict]:
        """Generate timeline events from findings"""
        events = []
        
        # Add process creation events
        if 'process_analysis' in findings:
            for proc in findings['process_analysis'].get('suspicious_processes', []):
                events.append({
                    'timestamp': datetime.now().isoformat(),  # Would use actual process creation time
                    'type': 'process_creation',
                    'description': f"Suspicious process: {proc['name']} (PID: {proc['pid']})",
                    'severity': 'high' if proc['suspicion_score'] > 50 else 'medium',
                    'data': proc
                })
        
        # Add network connection events
        if 'network_artifacts' in findings:
            for conn in findings['network_artifacts'].get('connections', []):
                if conn['state'] == 'ESTABLISHED':
                    events.append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'network_connection',
                        'description': f"Connection: {conn['local_addr']} -> {conn['remote_addr']}",
                        'severity': 'medium',
                        'data': conn
                    })
        
        return sorted(events, key=lambda x: x['timestamp'])
    
    def _calculate_severity(self, findings: Dict) -> str:
        """Calculate overall severity based on findings"""
        severity_score = 0
        
        # Check malware detection
        if findings.get('malware_detection', {}).get('malware_detected'):
            severity_score += 100
        
        # Check hidden processes
        hidden_count = findings.get('hidden_processes', {}).get('hidden_process_count', 0)
        if hidden_count > 0:
            severity_score += 50 * min(hidden_count, 3)
        
        # Check suspicious processes
        suspicious_procs = findings.get('process_analysis', {}).get('suspicious_processes', [])
        for proc in suspicious_procs:
            if proc['suspicion_score'] > 70:
                severity_score += 30
        
        # Determine severity level
        if severity_score >= 150:
            return 'critical'
        elif severity_score >= 80:
            return 'high'
        elif severity_score >= 40:
            return 'medium'
        else:
            return 'low'
    
    async def _cleanup(self):
        """Cleanup agent resources"""
        self.yara_rules = None
        if self.memory_parser:
            await self.memory_parser.cleanup()
        self.memory_parser = None
        self.pattern_matcher = None