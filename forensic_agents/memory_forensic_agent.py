#!/usr/bin/env python3
"""
Memory Forensic Agent
Advanced memory analysis for process injection, rootkits, and volatile artifacts
"""

import asyncio
import re
import json
import struct
from datetime import datetime, timezone
from typing import Dict, List, Any, Set, Tuple, Optional
import uuid
import binascii

from forensic_framework import ForensicAgent, Evidence, Finding

class MemoryForensicAgent(ForensicAgent):
    """Advanced memory forensics analysis agent"""
    
    def __init__(self):
        super().__init__("memory_forensic", "Memory Forensic Agent")
        self.injection_signatures: Dict[str, List[str]] = {}
        self.rootkit_patterns: List[Dict[str, Any]] = []
        self.suspicious_processes: Set[str] = set()
        self.memory_patterns: Dict[str, str] = {}
        self.shellcode_patterns: List[str] = []
        
    async def initialize(self) -> bool:
        """Initialize the memory forensic agent"""
        try:
            # Load injection detection signatures
            await self.load_injection_signatures()
            
            # Load rootkit detection patterns
            await self.load_rootkit_patterns()
            
            # Load memory analysis patterns
            await self.load_memory_patterns()
            
            self.logger.info("Memory Forensic Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Memory Forensic Agent: {e}")
            return False
    
    async def load_injection_signatures(self):
        """Load process injection detection signatures"""
        self.injection_signatures = {
            "dll_injection": [
                "CreateRemoteThread.*LoadLibrary",
                "WriteProcessMemory.*CreateRemoteThread",
                "SetWindowsHookEx.*DLL"
            ],
            "process_hollowing": [
                "CreateProcess.*SUSPENDED",
                "NtUnmapViewOfSection.*WriteProcessMemory",
                "ZwUnmapViewOfSection.*VirtualAllocEx"
            ],
            "atom_bombing": [
                "GlobalAddAtom.*NtQueueApcThread",
                "GlobalGetAtomName.*SetThreadContext"
            ],
            "thread_execution_hijacking": [
                "OpenThread.*SuspendThread.*SetThreadContext",
                "GetThreadContext.*SetThreadContext.*ResumeThread"
            ],
            "pe_injection": [
                "VirtualAllocEx.*WriteProcessMemory.*CreateRemoteThread",
                "NtCreateSection.*NtMapViewOfSection"
            ]
        }
        
        self.suspicious_processes.update([
            "svchost.exe", "explorer.exe", "winlogon.exe", "csrss.exe",
            "lsass.exe", "smss.exe", "wininit.exe", "services.exe"
        ])
        
        self.logger.info(f"Loaded {len(self.injection_signatures)} injection signature types")
    
    async def load_rootkit_patterns(self):
        """Load rootkit detection patterns"""
        self.rootkit_patterns = [
            {
                "name": "SSDT_Hook",
                "description": "System Service Descriptor Table hook",
                "pattern": r"nt!KiSystemService.*jmp.*[0-9a-fA-F]{8}",
                "severity": "CRITICAL"
            },
            {
                "name": "IDT_Hook", 
                "description": "Interrupt Descriptor Table hook",
                "pattern": r"idt\[.*\].*[0-9a-fA-F]{8}.*unknown",
                "severity": "CRITICAL"
            },
            {
                "name": "IRP_Hook",
                "description": "I/O Request Packet hook",
                "pattern": r"IRP.*MajorFunction.*unknown_module",
                "severity": "HIGH"
            },
            {
                "name": "Direct_Kernel_Object_Manipulation",
                "description": "Direct kernel object manipulation",
                "pattern": r"EPROCESS.*Flink.*Blink.*modified",
                "severity": "HIGH"
            },
            {
                "name": "Hidden_Process",
                "description": "Process hidden from standard enumeration",
                "pattern": r"PsActiveProcessHead.*missing_entry",
                "severity": "HIGH"
            }
        ]
        
        self.logger.info(f"Loaded {len(self.rootkit_patterns)} rootkit detection patterns")
    
    async def load_memory_patterns(self):
        """Load memory analysis patterns"""
        self.memory_patterns = {
            "shellcode_nop_sled": r"(\x90{10,})",  # NOP sled
            "shellcode_decoder": r"(\xeb\x0e|\xeb\x10|\xeb\x12)",  # Common decoder stubs
            "pe_header": r"MZ.{58}PE\x00\x00",  # PE header pattern
            "encrypted_payload": r"[\x00-\xFF]{100,}",  # Potential encrypted data
            "stack_pivot": r"(\x94|\x87|\x97)",  # Stack pivot instructions
            "rop_gadget": r"(\xc3|\xc2..)",  # RET instructions
        }
        
        # Common shellcode patterns
        self.shellcode_patterns = [
            r"\x31\xc0\x50\x68",  # XOR EAX, EAX; PUSH EAX; PUSH
            r"\x6a\x30\x58\x99",  # PUSH 30; POP EAX; CDQ
            r"\x64\x8b\x15\x30",  # MOV EDX, FS:[30] (PEB access)
            r"\xfc\x48\x83\xe4",  # CLD; DEC EAX; AND ESP (x64 shellcode)
            r"\x89\xe5\x31\xc0",  # MOV EBP, ESP; XOR EAX, EAX
        ]
        
        self.logger.info(f"Loaded {len(self.memory_patterns)} memory patterns")
    
    async def analyze(self, evidence: Evidence) -> List[Finding]:
        """Analyze memory evidence"""
        findings = []
        
        if evidence.type not in ['memory_dump', 'process_memory', 'heap_dump', 'stack_dump']:
            return findings
        
        try:
            # Process injection analysis
            injection_findings = await self.detect_process_injection(evidence)
            findings.extend(injection_findings)
            
            # Rootkit detection
            rootkit_findings = await self.detect_rootkits(evidence)
            findings.extend(rootkit_findings)
            
            # Shellcode detection
            shellcode_findings = await self.detect_shellcode(evidence)
            findings.extend(shellcode_findings)
            
            # Memory anomaly detection
            anomaly_findings = await self.detect_memory_anomalies(evidence)
            findings.extend(anomaly_findings)
            
            # Volatile artifact analysis
            artifact_findings = await self.analyze_volatile_artifacts(evidence)
            findings.extend(artifact_findings)
            
        except Exception as e:
            self.logger.error(f"Error analyzing memory evidence: {e}")
        
        return findings
    
    async def detect_process_injection(self, evidence: Evidence) -> List[Finding]:
        """Detect various process injection techniques"""
        findings = []
        
        data = evidence.data
        process_name = data.get('process_name', '')
        memory_content = data.get('content', '')
        api_calls = data.get('api_calls', [])
        
        # Check for injection patterns in API calls
        for injection_type, patterns in self.injection_signatures.items():
            for pattern in patterns:
                api_sequence = ' '.join(api_calls) if isinstance(api_calls, list) else str(api_calls)
                
                if re.search(pattern, api_sequence, re.IGNORECASE):
                    # Additional validation for legitimate processes
                    confidence = 0.75
                    if process_name.lower() in self.suspicious_processes:
                        confidence = 0.90  # Higher confidence for system processes
                    
                    finding = Finding(
                        id=str(uuid.uuid4()),
                        timestamp=datetime.now(timezone.utc),
                        agent_id=self.agent_id,
                        severity="HIGH",
                        category="Process Injection",
                        title=f"{injection_type.replace('_', ' ').title()} Detected",
                        description=f"Process {process_name} shows signs of {injection_type} "
                                   f"based on API call pattern: {pattern}",
                        evidence_ids=[evidence.id],
                        confidence=confidence,
                        recommendations=[
                            f"Investigate process {process_name} for malicious injection",
                            "Analyze injected code for malicious functionality",
                            "Check parent process and execution context",
                            "Review process memory for additional artifacts",
                            "Consider process isolation or termination"
                        ]
                    )
                    findings.append(finding)
        
        # Analyze memory for injection artifacts
        if memory_content:
            injection_artifacts = await self.analyze_injection_artifacts(evidence, memory_content)
            findings.extend(injection_artifacts)
        
        return findings
    
    async def analyze_injection_artifacts(self, evidence: Evidence, memory_content: str) -> List[Finding]:
        """Analyze memory content for injection artifacts"""
        findings = []
        
        try:
            # Convert to bytes if string
            if isinstance(memory_content, str):
                try:
                    memory_bytes = bytes.fromhex(memory_content)
                except ValueError:
                    memory_bytes = memory_content.encode('latin-1')
            else:
                memory_bytes = memory_content
            
            # Look for PE headers in unexpected locations
            pe_matches = []
            for match in re.finditer(rb'MZ.{58}PE\x00\x00', memory_bytes):
                pe_matches.append(match.start())
            
            if len(pe_matches) > 1:  # Multiple PE headers suggest injection
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity="HIGH",
                    category="Memory Injection",
                    title="Multiple PE Headers in Process Memory",
                    description=f"Found {len(pe_matches)} PE headers in process memory, "
                               f"indicating possible PE injection or process hollowing",
                    evidence_ids=[evidence.id],
                    confidence=0.80,
                    recommendations=[
                        "Analyze each PE header for malicious content",
                        "Check if additional PEs are legitimate DLLs",
                        "Review process loading behavior",
                        "Consider memory dumping for detailed analysis"
                    ]
                )
                findings.append(finding)
            
            # Look for reflective DLL loading patterns
            reflective_patterns = [
                rb'\x4c\x8b\xd1\xb8\x4c\x77\x26\x07',  # Reflective DLL hash
                rb'\x6a\x04\x68\x00\x10\x00\x00',      # VirtualAlloc pattern
                rb'\x8b\x45\x3c\x8b\x4c\x05\x78'       # PE parsing pattern
            ]
            
            for pattern in reflective_patterns:
                if pattern in memory_bytes:
                    finding = Finding(
                        id=str(uuid.uuid4()),
                        timestamp=datetime.now(timezone.utc),
                        agent_id=self.agent_id,
                        severity="HIGH",
                        category="Reflective Loading",
                        title="Reflective DLL Loading Pattern Detected",
                        description="Memory contains patterns consistent with reflective DLL loading",
                        evidence_ids=[evidence.id],
                        confidence=0.75,
                        recommendations=[
                            "Analyze reflective loading code for malicious intent",
                            "Extract and analyze loaded DLL",
                            "Check for additional injection techniques",
                            "Review process execution timeline"
                        ]
                    )
                    findings.append(finding)
                    break
            
        except Exception as e:
            self.logger.error(f"Error analyzing injection artifacts: {e}")
        
        return findings
    
    async def detect_rootkits(self, evidence: Evidence) -> List[Finding]:
        """Detect rootkit presence and techniques"""
        findings = []
        
        data = evidence.data
        memory_content = data.get('content', '')
        system_calls = data.get('system_calls', [])
        
        # Check for rootkit patterns
        for pattern_info in self.rootkit_patterns:
            pattern = pattern_info['pattern']
            
            # Check in memory content
            if memory_content and re.search(pattern, memory_content, re.IGNORECASE):
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity=pattern_info['severity'],
                    category="Rootkit Detection",
                    title=f"Rootkit Technique: {pattern_info['name']}",
                    description=f"{pattern_info['description']} detected in memory",
                    evidence_ids=[evidence.id],
                    confidence=0.85,
                    recommendations=[
                        "Perform comprehensive rootkit scan",
                        "Analyze kernel structures for modifications",
                        "Check system integrity and file signatures",
                        "Consider system rebuild if confirmed",
                        "Review security logs for initial compromise vector"
                    ]
                )
                findings.append(finding)
            
            # Check in system call patterns
            syscall_sequence = ' '.join(system_calls) if isinstance(system_calls, list) else str(system_calls)
            if syscall_sequence and re.search(pattern, syscall_sequence, re.IGNORECASE):
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity=pattern_info['severity'],
                    category="Rootkit Detection",
                    title=f"Rootkit System Call Pattern: {pattern_info['name']}",
                    description=f"{pattern_info['description']} detected in system call sequence",
                    evidence_ids=[evidence.id],
                    confidence=0.80,
                    recommendations=[
                        "Analyze system call hooking mechanisms",
                        "Check for kernel-level modifications",
                        "Review driver loading and unloading events",
                        "Perform offline analysis if possible"
                    ]
                )
                findings.append(finding)
        
        # Detect hidden processes
        hidden_process_findings = await self.detect_hidden_processes(evidence)
        findings.extend(hidden_process_findings)
        
        return findings
    
    async def detect_hidden_processes(self, evidence: Evidence) -> List[Finding]:
        """Detect processes hidden by rootkits"""
        findings = []
        
        data = evidence.data
        process_list = data.get('process_list', [])
        memory_content = data.get('content', '')
        
        if not process_list and not memory_content:
            return findings
        
        # Look for EPROCESS structures in memory
        if memory_content:
            try:
                if isinstance(memory_content, str):
                    memory_bytes = memory_content.encode('latin-1')
                else:
                    memory_bytes = memory_content
                
                # Simple heuristic for EPROCESS structure detection
                # Look for process name patterns in memory
                process_name_pattern = rb'[A-Za-z0-9_]{1,15}\.exe\x00'
                process_names_in_memory = re.findall(process_name_pattern, memory_bytes)
                
                # Compare with reported process list
                reported_processes = {proc.get('name', '').lower() for proc in process_list}
                memory_processes = {name.decode('latin-1').lower() for name in process_names_in_memory}
                
                hidden_processes = memory_processes - reported_processes
                
                if hidden_processes:
                    finding = Finding(
                        id=str(uuid.uuid4()),
                        timestamp=datetime.now(timezone.utc),
                        agent_id=self.agent_id,
                        severity="HIGH",
                        category="Hidden Process",
                        title="Hidden Processes Detected",
                        description=f"Found {len(hidden_processes)} processes in memory "
                                   f"not reported by standard enumeration: "
                                   f"{', '.join(list(hidden_processes)[:5])}",
                        evidence_ids=[evidence.id],
                        confidence=0.70,
                        recommendations=[
                            "Use alternative process enumeration methods",
                            "Analyze process hiding techniques",
                            "Check for rootkit presence",
                            "Review process creation events in logs",
                            "Consider memory forensics tools for detailed analysis"
                        ]
                    )
                    findings.append(finding)
            
            except Exception as e:
                self.logger.error(f"Error detecting hidden processes: {e}")
        
        return findings
    
    async def detect_shellcode(self, evidence: Evidence) -> List[Finding]:
        """Detect shellcode in memory"""
        findings = []
        
        data = evidence.data
        memory_content = data.get('content', '')
        
        if not memory_content:
            return findings
        
        try:
            # Convert to bytes
            if isinstance(memory_content, str):
                try:
                    memory_bytes = bytes.fromhex(memory_content)
                except ValueError:
                    memory_bytes = memory_content.encode('latin-1')
            else:
                memory_bytes = memory_content
            
            shellcode_indicators = 0
            detected_patterns = []
            
            # Check for shellcode patterns
            for pattern in self.shellcode_patterns:
                if isinstance(pattern, str):
                    pattern_bytes = bytes.fromhex(pattern.replace('\\x', ''))
                else:
                    pattern_bytes = pattern
                
                if pattern_bytes in memory_bytes:
                    shellcode_indicators += 1
                    detected_patterns.append(pattern)
            
            # Check for NOP sleds
            nop_pattern = rb'\x90{10,}'  # 10 or more NOPs
            nop_matches = re.findall(nop_pattern, memory_bytes)
            if nop_matches:
                shellcode_indicators += len(nop_matches)
                detected_patterns.append("NOP sled")
            
            # Check for common shellcode instructions
            common_instructions = [
                rb'\x31\xc0',  # XOR EAX, EAX
                rb'\x50',      # PUSH EAX
                rb'\x58',      # POP EAX
                rb'\x99',      # CDQ
                rb'\xc3',      # RET
            ]
            
            instruction_count = 0
            for instr in common_instructions:
                instruction_count += len(re.findall(instr, memory_bytes))
            
            if instruction_count > 20:  # Threshold for instruction density
                shellcode_indicators += 1
                detected_patterns.append("High instruction density")
            
            if shellcode_indicators >= 2:
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity="HIGH",
                    category="Shellcode Detection",
                    title="Potential Shellcode in Memory",
                    description=f"Memory contains {shellcode_indicators} shellcode indicators: "
                               f"{', '.join(detected_patterns[:3])}",
                    evidence_ids=[evidence.id],
                    confidence=0.75,
                    recommendations=[
                        "Analyze shellcode for malicious functionality",
                        "Determine shellcode injection method",
                        "Check for payload delivery mechanism",
                        "Review process execution context",
                        "Consider sandbox analysis of shellcode"
                    ]
                )
                findings.append(finding)
        
        except Exception as e:
            self.logger.error(f"Error detecting shellcode: {e}")
        
        return findings
    
    async def detect_memory_anomalies(self, evidence: Evidence) -> List[Finding]:
        """Detect memory layout and content anomalies"""
        findings = []
        
        data = evidence.data
        memory_regions = data.get('memory_regions', [])
        memory_permissions = data.get('memory_permissions', {})
        
        # Check for suspicious memory permissions
        for region, permissions in memory_permissions.items():
            if 'RWX' in permissions or ('RW' in permissions and 'X' in permissions):
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity="MEDIUM",
                    category="Memory Anomaly",
                    title="Suspicious Memory Permissions",
                    description=f"Memory region {region} has suspicious RWX permissions, "
                               f"which may indicate code injection or shellcode",
                    evidence_ids=[evidence.id],
                    confidence=0.65,
                    recommendations=[
                        f"Analyze content of memory region {region}",
                        "Check for injected code or shellcode",
                        "Review process for legitimate need for RWX memory",
                        "Monitor for code execution in this region"
                    ]
                )
                findings.append(finding)
        
        # Check for unusual memory allocation patterns
        if memory_regions:
            large_allocations = [r for r in memory_regions if r.get('size', 0) > 10*1024*1024]  # > 10MB
            
            if len(large_allocations) > 5:
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity="MEDIUM",
                    category="Memory Anomaly",
                    title="Unusual Memory Allocation Pattern",
                    description=f"Process has {len(large_allocations)} large memory allocations "
                               f"(>10MB), which may indicate malicious behavior",
                    evidence_ids=[evidence.id],
                    confidence=0.60,
                    recommendations=[
                        "Analyze content of large memory allocations",
                        "Check for data staging or payload storage",
                        "Review process memory usage patterns",
                        "Monitor for memory-based attacks"
                    ]
                )
                findings.append(finding)
        
        return findings
    
    async def analyze_volatile_artifacts(self, evidence: Evidence) -> List[Finding]:
        """Analyze volatile artifacts in memory"""
        findings = []
        
        data = evidence.data
        network_connections = data.get('network_connections', [])
        loaded_modules = data.get('loaded_modules', [])
        handles = data.get('handles', [])
        
        # Analyze network connections
        if network_connections:
            suspicious_connections = []
            for conn in network_connections:
                dst_ip = conn.get('dst_ip', '')
                dst_port = conn.get('dst_port', 0)
                
                # Check for suspicious ports
                suspicious_ports = {4444, 5555, 6666, 7777, 8888, 9999, 31337}
                if dst_port in suspicious_ports:
                    suspicious_connections.append(conn)
            
            if suspicious_connections:
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity="MEDIUM",
                    category="Suspicious Network Activity",
                    title="Suspicious Network Connections in Memory",
                    description=f"Found {len(suspicious_connections)} suspicious network "
                               f"connections using commonly malicious ports",
                    evidence_ids=[evidence.id],
                    confidence=0.70,
                    recommendations=[
                        "Investigate destination IPs and ports",
                        "Check for C2 communication patterns",
                        "Review network traffic for this process",
                        "Analyze connection establishment timeline"
                    ]
                )
                findings.append(finding)
        
        # Analyze loaded modules
        if loaded_modules:
            unsigned_modules = [mod for mod in loaded_modules if not mod.get('signed', True)]
            
            if len(unsigned_modules) > len(loaded_modules) * 0.3:  # >30% unsigned
                finding = Finding(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    agent_id=self.agent_id,
                    severity="MEDIUM",
                    category="Unsigned Modules",
                    title="High Percentage of Unsigned Modules",
                    description=f"Process has {len(unsigned_modules)} unsigned modules "
                               f"({len(unsigned_modules)/len(loaded_modules)*100:.1f}% of total)",
                    evidence_ids=[evidence.id],
                    confidence=0.65,
                    recommendations=[
                        "Analyze unsigned modules for malicious content",
                        "Check module origins and legitimacy",
                        "Review module loading sequence",
                        "Consider signature validation bypass techniques"
                    ]
                )
                findings.append(finding)
        
        return findings