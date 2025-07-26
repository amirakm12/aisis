"""
Disk Forensics Agent
====================

Specialized agent for analyzing disk images, file systems,
and recovering deleted data
"""

import asyncio
import hashlib
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pytsk3
import pyewf
import magic
import exifread

from .base_agent import BaseAgent, AgentCapabilities
from ..utils.file_carver import FileCarver
from ..utils.registry_parser import RegistryParser


class DiskForensicsAgent(BaseAgent):
    """
    Advanced disk forensics agent with capabilities:
    - File system analysis (NTFS, FAT, EXT)
    - Deleted file recovery
    - File carving
    - Registry analysis
    - Timeline generation
    - Artifact extraction
    - Encryption detection
    """
    
    def __init__(self):
        super().__init__("DiskForensicsAgent", "2.0.0")
        self.file_carver = None
        self.registry_parser = None
        self.file_magic = None
        
    def _define_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            evidence_types=["disk_image", "dd_image", "e01_image", "vmdk"],
            analysis_types=[
                "filesystem_analysis",
                "deleted_file_recovery",
                "file_carving",
                "registry_analysis",
                "timeline_generation",
                "artifact_extraction",
                "encryption_detection"
            ],
            required_tools=["sleuthkit", "ewf-tools", "foremost"],
            parallel_capable=True,
            gpu_accelerated=False
        )
    
    async def _setup(self):
        """Initialize disk forensics tools"""
        self.file_carver = FileCarver()
        self.registry_parser = RegistryParser()
        self.file_magic = magic.Magic(mime=True)
    
    async def analyze(self, evidence_data: Dict[str, Any], 
                     case_context: Any) -> Dict[str, Any]:
        """Perform comprehensive disk analysis"""
        if not await self.validate_evidence(evidence_data):
            return {"error": "Invalid evidence type for disk analysis"}
        
        self.logger.info(f"Starting disk analysis for {evidence_data['id']}")
        
        image_path = Path(evidence_data['path'])
        results = {
            'agent': self.name,
            'version': self.version,
            'analysis_start': datetime.now().isoformat(),
            'evidence_id': evidence_data['id'],
            'findings': {}
        }
        
        # Open disk image
        img_info = self._open_disk_image(image_path)
        if not img_info:
            return {"error": "Failed to open disk image"}
        
        # Run analyses in parallel
        analysis_tasks = [
            self._analyze_filesystem(img_info),
            self._recover_deleted_files(img_info),
            self._carve_files(image_path),
            self._analyze_registry(img_info),
            self._extract_artifacts(img_info),
            self._detect_encryption(img_info),
            self._generate_disk_timeline(img_info)
        ]
        
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Process results
        analysis_names = [
            'filesystem_analysis',
            'deleted_files',
            'carved_files',
            'registry_analysis',
            'artifacts',
            'encryption_detection',
            'timeline'
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
    
    def _open_disk_image(self, image_path: Path) -> Optional[pytsk3.Img_Info]:
        """Open disk image file"""
        try:
            # Check if it's an E01 image
            if image_path.suffix.lower() == '.e01':
                filenames = pyewf.glob(str(image_path))
                ewf_handle = pyewf.handle()
                ewf_handle.open(filenames)
                return EWFImgInfo(ewf_handle)
            else:
                # Regular disk image
                return pytsk3.Img_Info(str(image_path))
        except Exception as e:
            self.logger.error(f"Failed to open disk image: {e}")
            return None
    
    async def _analyze_filesystem(self, img_info) -> Dict[str, Any]:
        """Analyze file system structure and metadata"""
        fs_analysis = {
            'partitions': [],
            'file_systems': [],
            'total_files': 0,
            'suspicious_files': [],
            'large_files': [],
            'recent_files': []
        }
        
        try:
            # Try to read partition table
            try:
                volume = pytsk3.Volume_Info(img_info)
                
                for part in volume:
                    if part.len > 0:
                        partition_info = {
                            'index': part.addr,
                            'type': part.desc.decode('utf-8'),
                            'start': part.start,
                            'length': part.len,
                            'size_mb': part.len * 512 / 1024 / 1024
                        }
                        fs_analysis['partitions'].append(partition_info)
                        
                        # Try to open file system
                        try:
                            fs = pytsk3.FS_Info(img_info, offset=part.start * 512)
                            fs_info = await self._analyze_fs_partition(fs)
                            fs_info['partition'] = part.addr
                            fs_analysis['file_systems'].append(fs_info)
                        except:
                            self.logger.warning(f"Could not open filesystem on partition {part.addr}")
                            
            except:
                # No partition table, try as single filesystem
                fs = pytsk3.FS_Info(img_info)
                fs_info = await self._analyze_fs_partition(fs)
                fs_analysis['file_systems'].append(fs_info)
                
        except Exception as e:
            self.logger.error(f"Filesystem analysis error: {e}")
        
        return fs_analysis
    
    async def _analyze_fs_partition(self, fs) -> Dict[str, Any]:
        """Analyze a single filesystem partition"""
        fs_info = {
            'type': fs.info.ftype_str.decode('utf-8'),
            'block_size': fs.info.block_size,
            'total_blocks': fs.info.block_count,
            'files': [],
            'directories': [],
            'suspicious_paths': []
        }
        
        # Suspicious file patterns
        suspicious_patterns = [
            'mimikatz', 'lazagne', 'procdump', 'pwdump',
            'nc.exe', 'netcat', 'psexec', 'metasploit',
            '.locked', '.encrypted', 'ransom', 'decrypt'
        ]
        
        # Walk filesystem
        file_count = 0
        dir_count = 0
        
        def walk_directory(directory, path="/"):
            nonlocal file_count, dir_count
            
            for entry in directory:
                if entry.info.name.name in [b".", b".."]:
                    continue
                    
                try:
                    name = entry.info.name.name.decode('utf-8')
                    full_path = os.path.join(path, name)
                    
                    # Check if it's a directory
                    if entry.info.meta and entry.info.meta.type == pytsk3.TSK_FS_META_TYPE_DIR:
                        dir_count += 1
                        fs_info['directories'].append({
                            'path': full_path,
                            'created': self._convert_timestamp(entry.info.meta.crtime),
                            'modified': self._convert_timestamp(entry.info.meta.mtime)
                        })
                        
                        # Recurse into subdirectory
                        if file_count < 10000:  # Limit recursion
                            sub_directory = entry.as_directory()
                            walk_directory(sub_directory, full_path)
                    else:
                        file_count += 1
                        
                        # Get file info
                        if entry.info.meta:
                            file_info = {
                                'path': full_path,
                                'size': entry.info.meta.size,
                                'created': self._convert_timestamp(entry.info.meta.crtime),
                                'modified': self._convert_timestamp(entry.info.meta.mtime),
                                'accessed': self._convert_timestamp(entry.info.meta.atime)
                            }
                            
                            # Check for suspicious files
                            name_lower = name.lower()
                            for pattern in suspicious_patterns:
                                if pattern in name_lower:
                                    fs_info['suspicious_paths'].append({
                                        'path': full_path,
                                        'pattern': pattern,
                                        'size': file_info['size']
                                    })
                                    break
                            
                            # Track large files
                            if file_info['size'] > 100 * 1024 * 1024:  # 100MB
                                fs_info['files'].append(file_info)
                                
                except Exception as e:
                    continue
        
        try:
            root_dir = fs.open_dir(path="/")
            walk_directory(root_dir)
        except:
            self.logger.error("Could not open root directory")
        
        fs_info['total_files'] = file_count
        fs_info['total_directories'] = dir_count
        
        return fs_info
    
    def _convert_timestamp(self, timestamp):
        """Convert filesystem timestamp to ISO format"""
        if timestamp:
            return datetime.fromtimestamp(timestamp).isoformat()
        return None
    
    async def _recover_deleted_files(self, img_info) -> Dict[str, Any]:
        """Recover deleted files from the disk"""
        recovery_results = {
            'recovered_count': 0,
            'recovered_files': [],
            'potentially_sensitive': []
        }
        
        sensitive_extensions = [
            '.doc', '.docx', '.xls', '.xlsx', '.pdf', '.txt',
            '.jpg', '.png', '.pst', '.ost', '.eml', '.msg'
        ]
        
        try:
            # Open filesystem
            fs = pytsk3.FS_Info(img_info)
            
            # Look for deleted files
            for file_object in self._find_deleted_files(fs):
                if file_object.info.meta:
                    # Get file name
                    name = file_object.info.name.name.decode('utf-8', errors='ignore')
                    
                    if name and file_object.info.meta.size > 0:
                        file_info = {
                            'name': name,
                            'size': file_object.info.meta.size,
                            'deleted_time': self._convert_timestamp(file_object.info.meta.mtime),
                            'recoverable': True
                        }
                        
                        recovery_results['recovered_files'].append(file_info)
                        recovery_results['recovered_count'] += 1
                        
                        # Check if potentially sensitive
                        ext = os.path.splitext(name)[1].lower()
                        if ext in sensitive_extensions:
                            recovery_results['potentially_sensitive'].append(file_info)
                        
                        # Limit results
                        if recovery_results['recovered_count'] > 1000:
                            break
                            
        except Exception as e:
            self.logger.error(f"Deleted file recovery error: {e}")
        
        return recovery_results
    
    def _find_deleted_files(self, fs, directory=None, path="/"):
        """Generator to find deleted files"""
        if directory is None:
            directory = fs.open_dir(path="/")
            
        for entry in directory:
            if entry.info.name.name in [b".", b".."]:
                continue
                
            # Check if file is deleted
            if entry.info.meta and entry.info.meta.flags & pytsk3.TSK_FS_META_FLAG_UNALLOC:
                yield entry
                
            # Recurse into directories
            if entry.info.meta and entry.info.meta.type == pytsk3.TSK_FS_META_TYPE_DIR:
                try:
                    sub_directory = entry.as_directory()
                    name = entry.info.name.name.decode('utf-8', errors='ignore')
                    sub_path = os.path.join(path, name)
                    
                    yield from self._find_deleted_files(fs, sub_directory, sub_path)
                except:
                    continue
    
    async def _carve_files(self, image_path: Path) -> Dict[str, Any]:
        """Carve files from unallocated space"""
        carving_results = {
            'carved_count': 0,
            'carved_files': [],
            'file_types': defaultdict(int)
        }
        
        # Use file carver to extract files
        carved = await self.file_carver.carve(image_path)
        
        for file_info in carved:
            carving_results['carved_files'].append({
                'type': file_info['type'],
                'size': file_info['size'],
                'offset': file_info['offset'],
                'header': file_info['header']
            })
            carving_results['file_types'][file_info['type']] += 1
            carving_results['carved_count'] += 1
        
        carving_results['file_types'] = dict(carving_results['file_types'])
        
        return carving_results
    
    async def _analyze_registry(self, img_info) -> Dict[str, Any]:
        """Analyze Windows registry hives"""
        registry_analysis = {
            'hives_found': [],
            'user_accounts': [],
            'installed_programs': [],
            'autostart_entries': [],
            'suspicious_entries': [],
            'usb_devices': []
        }
        
        # Registry hive paths
        hive_paths = [
            '/Windows/System32/config/SAM',
            '/Windows/System32/config/SYSTEM',
            '/Windows/System32/config/SOFTWARE',
            '/Windows/System32/config/SECURITY',
            '/Users/*/NTUSER.DAT'
        ]
        
        try:
            fs = pytsk3.FS_Info(img_info)
            
            for hive_path in hive_paths:
                try:
                    # Extract registry hive
                    hive_data = await self._extract_file(fs, hive_path)
                    if hive_data:
                        hive_name = os.path.basename(hive_path)
                        registry_analysis['hives_found'].append(hive_name)
                        
                        # Parse registry hive
                        if hive_name == 'SAM':
                            users = await self.registry_parser.parse_sam(hive_data)
                            registry_analysis['user_accounts'] = users
                        elif hive_name == 'SOFTWARE':
                            programs = await self.registry_parser.parse_software(hive_data)
                            registry_analysis['installed_programs'] = programs[:100]  # Limit
                        elif hive_name == 'SYSTEM':
                            usb = await self.registry_parser.parse_usb_devices(hive_data)
                            registry_analysis['usb_devices'] = usb
                            
                except Exception as e:
                    self.logger.debug(f"Could not process {hive_path}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Registry analysis error: {e}")
        
        return registry_analysis
    
    async def _extract_file(self, fs, path: str) -> Optional[bytes]:
        """Extract file content from filesystem"""
        try:
            file_object = fs.open(path)
            
            # Read file content
            offset = 0
            size = file_object.info.meta.size
            content = b""
            
            while offset < size:
                available_to_read = min(1024 * 1024, size - offset)  # 1MB chunks
                data = file_object.read_random(offset, available_to_read)
                if not data:
                    break
                content += data
                offset += len(data)
                
            return content
            
        except Exception as e:
            return None
    
    async def _extract_artifacts(self, img_info) -> Dict[str, Any]:
        """Extract various forensic artifacts"""
        artifacts = {
            'browser_history': [],
            'event_logs': [],
            'prefetch_files': [],
            'lnk_files': [],
            'thumbnails': []
        }
        
        # Artifact paths
        artifact_paths = {
            'browser_history': [
                '/Users/*/AppData/Local/Google/Chrome/User Data/Default/History',
                '/Users/*/AppData/Local/Mozilla/Firefox/Profiles/*/places.sqlite',
                '/Users/*/AppData/Local/Microsoft/Edge/User Data/Default/History'
            ],
            'event_logs': [
                '/Windows/System32/winevt/Logs/*.evtx'
            ],
            'prefetch_files': [
                '/Windows/Prefetch/*.pf'
            ],
            'lnk_files': [
                '/Users/*/AppData/Roaming/Microsoft/Windows/Recent/*.lnk',
                '/Users/*/Desktop/*.lnk'
            ]
        }
        
        try:
            fs = pytsk3.FS_Info(img_info)
            
            # Extract browser history
            for browser_path in artifact_paths['browser_history']:
                files = await self._find_files_by_pattern(fs, browser_path)
                for file_path in files[:10]:  # Limit
                    artifacts['browser_history'].append({
                        'path': file_path,
                        'browser': self._identify_browser(file_path)
                    })
            
            # Extract event logs
            for log_pattern in artifact_paths['event_logs']:
                files = await self._find_files_by_pattern(fs, log_pattern)
                for file_path in files[:20]:  # Limit
                    artifacts['event_logs'].append({
                        'path': file_path,
                        'type': os.path.basename(file_path).replace('.evtx', '')
                    })
                    
        except Exception as e:
            self.logger.error(f"Artifact extraction error: {e}")
        
        return artifacts
    
    def _identify_browser(self, path: str) -> str:
        """Identify browser from artifact path"""
        if 'Chrome' in path:
            return 'Chrome'
        elif 'Firefox' in path:
            return 'Firefox'
        elif 'Edge' in path:
            return 'Edge'
        else:
            return 'Unknown'
    
    async def _find_files_by_pattern(self, fs, pattern: str) -> List[str]:
        """Find files matching a pattern"""
        # This is a simplified implementation
        # Real implementation would handle wildcards properly
        found_files = []
        
        # Implementation would search filesystem for matching files
        
        return found_files
    
    async def _detect_encryption(self, img_info) -> Dict[str, Any]:
        """Detect encrypted volumes and files"""
        encryption_info = {
            'encrypted_volumes': [],
            'encrypted_files': [],
            'ransomware_indicators': [],
            'encryption_tools': []
        }
        
        # Known encryption signatures
        encryption_signatures = {
            b'LUKS\xba\xbe': 'LUKS',
            b'TrueCrypt': 'TrueCrypt',
            b'VeraCrypt': 'VeraCrypt',
            b'BitLocker': 'BitLocker'
        }
        
        # Ransomware indicators
        ransomware_extensions = [
            '.locked', '.encrypted', '.crypto', '.enc',
            '.locky', '.cerber', '.zepto', '.odin'
        ]
        
        ransomware_notes = [
            'DECRYPT_INSTRUCTION', 'HOW_TO_DECRYPT', 'README_TO_DECRYPT',
            'YOUR_FILES_ARE_ENCRYPTED', 'HELP_DECRYPT'
        ]
        
        try:
            # Check for encrypted volumes
            with open(img_info.get_filename(), 'rb') as f:
                # Check first MB for encryption signatures
                header = f.read(1024 * 1024)
                
                for signature, enc_type in encryption_signatures.items():
                    if signature in header:
                        encryption_info['encrypted_volumes'].append({
                            'type': enc_type,
                            'offset': header.find(signature)
                        })
            
            # Check for ransomware indicators in filesystem
            fs = pytsk3.FS_Info(img_info)
            
            # Look for ransomware file extensions and notes
            # (Implementation would walk filesystem looking for indicators)
            
        except Exception as e:
            self.logger.error(f"Encryption detection error: {e}")
        
        return encryption_info
    
    async def _generate_disk_timeline(self, img_info) -> Dict[str, Any]:
        """Generate timeline of disk activity"""
        timeline = {
            'events': [],
            'summary': {
                'total_events': 0,
                'date_range': {},
                'activity_peaks': []
            }
        }
        
        try:
            fs = pytsk3.FS_Info(img_info)
            
            # Collect timeline events
            events = []
            
            def collect_timeline(directory, path="/"):
                for entry in directory:
                    if entry.info.name.name in [b".", b".."]:
                        continue
                        
                    try:
                        if entry.info.meta:
                            name = entry.info.name.name.decode('utf-8', errors='ignore')
                            full_path = os.path.join(path, name)
                            
                            # Add creation event
                            if entry.info.meta.crtime:
                                events.append({
                                    'timestamp': self._convert_timestamp(entry.info.meta.crtime),
                                    'type': 'created',
                                    'path': full_path,
                                    'size': entry.info.meta.size
                                })
                            
                            # Add modification event
                            if entry.info.meta.mtime:
                                events.append({
                                    'timestamp': self._convert_timestamp(entry.info.meta.mtime),
                                    'type': 'modified',
                                    'path': full_path,
                                    'size': entry.info.meta.size
                                })
                            
                            # Add access event
                            if entry.info.meta.atime:
                                events.append({
                                    'timestamp': self._convert_timestamp(entry.info.meta.atime),
                                    'type': 'accessed',
                                    'path': full_path,
                                    'size': entry.info.meta.size
                                })
                                
                    except:
                        continue
            
            # Collect timeline from root
            root_dir = fs.open_dir(path="/")
            collect_timeline(root_dir)
            
            # Sort events by timestamp
            events.sort(key=lambda x: x['timestamp'] if x['timestamp'] else '')
            
            # Keep most recent 1000 events
            timeline['events'] = events[-1000:]
            timeline['summary']['total_events'] = len(events)
            
            if events:
                timeline['summary']['date_range'] = {
                    'start': events[0]['timestamp'],
                    'end': events[-1]['timestamp']
                }
                
        except Exception as e:
            self.logger.error(f"Timeline generation error: {e}")
        
        return timeline
    
    async def _generate_timeline_events(self, findings: Dict) -> List[Dict]:
        """Generate timeline events from disk findings"""
        events = []
        
        # Add suspicious file events
        if 'filesystem_analysis' in findings:
            for fs in findings['filesystem_analysis'].get('file_systems', []):
                for sus_file in fs.get('suspicious_paths', []):
                    events.append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'suspicious_file',
                        'description': f"Suspicious file found: {sus_file['path']}",
                        'severity': 'high',
                        'data': sus_file
                    })
        
        # Add encryption detection events
        if 'encryption_detection' in findings:
            for indicator in findings['encryption_detection'].get('ransomware_indicators', []):
                events.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'ransomware_indicator',
                    'description': f"Ransomware indicator detected: {indicator}",
                    'severity': 'critical',
                    'data': indicator
                })
        
        return sorted(events, key=lambda x: x['timestamp'])
    
    def _calculate_severity(self, findings: Dict) -> str:
        """Calculate overall severity based on findings"""
        severity_score = 0
        
        # Check for suspicious files
        if findings.get('filesystem_analysis', {}):
            for fs in findings['filesystem_analysis'].get('file_systems', []):
                suspicious_count = len(fs.get('suspicious_paths', []))
                if suspicious_count > 0:
                    severity_score += min(suspicious_count * 10, 50)
        
        # Check for ransomware
        if findings.get('encryption_detection', {}):
            if findings['encryption_detection'].get('ransomware_indicators'):
                severity_score += 100
        
        # Check for deleted sensitive files
        if findings.get('deleted_files', {}):
            sensitive_count = len(findings['deleted_files'].get('potentially_sensitive', []))
            if sensitive_count > 0:
                severity_score += min(sensitive_count * 5, 30)
        
        # Determine severity level
        if severity_score >= 100:
            return 'critical'
        elif severity_score >= 50:
            return 'high'
        elif severity_score >= 20:
            return 'medium'
        else:
            return 'low'
    
    async def _cleanup(self):
        """Cleanup agent resources"""
        if self.file_carver:
            await self.file_carver.cleanup()
        if self.registry_parser:
            await self.registry_parser.cleanup()


class EWFImgInfo(pytsk3.Img_Info):
    """Wrapper for EWF image files"""
    
    def __init__(self, ewf_handle):
        self._ewf_handle = ewf_handle
        super(EWFImgInfo, self).__init__(url="", type=pytsk3.TSK_IMG_TYPE_EXTERNAL)
        
    def close(self):
        self._ewf_handle.close()
        
    def read(self, offset, size):
        self._ewf_handle.seek(offset)
        return self._ewf_handle.read(size)
        
    def get_size(self):
        return self._ewf_handle.get_media_size()