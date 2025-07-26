"""
Advanced Forensic Engine
=======================

Core engine for coordinating forensic analysis operations with support for:
- Multi-threaded evidence processing
- Real-time analysis pipelines
- Distributed agent coordination
- Machine learning integration
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from ..agents.base_agent import BaseAgent
from ..utils.crypto_utils import CryptoAnalyzer
from ..utils.ml_utils import AnomalyDetector


class EvidenceType(Enum):
    """Types of digital evidence"""
    MEMORY_DUMP = "memory_dump"
    DISK_IMAGE = "disk_image"
    NETWORK_CAPTURE = "network_capture"
    LOG_FILE = "log_file"
    REGISTRY = "registry"
    BROWSER_ARTIFACTS = "browser_artifacts"
    MOBILE_BACKUP = "mobile_backup"
    CLOUD_DATA = "cloud_data"
    CRYPTOCURRENCY = "cryptocurrency"
    MALWARE_SAMPLE = "malware_sample"


class AnalysisStatus(Enum):
    """Status of forensic analysis"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPICIOUS = "suspicious"
    CRITICAL = "critical"


@dataclass
class ForensicCase:
    """Represents a forensic investigation case"""
    case_id: str
    name: str
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    evidence_items: List[str] = field(default_factory=list)
    findings: Dict[str, Any] = field(default_factory=dict)
    status: AnalysisStatus = AnalysisStatus.PENDING
    priority: int = 5  # 1-10 scale
    tags: List[str] = field(default_factory=list)
    chain_of_custody: List[Dict] = field(default_factory=list)


class ForensicEngine:
    """
    Advanced forensic analysis engine with multi-agent support
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.crypto_analyzer = CryptoAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        
        # Thread pools for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config['max_threads'])
        self.process_pool = ProcessPoolExecutor(max_workers=self.config['max_processes'])
        
        # Active cases and agents
        self.cases: Dict[str, ForensicCase] = {}
        self.active_agents: Dict[str, BaseAgent] = {}
        
        # Analysis plugins
        self.analysis_plugins = {}
        self._load_plugins()
        
        # Real-time monitoring
        self.monitoring_active = False
        self.event_queue = asyncio.Queue()
        
    def _default_config(self) -> Dict:
        """Default configuration for forensic engine"""
        return {
            'max_threads': 8,
            'max_processes': 4,
            'cache_size': 1024 * 1024 * 100,  # 100MB
            'analysis_timeout': 3600,  # 1 hour
            'enable_ml': True,
            'enable_gpu': True,
            'evidence_path': Path('./evidence'),
            'output_path': Path('./forensic_output'),
            'plugins_path': Path('./plugins')
        }
    
    def _load_plugins(self):
        """Load analysis plugins dynamically"""
        plugins_path = self.config['plugins_path']
        if plugins_path.exists():
            for plugin_file in plugins_path.glob('*.py'):
                try:
                    # Dynamic plugin loading logic here
                    pass
                except Exception as e:
                    self.logger.error(f"Failed to load plugin {plugin_file}: {e}")
    
    async def create_case(self, name: str, description: str, priority: int = 5) -> ForensicCase:
        """Create a new forensic case"""
        case_id = hashlib.sha256(f"{name}{datetime.now()}".encode()).hexdigest()[:16]
        case = ForensicCase(
            case_id=case_id,
            name=name,
            description=description,
            priority=priority
        )
        
        self.cases[case_id] = case
        self.logger.info(f"Created forensic case: {case_id}")
        
        # Initialize case directory structure
        case_path = self.config['output_path'] / case_id
        case_path.mkdir(parents=True, exist_ok=True)
        (case_path / 'evidence').mkdir(exist_ok=True)
        (case_path / 'analysis').mkdir(exist_ok=True)
        (case_path / 'reports').mkdir(exist_ok=True)
        
        return case
    
    async def add_evidence(self, case_id: str, evidence_path: Path, 
                          evidence_type: EvidenceType) -> Dict[str, Any]:
        """Add evidence to a case with integrity verification"""
        if case_id not in self.cases:
            raise ValueError(f"Case {case_id} not found")
        
        case = self.cases[case_id]
        
        # Calculate evidence hash
        evidence_hash = await self._calculate_hash(evidence_path)
        
        # Create evidence record
        evidence_record = {
            'id': hashlib.sha256(f"{evidence_path}{datetime.now()}".encode()).hexdigest()[:16],
            'path': str(evidence_path),
            'type': evidence_type.value,
            'hash': evidence_hash,
            'size': evidence_path.stat().st_size,
            'added_at': datetime.now().isoformat(),
            'verified': True
        }
        
        # Update chain of custody
        case.chain_of_custody.append({
            'action': 'evidence_added',
            'evidence_id': evidence_record['id'],
            'timestamp': datetime.now().isoformat(),
            'hash': evidence_hash
        })
        
        case.evidence_items.append(evidence_record['id'])
        
        # Store evidence metadata
        metadata_path = self.config['output_path'] / case_id / 'evidence' / f"{evidence_record['id']}.json"
        with open(metadata_path, 'w') as f:
            json.dump(evidence_record, f, indent=2)
        
        self.logger.info(f"Added evidence {evidence_record['id']} to case {case_id}")
        
        return evidence_record
    
    async def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    async def analyze_evidence(self, case_id: str, evidence_id: str, 
                             agents: List[str]) -> Dict[str, Any]:
        """Analyze evidence using specified agents"""
        if case_id not in self.cases:
            raise ValueError(f"Case {case_id} not found")
        
        case = self.cases[case_id]
        case.status = AnalysisStatus.IN_PROGRESS
        
        # Load evidence metadata
        metadata_path = self.config['output_path'] / case_id / 'evidence' / f"{evidence_id}.json"
        with open(metadata_path, 'r') as f:
            evidence_data = json.load(f)
        
        results = {
            'evidence_id': evidence_id,
            'analysis_start': datetime.now().isoformat(),
            'agent_results': {}
        }
        
        # Run analysis with each specified agent
        tasks = []
        for agent_name in agents:
            if agent_name in self.active_agents:
                agent = self.active_agents[agent_name]
                task = asyncio.create_task(
                    agent.analyze(evidence_data, case)
                )
                tasks.append((agent_name, task))
        
        # Collect results
        for agent_name, task in tasks:
            try:
                agent_result = await task
                results['agent_results'][agent_name] = agent_result
                
                # Check for critical findings
                if agent_result.get('severity') == 'critical':
                    case.status = AnalysisStatus.CRITICAL
                    await self._trigger_alert(case_id, agent_name, agent_result)
                    
            except Exception as e:
                self.logger.error(f"Agent {agent_name} failed: {e}")
                results['agent_results'][agent_name] = {'error': str(e)}
        
        results['analysis_end'] = datetime.now().isoformat()
        
        # Store analysis results
        analysis_path = self.config['output_path'] / case_id / 'analysis' / f"{evidence_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Update case findings
        case.findings[evidence_id] = results
        
        return results
    
    async def _trigger_alert(self, case_id: str, agent_name: str, findings: Dict):
        """Trigger alert for critical findings"""
        alert = {
            'case_id': case_id,
            'agent': agent_name,
            'findings': findings,
            'timestamp': datetime.now().isoformat(),
            'severity': 'critical'
        }
        
        # Add to event queue for real-time monitoring
        await self.event_queue.put(alert)
        
        self.logger.critical(f"CRITICAL FINDING in case {case_id} by {agent_name}")
    
    def register_agent(self, agent_name: str, agent: BaseAgent):
        """Register a forensic analysis agent"""
        self.active_agents[agent_name] = agent
        self.logger.info(f"Registered agent: {agent_name}")
    
    async def generate_timeline(self, case_id: str) -> Dict[str, Any]:
        """Generate forensic timeline from all evidence"""
        if case_id not in self.cases:
            raise ValueError(f"Case {case_id} not found")
        
        case = self.cases[case_id]
        timeline_events = []
        
        # Collect timeline events from all analyses
        for evidence_id, findings in case.findings.items():
            for agent_name, agent_results in findings.get('agent_results', {}).items():
                if 'timeline_events' in agent_results:
                    timeline_events.extend(agent_results['timeline_events'])
        
        # Sort events chronologically
        timeline_events.sort(key=lambda x: x.get('timestamp', ''))
        
        # Analyze timeline for patterns
        timeline_analysis = await self._analyze_timeline_patterns(timeline_events)
        
        timeline = {
            'case_id': case_id,
            'generated_at': datetime.now().isoformat(),
            'event_count': len(timeline_events),
            'events': timeline_events,
            'analysis': timeline_analysis
        }
        
        # Store timeline
        timeline_path = self.config['output_path'] / case_id / 'analysis' / 'timeline.json'
        with open(timeline_path, 'w') as f:
            json.dump(timeline, f, indent=2)
        
        return timeline
    
    async def _analyze_timeline_patterns(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze timeline for patterns and anomalies"""
        if not events:
            return {}
        
        # Extract timestamps
        timestamps = []
        for event in events:
            if 'timestamp' in event:
                try:
                    ts = datetime.fromisoformat(event['timestamp'])
                    timestamps.append(ts.timestamp())
                except:
                    pass
        
        if not timestamps:
            return {}
        
        # Calculate time deltas
        timestamps = np.array(sorted(timestamps))
        deltas = np.diff(timestamps)
        
        analysis = {
            'total_duration': timestamps[-1] - timestamps[0],
            'event_frequency': len(events) / (timestamps[-1] - timestamps[0]) if timestamps[-1] != timestamps[0] else 0,
            'burst_periods': [],
            'quiet_periods': []
        }
        
        # Detect burst periods (many events in short time)
        if len(deltas) > 0:
            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas)
            
            for i, delta in enumerate(deltas):
                if delta < mean_delta - 2 * std_delta:
                    analysis['burst_periods'].append({
                        'start_index': i,
                        'end_index': i + 1,
                        'duration': delta
                    })
                elif delta > mean_delta + 2 * std_delta:
                    analysis['quiet_periods'].append({
                        'start_index': i,
                        'end_index': i + 1,
                        'duration': delta
                    })
        
        return analysis
    
    async def export_report(self, case_id: str, format: str = 'html') -> Path:
        """Export comprehensive forensic report"""
        if case_id not in self.cases:
            raise ValueError(f"Case {case_id} not found")
        
        case = self.cases[case_id]
        
        # Prepare report data
        report_data = {
            'case': {
                'id': case.case_id,
                'name': case.name,
                'description': case.description,
                'created_at': case.created_at.isoformat(),
                'status': case.status.value,
                'priority': case.priority
            },
            'evidence_count': len(case.evidence_items),
            'findings': case.findings,
            'chain_of_custody': case.chain_of_custody,
            'generated_at': datetime.now().isoformat()
        }
        
        # Generate report based on format
        if format == 'html':
            report_path = await self._generate_html_report(case_id, report_data)
        elif format == 'pdf':
            report_path = await self._generate_pdf_report(case_id, report_data)
        elif format == 'json':
            report_path = self.config['output_path'] / case_id / 'reports' / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported report format: {format}")
        
        self.logger.info(f"Generated {format} report for case {case_id}: {report_path}")
        
        return report_path
    
    async def _generate_html_report(self, case_id: str, data: Dict) -> Path:
        """Generate HTML forensic report"""
        # HTML report generation logic
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forensic Report - Case {case_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .critical {{ background-color: #e74c3c; color: white; }}
                .warning {{ background-color: #f39c12; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Forensic Analysis Report</h1>
                <h2>Case: {data['case']['name']}</h2>
                <p>Generated: {data['generated_at']}</p>
            </div>
            
            <div class="section">
                <h3>Case Summary</h3>
                <p><strong>Case ID:</strong> {data['case']['id']}</p>
                <p><strong>Description:</strong> {data['case']['description']}</p>
                <p><strong>Status:</strong> {data['case']['status']}</p>
                <p><strong>Priority:</strong> {data['case']['priority']}/10</p>
                <p><strong>Evidence Items:</strong> {data['evidence_count']}</p>
            </div>
            
            <!-- Additional sections would be generated here -->
            
        </body>
        </html>
        """
        
        report_path = self.config['output_path'] / case_id / 'reports' / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    async def _generate_pdf_report(self, case_id: str, data: Dict) -> Path:
        """Generate PDF forensic report"""
        # PDF generation would require additional library like reportlab
        # Placeholder for PDF generation
        pass
    
    async def start_monitoring(self):
        """Start real-time monitoring for critical events"""
        self.monitoring_active = True
        
        async def monitor_loop():
            while self.monitoring_active:
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    # Process critical event
                    await self._handle_critical_event(event)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        asyncio.create_task(monitor_loop())
        self.logger.info("Started real-time monitoring")
    
    async def _handle_critical_event(self, event: Dict):
        """Handle critical forensic events"""
        # Implement critical event handling logic
        # Could include notifications, automated responses, etc.
        pass
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        self.logger.info("Stopped real-time monitoring")
    
    def shutdown(self):
        """Cleanup resources"""
        self.stop_monitoring()
        self.thread_pool.shutdown()
        self.process_pool.shutdown()
        self.logger.info("Forensic engine shutdown complete")