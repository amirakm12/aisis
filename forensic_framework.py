#!/usr/bin/env python3
"""
Advanced Forensic Analysis Framework
A comprehensive digital forensics toolkit with AI-powered agents
"""

import asyncio
import json
import logging
import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forensic_analysis.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class Evidence:
    """Evidence data structure"""
    id: str
    timestamp: datetime
    source: str
    type: str
    data: Dict[str, Any]
    hash: str
    metadata: Dict[str, Any]
    chain_of_custody: List[str]

@dataclass
class Finding:
    """Analysis finding data structure"""
    id: str
    timestamp: datetime
    agent_id: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    category: str
    title: str
    description: str
    evidence_ids: List[str]
    confidence: float
    recommendations: List[str]

class ForensicAgent(ABC):
    """Abstract base class for forensic agents"""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")
        self.is_active = False
        
    @abstractmethod
    async def analyze(self, evidence: Evidence) -> List[Finding]:
        """Analyze evidence and return findings"""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent"""
        pass
    
    async def start(self):
        """Start the agent"""
        self.is_active = True
        await self.initialize()
        self.logger.info(f"Agent {self.name} started")
    
    async def stop(self):
        """Stop the agent"""
        self.is_active = False
        self.logger.info(f"Agent {self.name} stopped")

class EvidenceDatabase:
    """SQLite database for evidence and findings"""
    
    def __init__(self, db_path: str = "forensic_evidence.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Evidence table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                type TEXT NOT NULL,
                data TEXT NOT NULL,
                hash TEXT NOT NULL,
                metadata TEXT NOT NULL,
                chain_of_custody TEXT NOT NULL
            )
        ''')
        
        # Findings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS findings (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                evidence_ids TEXT NOT NULL,
                confidence REAL NOT NULL,
                recommendations TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_evidence(self, evidence: Evidence):
        """Store evidence in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            evidence.id,
            evidence.timestamp.isoformat(),
            evidence.source,
            evidence.type,
            json.dumps(evidence.data),
            evidence.hash,
            json.dumps(evidence.metadata),
            json.dumps(evidence.chain_of_custody)
        ))
        
        conn.commit()
        conn.close()
    
    def store_finding(self, finding: Finding):
        """Store finding in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO findings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            finding.id,
            finding.timestamp.isoformat(),
            finding.agent_id,
            finding.severity,
            finding.category,
            finding.title,
            finding.description,
            json.dumps(finding.evidence_ids),
            finding.confidence,
            json.dumps(finding.recommendations)
        ))
        
        conn.commit()
        conn.close()
    
    def get_evidence_by_type(self, evidence_type: str) -> List[Evidence]:
        """Retrieve evidence by type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM evidence WHERE type = ?', (evidence_type,))
        rows = cursor.fetchall()
        conn.close()
        
        evidence_list = []
        for row in rows:
            evidence = Evidence(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                source=row[2],
                type=row[3],
                data=json.loads(row[4]),
                hash=row[5],
                metadata=json.loads(row[6]),
                chain_of_custody=json.loads(row[7])
            )
            evidence_list.append(evidence)
        
        return evidence_list

class AdvancedForensicFramework:
    """Main forensic analysis framework"""
    
    def __init__(self):
        self.agents: Dict[str, ForensicAgent] = {}
        self.database = EvidenceDatabase()
        self.logger = logging.getLogger("ForensicFramework")
        self.analysis_queue = asyncio.Queue()
        self.is_running = False
        
    def register_agent(self, agent: ForensicAgent):
        """Register a forensic agent"""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.name}")
    
    async def start_framework(self):
        """Start the forensic framework"""
        self.is_running = True
        self.logger.info("Starting Forensic Framework")
        
        # Start all agents
        for agent in self.agents.values():
            await agent.start()
        
        # Start analysis worker
        asyncio.create_task(self.analysis_worker())
        
        self.logger.info("Forensic Framework started successfully")
    
    async def stop_framework(self):
        """Stop the forensic framework"""
        self.is_running = False
        self.logger.info("Stopping Forensic Framework")
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
        
        self.logger.info("Forensic Framework stopped")
    
    async def submit_evidence(self, evidence: Evidence):
        """Submit evidence for analysis"""
        # Store evidence in database
        self.database.store_evidence(evidence)
        
        # Add to analysis queue
        await self.analysis_queue.put(evidence)
        
        self.logger.info(f"Evidence submitted: {evidence.id}")
    
    async def analysis_worker(self):
        """Worker that processes evidence through agents"""
        while self.is_running:
            try:
                # Get evidence from queue
                evidence = await asyncio.wait_for(
                    self.analysis_queue.get(), timeout=1.0
                )
                
                # Run analysis through all agents
                tasks = []
                for agent in self.agents.values():
                    if agent.is_active:
                        tasks.append(agent.analyze(evidence))
                
                # Wait for all analyses to complete
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process findings
                    for result in results:
                        if isinstance(result, Exception):
                            self.logger.error(f"Agent analysis failed: {result}")
                        elif isinstance(result, list):
                            for finding in result:
                                self.database.store_finding(finding)
                                self.logger.info(f"Finding recorded: {finding.title}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Analysis worker error: {e}")
    
    def create_evidence(self, source: str, evidence_type: str, 
                       data: Dict[str, Any], metadata: Dict[str, Any] = None) -> Evidence:
        """Create evidence object with proper chain of custody"""
        evidence_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Calculate hash of data
        data_str = json.dumps(data, sort_keys=True)
        hash_value = hashlib.sha256(data_str.encode()).hexdigest()
        
        if metadata is None:
            metadata = {}
        
        chain_of_custody = [f"Created by {source} at {timestamp.isoformat()}"]
        
        return Evidence(
            id=evidence_id,
            timestamp=timestamp,
            source=source,
            type=evidence_type,
            data=data,
            hash=hash_value,
            metadata=metadata,
            chain_of_custody=chain_of_custody
        )
    
    async def get_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        conn = sqlite3.connect(self.database.db_path)
        cursor = conn.cursor()
        
        # Get findings summary
        cursor.execute('''
            SELECT severity, COUNT(*) as count 
            FROM findings 
            GROUP BY severity
        ''')
        severity_counts = dict(cursor.fetchall())
        
        # Get recent findings
        cursor.execute('''
            SELECT * FROM findings 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        recent_findings = cursor.fetchall()
        
        # Get evidence summary
        cursor.execute('''
            SELECT type, COUNT(*) as count 
            FROM evidence 
            GROUP BY type
        ''')
        evidence_counts = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_findings": sum(severity_counts.values()),
                "severity_breakdown": severity_counts,
                "evidence_types": evidence_counts,
                "active_agents": len([a for a in self.agents.values() if a.is_active])
            },
            "recent_findings": recent_findings[:5]  # Top 5 most recent
        }

# Export the main framework class
__all__ = ['AdvancedForensicFramework', 'ForensicAgent', 'Evidence', 'Finding']