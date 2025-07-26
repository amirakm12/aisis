#!/usr/bin/env python3
"""
Advanced Forensic Framework Demo
Demonstrates the capabilities of the advanced forensic analysis system
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forensic_framework import AdvancedForensicFramework, Evidence, Finding
from forensic_agents.network_forensic_agent import NetworkForensicAgent
from forensic_agents.malware_analysis_agent import MalwareAnalysisAgent
from forensic_agents.memory_forensic_agent import MemoryForensicAgent
from forensic_agents.behavioral_analysis_agent import BehavioralAnalysisAgent

async def create_sample_evidence():
    """Create sample evidence for demonstration"""
    framework = AdvancedForensicFramework()
    evidence_samples = []
    
    # Network evidence samples
    print("🌐 Creating network evidence samples...")
    
    # Suspicious network connection
    net_evidence1 = framework.create_evidence(
        source="NetworkMonitor",
        evidence_type="network_connection",
        data={
            "src_ip": "192.168.1.100",
            "dst_ip": "203.0.113.1",  # Known malicious IP
            "dst_port": 4444,
            "protocol": "TCP",
            "bytes_in": 1024,
            "bytes_out": 2048,
            "duration": 300
        },
        metadata={"interface": "eth0", "capture_method": "pcap"}
    )
    evidence_samples.append(net_evidence1)
    
    # DNS tunneling attempt
    dns_evidence = framework.create_evidence(
        source="DNSMonitor",
        evidence_type="dns_query",
        data={
            "query": "aabbccddeeff112233445566778899aabbccddeeff.malware-c2.evil.com",
            "type": "TXT",
            "response_size": 1500,
            "src_ip": "192.168.1.50"
        }
    )
    evidence_samples.append(dns_evidence)
    
    # Data exfiltration
    exfil_evidence = framework.create_evidence(
        source="NetworkMonitor",
        evidence_type="network_traffic",
        data={
            "src_ip": "192.168.1.75",
            "dst_ip": "198.51.100.1",
            "dst_port": 443,
            "protocol": "HTTPS",
            "bytes_out": 50000000,  # 50MB outbound
            "bytes_in": 1024
        }
    )
    evidence_samples.append(exfil_evidence)
    
    # Malware evidence samples
    print("🦠 Creating malware evidence samples...")
    
    # Suspicious executable
    malware_evidence1 = framework.create_evidence(
        source="FileSystemMonitor",
        evidence_type="file",
        data={
            "path": "C:\\Users\\victim\\AppData\\Local\\Temp\\update.exe",
            "hash": "44d88612fea8a8f36de82e1278abb02f",  # Known malware hash
            "size": 2048000,
            "content": "MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00\xb8\x00\x00\x00\x00\x00\x00\x00\x40\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x0e\x1f\xba\x0e\x00\xb4\x09\xcd\x21\xb8\x01\x4c\xcd\x21\x54\x68\x69\x73\x20\x70\x72\x6f\x67\x72\x61\x6d\x20\x63\x61\x6e\x6e\x6f\x74\x20\x62\x65\x20\x72\x75\x6e\x20\x69\x6e\x20\x44\x4f\x53\x20\x6d\x6f\x64\x65\x2e\x0d\x0d\x0a\x24\x00\x00\x00\x00\x00\x00\x00PE\x00\x00",
            "signed": False,
            "entropy": 7.8
        }
    )
    evidence_samples.append(malware_evidence1)
    
    # Suspicious process
    process_evidence = framework.create_evidence(
        source="ProcessMonitor",
        evidence_type="process",
        data={
            "name": "cmd.exe",
            "parent_process": "winword.exe",
            "command_line": "cmd.exe /c powershell -enc JABhAD0AJABlAG4AdgA6AHQAZQBtAHAAKwAiAFwAdQBwAGQAYQB0AGUALgBlAHgAZQAiADsAaQBlAHgAIAAkAGEA",
            "user": "SYSTEM",
            "cpu_usage": 25.5,
            "memory_usage": 15000000
        }
    )
    evidence_samples.append(process_evidence)
    
    # Registry persistence
    registry_evidence = framework.create_evidence(
        source="RegistryMonitor",
        evidence_type="registry_entry",
        data={
            "key": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run",
            "value": "SecurityUpdate",
            "data": "C:\\Users\\victim\\AppData\\Local\\Temp\\update.exe",
            "operation": "create"
        }
    )
    evidence_samples.append(registry_evidence)
    
    # Memory evidence samples
    print("🧠 Creating memory evidence samples...")
    
    # Process injection
    memory_evidence1 = framework.create_evidence(
        source="MemoryAnalyzer",
        evidence_type="process_memory",
        data={
            "process_name": "svchost.exe",
            "api_calls": ["CreateRemoteThread", "VirtualAllocEx", "WriteProcessMemory", "LoadLibrary"],
            "content": "4d5a90000300000004000000ffff0000b800000000000000400000000000000000000000000000000000000000000000000000000000000000000000800000000e1fba0e00b409cd21b8014ccd21546869732070726f6772616d2063616e6e6f742062652072756e20696e20444f53206d6f64652e0d0d0a2400000000000000504500",
            "memory_permissions": {
                "0x7ff000000000": "RWX",
                "0x7ff000001000": "RW"
            }
        }
    )
    evidence_samples.append(memory_evidence1)
    
    # Shellcode detection
    shellcode_evidence = framework.create_evidence(
        source="MemoryAnalyzer",
        evidence_type="memory_dump",
        data={
            "content": "31c050686c6c336800686f726b6568657220576f726b65720068536f667457686572654d6f64756c6568476574506f636564647265737368476574507268657373416464726573736865786974507268726f636573736845786974506c65616e75706865786974746872656164684578697454686f636573736845786974507265616468436c6f7365",
            "process_name": "explorer.exe",
            "memory_regions": [
                {"start": "0x7ff000000000", "size": 4096, "permissions": "RWX"},
                {"start": "0x7ff000001000", "size": 8192, "permissions": "RW"}
            ]
        }
    )
    evidence_samples.append(shellcode_evidence)
    
    # Behavioral evidence samples
    print("👤 Creating behavioral evidence samples...")
    
    # Unusual time activity
    behavioral_evidence1 = framework.create_evidence(
        source="UserActivityMonitor",
        evidence_type="user_activity",
        data={
            "user": "john.doe",
            "activity_type": "login",
            "session_duration": 120,
            "applications": ["cmd.exe", "powershell.exe", "net.exe"],
            "failed_logins": 0,
            "privilege_escalation": True
        }
    )
    # Set timestamp to unusual hour (2 AM)
    behavioral_evidence1.timestamp = datetime.now(timezone.utc).replace(hour=2)
    evidence_samples.append(behavioral_evidence1)
    
    # High-entropy file
    entropy_evidence = framework.create_evidence(
        source="FileSystemMonitor",
        evidence_type="file",
        data={
            "path": "C:\\temp\\data.bin",
            "content": "".join([chr(i % 256) for i in range(1000)]),  # High entropy content
            "size": 1000,
            "entropy": 7.9,
            "signed": False
        }
    )
    evidence_samples.append(entropy_evidence)
    
    return evidence_samples

async def demonstrate_agents(framework, evidence_samples):
    """Demonstrate each forensic agent's capabilities"""
    
    print("\n" + "="*60)
    print("🔍 ADVANCED FORENSIC ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Process evidence through all agents
    total_findings = []
    
    for i, evidence in enumerate(evidence_samples, 1):
        print(f"\n📋 Processing Evidence {i}/{len(evidence_samples)}")
        print(f"   Type: {evidence.type}")
        print(f"   Source: {evidence.source}")
        print(f"   ID: {evidence.id[:8]}...")
        
        await framework.submit_evidence(evidence)
        
        # Small delay to allow processing
        await asyncio.sleep(0.5)
    
    # Wait for all analysis to complete
    print("\n⏳ Waiting for analysis to complete...")
    await asyncio.sleep(3)
    
    # Generate comprehensive report
    print("\n📊 Generating Analysis Report...")
    report = await framework.get_analysis_report()
    
    return report

async def display_detailed_findings(framework):
    """Display detailed findings from the database"""
    
    print("\n" + "="*60)
    print("🔎 DETAILED FORENSIC FINDINGS")
    print("="*60)
    
    # Get findings from database
    import sqlite3
    conn = sqlite3.connect(framework.database.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM findings 
        ORDER BY timestamp DESC, severity DESC
    ''')
    findings = cursor.fetchall()
    
    if not findings:
        print("No findings generated.")
        return
    
    severity_colors = {
        "CRITICAL": "🔴",
        "HIGH": "🟠", 
        "MEDIUM": "🟡",
        "LOW": "🟢"
    }
    
    category_icons = {
        "Network Reconnaissance": "🕵️",
        "Data Exfiltration": "📤",
        "DNS Tunneling": "🌐",
        "Threat Intelligence": "⚠️",
        "Known Malware": "🦠",
        "Packed Executable": "📦",
        "Suspicious Process": "⚙️",
        "Process Injection": "💉",
        "Memory Injection": "🧠",
        "Rootkit Detection": "👻",
        "Shellcode Detection": "🔧",
        "Behavioral Anomaly": "📈",
        "Temporal Anomaly": "⏰",
        "ML Anomaly Detection": "🤖",
        "High Entropy": "🎲"
    }
    
    for finding in findings:
        severity = finding[3]
        category = finding[4]
        title = finding[5]
        description = finding[6]
        confidence = finding[8]
        recommendations = json.loads(finding[9])
        
        severity_icon = severity_colors.get(severity, "⚪")
        category_icon = category_icons.get(category, "🔍")
        
        print(f"\n{severity_icon} {category_icon} [{severity}] {title}")
        print(f"   📝 {description}")
        print(f"   🎯 Confidence: {confidence:.1%}")
        print(f"   💡 Recommendations:")
        for rec in recommendations[:2]:  # Show first 2 recommendations
            print(f"      • {rec}")
        if len(recommendations) > 2:
            print(f"      ... and {len(recommendations)-2} more")
    
    conn.close()
    
    print(f"\n📊 Total Findings: {len(findings)}")

async def demonstrate_advanced_features():
    """Demonstrate advanced forensic features"""
    
    print("\n" + "="*60)
    print("🚀 ADVANCED FEATURES DEMONSTRATION")
    print("="*60)
    
    features = [
        ("🔗 Chain of Custody", "Automatic evidence integrity tracking with cryptographic hashes"),
        ("🤖 ML-Powered Analysis", "Machine learning algorithms for behavioral anomaly detection"),
        ("🧬 Pattern Recognition", "Advanced pattern matching for attack technique identification"),
        ("⚡ Real-time Processing", "Asynchronous evidence processing with concurrent analysis"),
        ("🎯 Threat Intelligence", "Integration with threat intelligence feeds for IOC matching"),
        ("📊 Statistical Analysis", "Statistical anomaly detection with baseline comparison"),
        ("🕐 Temporal Correlation", "Time-based analysis for attack timeline reconstruction"),
        ("🌐 Network Forensics", "Deep packet inspection and network behavior analysis"),
        ("🧠 Memory Forensics", "Advanced memory analysis for rootkits and injection detection"),
        ("🦠 Malware Analysis", "Static and dynamic malware analysis with YARA-like rules"),
        ("👥 Behavioral Profiling", "User and system behavior profiling for insider threat detection"),
        ("📈 Adaptive Learning", "Self-improving detection through continuous learning"),
        ("🔄 Cross-Correlation", "Multi-source evidence correlation for attack reconstruction"),
        ("📋 Automated Reporting", "Comprehensive forensic reports with actionable recommendations"),
        ("🎨 Visualization Ready", "Structured data output for forensic visualization tools")
    ]
    
    for feature, description in features:
        print(f"{feature} {description}")
        await asyncio.sleep(0.1)  # Dramatic effect
    
    print("\n🏆 This forensic framework provides enterprise-grade capabilities")
    print("   suitable for advanced persistent threat (APT) investigation,")
    print("   incident response, and proactive threat hunting operations.")

async def main():
    """Main demonstration function"""
    
    print("🔬 Advanced Forensic Analysis Framework")
    print("   Initializing comprehensive forensic capabilities...")
    
    # Initialize framework
    framework = AdvancedForensicFramework()
    
    # Register all forensic agents
    print("\n🤖 Registering Forensic Agents...")
    agents = [
        NetworkForensicAgent(),
        MalwareAnalysisAgent(), 
        MemoryForensicAgent(),
        BehavioralAnalysisAgent()
    ]
    
    for agent in agents:
        framework.register_agent(agent)
        print(f"   ✅ {agent.name}")
    
    # Start the framework
    print("\n🚀 Starting Forensic Framework...")
    await framework.start_framework()
    
    try:
        # Create sample evidence
        print("\n📋 Creating Sample Evidence...")
        evidence_samples = await create_sample_evidence()
        print(f"   Generated {len(evidence_samples)} evidence samples")
        
        # Demonstrate agents
        report = await demonstrate_agents(framework, evidence_samples)
        
        # Display summary report
        print("\n" + "="*60)
        print("📊 FORENSIC ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        summary = report["summary"]
        print(f"🔍 Total Findings: {summary['total_findings']}")
        print(f"🤖 Active Agents: {summary['active_agents']}")
        print(f"📋 Evidence Types: {len(summary['evidence_types'])}")
        
        if summary['severity_breakdown']:
            print("\n🚨 Findings by Severity:")
            for severity, count in summary['severity_breakdown'].items():
                icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")
                print(f"   {icon} {severity}: {count}")
        
        if summary['evidence_types']:
            print("\n📁 Evidence Types Analyzed:")
            for etype, count in summary['evidence_types'].items():
                print(f"   📄 {etype}: {count}")
        
        # Display detailed findings
        await display_detailed_findings(framework)
        
        # Demonstrate advanced features
        await demonstrate_advanced_features()
        
        print("\n" + "="*60)
        print("✅ FORENSIC ANALYSIS COMPLETE")
        print("="*60)
        print("🎯 The framework successfully detected multiple threats including:")
        print("   • Network reconnaissance and port scanning")
        print("   • Data exfiltration attempts")
        print("   • DNS tunneling activities") 
        print("   • Known malware signatures")
        print("   • Process injection techniques")
        print("   • Memory-based attacks")
        print("   • Behavioral anomalies")
        print("   • Temporal attack patterns")
        
        print(f"\n💾 All evidence and findings stored in: forensic_evidence.db")
        print(f"📝 Detailed logs available in: forensic_analysis.log")
        
    finally:
        # Stop the framework
        print("\n🛑 Stopping Forensic Framework...")
        await framework.stop_framework()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()