"""
Advanced Forensic System Usage Example
=====================================

Demonstrates how to use the advanced forensic analysis system
with multiple specialized agents
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime

from forensic_system.core import ForensicEngine
from forensic_system.agents import (
    MemoryForensicsAgent,
    NetworkForensicsAgent,
    DiskForensicsAgent,
    MalwareAnalysisAgent
)


async def main():
    """Main example demonstrating forensic system usage"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize forensic engine
    print("🔬 Initializing Advanced Forensic Analysis System...")
    engine = ForensicEngine()
    
    # Initialize specialized agents
    print("🤖 Initializing forensic agents...")
    
    memory_agent = MemoryForensicsAgent()
    await memory_agent.initialize()
    engine.register_agent("memory_forensics", memory_agent)
    
    network_agent = NetworkForensicsAgent()
    await network_agent.initialize()
    engine.register_agent("network_forensics", network_agent)
    
    disk_agent = DiskForensicsAgent()
    await disk_agent.initialize()
    engine.register_agent("disk_forensics", disk_agent)
    
    malware_agent = MalwareAnalysisAgent()
    await malware_agent.initialize()
    engine.register_agent("malware_analysis", malware_agent)
    
    print("✅ All agents initialized successfully\n")
    
    # Create a forensic case
    case = await engine.create_case(
        name="Suspected Data Breach Investigation",
        description="Investigation of potential data exfiltration and malware infection",
        priority=9
    )
    
    print(f"📁 Created case: {case.name} (ID: {case.case_id})\n")
    
    # Example 1: Memory Dump Analysis
    print("=" * 60)
    print("🧠 MEMORY FORENSICS ANALYSIS")
    print("=" * 60)
    
    # Add memory dump evidence
    memory_dump_path = Path("./evidence/memory.dmp")
    if memory_dump_path.exists():
        memory_evidence = await engine.add_evidence(
            case.case_id,
            memory_dump_path,
            EvidenceType.MEMORY_DUMP
        )
        
        print(f"Added memory dump: {memory_evidence['id']}")
        
        # Analyze with memory forensics agent
        memory_results = await engine.analyze_evidence(
            case.case_id,
            memory_evidence['id'],
            ['memory_forensics']
        )
        
        # Display key findings
        findings = memory_results['agent_results']['memory_forensics']['findings']
        
        print("\n📊 Memory Analysis Results:")
        print(f"- Total processes: {findings['process_analysis']['total_processes']}")
        print(f"- Suspicious processes: {len(findings['process_analysis']['suspicious_processes'])}")
        
        if findings['malware_detection']['malware_detected']:
            print("⚠️  MALWARE DETECTED IN MEMORY!")
            for match in findings['malware_detection']['yara_matches']:
                print(f"   - {match['rule']} ({', '.join(match['tags'])})")
        
        if findings['hidden_processes']['hidden_process_count'] > 0:
            print(f"🕵️  Found {findings['hidden_processes']['hidden_process_count']} hidden processes!")
        
        print(f"\n- Network connections: {len(findings['network_artifacts']['connections'])}")
        print(f"- Suspicious URLs: {len(findings['network_artifacts']['urls'])}")
        
    # Example 2: Network Capture Analysis
    print("\n" + "=" * 60)
    print("🌐 NETWORK FORENSICS ANALYSIS")
    print("=" * 60)
    
    pcap_path = Path("./evidence/capture.pcap")
    if pcap_path.exists():
        network_evidence = await engine.add_evidence(
            case.case_id,
            pcap_path,
            EvidenceType.NETWORK_CAPTURE
        )
        
        print(f"Added network capture: {network_evidence['id']}")
        
        # Analyze with network forensics agent
        network_results = await engine.analyze_evidence(
            case.case_id,
            network_evidence['id'],
            ['network_forensics']
        )
        
        findings = network_results['agent_results']['network_forensics']['findings']
        
        print("\n📊 Network Analysis Results:")
        print(f"- Total packets: {findings['packet_analysis']['total_packets']}")
        print(f"- Suspicious packets: {len(findings['packet_analysis']['suspicious_packets'])}")
        
        # Check for intrusions
        intrusions = findings['intrusion_detection']
        if intrusions['port_scans']:
            print("\n🚨 PORT SCANS DETECTED:")
            for scan in intrusions['port_scans'][:3]:
                print(f"   - From {scan['source_ip']} scanning {scan['port_count']} ports")
        
        if intrusions['brute_force_attempts']:
            print("\n🔨 BRUTE FORCE ATTEMPTS:")
            for attempt in intrusions['brute_force_attempts'][:3]:
                print(f"   - {attempt['source_ip']} -> {attempt['target']} ({attempt['attempt_count']} attempts)")
        
        # Check for C2 communication
        if findings['c2_detection']['beaconing']:
            print("\n🎯 POTENTIAL C2 BEACONING:")
            for beacon in findings['c2_detection']['beaconing'][:3]:
                print(f"   - {beacon['connection']} (interval: {beacon['average_interval']:.2f}s)")
    
    # Example 3: Disk Image Analysis
    print("\n" + "=" * 60)
    print("💾 DISK FORENSICS ANALYSIS")
    print("=" * 60)
    
    disk_image_path = Path("./evidence/disk.dd")
    if disk_image_path.exists():
        disk_evidence = await engine.add_evidence(
            case.case_id,
            disk_image_path,
            EvidenceType.DISK_IMAGE
        )
        
        print(f"Added disk image: {disk_evidence['id']}")
        
        # Analyze with disk forensics agent
        disk_results = await engine.analyze_evidence(
            case.case_id,
            disk_evidence['id'],
            ['disk_forensics']
        )
        
        findings = disk_results['agent_results']['disk_forensics']['findings']
        
        print("\n📊 Disk Analysis Results:")
        
        # File system analysis
        if 'filesystem_analysis' in findings:
            fs_analysis = findings['filesystem_analysis']
            print(f"- Partitions found: {len(fs_analysis['partitions'])}")
            
            for fs in fs_analysis['file_systems']:
                print(f"\n  File System: {fs['type']}")
                print(f"  - Total files: {fs['total_files']}")
                print(f"  - Total directories: {fs['total_directories']}")
                
                if fs['suspicious_paths']:
                    print(f"  - ⚠️  Suspicious files: {len(fs['suspicious_paths'])}")
                    for sus in fs['suspicious_paths'][:3]:
                        print(f"      • {sus['path']} (pattern: {sus['pattern']})")
        
        # Deleted files
        if 'deleted_files' in findings:
            deleted = findings['deleted_files']
            print(f"\n- Deleted files recovered: {deleted['recovered_count']}")
            if deleted['potentially_sensitive']:
                print(f"  - ⚠️  Potentially sensitive deleted files: {len(deleted['potentially_sensitive'])}")
        
        # Encryption detection
        if 'encryption_detection' in findings:
            encryption = findings['encryption_detection']
            if encryption['ransomware_indicators']:
                print("\n🔐 RANSOMWARE INDICATORS DETECTED!")
                for indicator in encryption['ransomware_indicators']:
                    print(f"   - {indicator}")
    
    # Example 4: Malware Analysis
    print("\n" + "=" * 60)
    print("🦠 MALWARE ANALYSIS")
    print("=" * 60)
    
    malware_path = Path("./evidence/suspicious.exe")
    if malware_path.exists():
        malware_evidence = await engine.add_evidence(
            case.case_id,
            malware_path,
            EvidenceType.MALWARE_SAMPLE
        )
        
        print(f"Added malware sample: {malware_evidence['id']}")
        
        # Analyze with malware analysis agent
        malware_results = await engine.analyze_evidence(
            case.case_id,
            malware_evidence['id'],
            ['malware_analysis']
        )
        
        findings = malware_results['agent_results']['malware_analysis']
        
        print("\n📊 Malware Analysis Results:")
        print(f"- Sample: {findings['sample_info']['filename']}")
        print(f"- Type: {findings['sample_info']['type']}")
        print(f"- SHA256: {findings['sample_info']['hashes']['sha256']}")
        
        # Static analysis
        if 'static_analysis' in findings['findings']:
            static = findings['findings']['static_analysis']
            
            if static['yara_matches']:
                print("\n🎯 YARA Matches:")
                for match in static['yara_matches']:
                    print(f"   - {match['rule']} ({', '.join(match['tags'])})")
            
            # Check sections for packing
            high_entropy_sections = [s for s in static.get('sections', []) 
                                   if s.get('entropy', 0) > 7.0]
            if high_entropy_sections:
                print(f"\n📦 Possible packing detected ({len(high_entropy_sections)} high-entropy sections)")
        
        # Anti-analysis techniques
        if 'anti_analysis_detection' in findings['findings']:
            anti = findings['findings']['anti_analysis_detection']
            techniques = []
            if anti['anti_debug']:
                techniques.append("Anti-Debug")
            if anti['anti_vm']:
                techniques.append("Anti-VM")
            if anti['packing']:
                techniques.append("Packing")
            
            if techniques:
                print(f"\n🛡️  Anti-Analysis Techniques: {', '.join(techniques)}")
        
        # Behavioral analysis
        if 'behavioral_analysis' in findings['findings']:
            behavioral = findings['findings']['behavioral_analysis']
            
            if behavioral['persistence_mechanisms']:
                print("\n🔒 Persistence Mechanisms:")
                for persist in behavioral['persistence_mechanisms']:
                    print(f"   - {persist['type']}: {persist.get('key', persist.get('path', 'N/A'))}")
        
        # Threat assessment
        print(f"\n⚡ Threat Score: {findings.get('threat_score', 0)}/100")
        print(f"🚨 Severity: {findings.get('severity', 'unknown').upper()}")
        
        # Recommendations
        if findings.get('recommendations'):
            print("\n💡 Recommendations:")
            for rec in findings['recommendations'][:5]:
                print(f"   • {rec}")
    
    # Generate forensic timeline
    print("\n" + "=" * 60)
    print("📅 FORENSIC TIMELINE GENERATION")
    print("=" * 60)
    
    timeline = await engine.generate_timeline(case.case_id)
    
    print(f"\nGenerated timeline with {timeline['event_count']} events")
    
    # Show recent critical events
    critical_events = [e for e in timeline['events'] if e.get('severity') in ['critical', 'high']]
    if critical_events:
        print("\n🚨 Critical Events:")
        for event in critical_events[:5]:
            print(f"   - [{event['type']}] {event['description']}")
    
    # Generate final report
    print("\n" + "=" * 60)
    print("📄 GENERATING FORENSIC REPORT")
    print("=" * 60)
    
    report_path = await engine.export_report(case.case_id, format='html')
    print(f"\n✅ Forensic report generated: {report_path}")
    
    # Case summary
    print("\n" + "=" * 60)
    print("📊 CASE SUMMARY")
    print("=" * 60)
    print(f"Case: {case.name}")
    print(f"Status: {case.status.value}")
    print(f"Evidence items: {len(case.evidence_items)}")
    print(f"Total findings: {len(case.findings)}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    engine.shutdown()
    
    print("\n✅ Forensic analysis complete!")


class EvidenceType:
    """Evidence type enumeration for the example"""
    MEMORY_DUMP = "memory_dump"
    NETWORK_CAPTURE = "network_capture"
    DISK_IMAGE = "disk_image"
    MALWARE_SAMPLE = "malware_sample"


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())