"""
Metrics Collector - Monitors system performance and usage
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..utils.config import MonitoringConfig


@dataclass
class QueryMetric:
    """Query performance metric"""
    query: str
    response_time: float
    sources_retrieved: int
    timestamp: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class IngestionMetric:
    """Document ingestion metric"""
    file_path: str
    processing_time: float
    chunks_created: int
    file_size: int
    timestamp: float
    success: bool
    error_message: Optional[str] = None


class MetricsCollector:
    """Collector for system metrics and performance monitoring"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.query_metrics: List[QueryMetric] = []
        self.ingestion_metrics: List[IngestionMetric] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.performance_stats: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize metrics collector"""
        if self.config.enable_metrics:
            print("Metrics collection enabled")
    
    async def record_query_metrics(
        self,
        query: str,
        response_time: float,
        sources_retrieved: int,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Record query performance metrics"""
        if not self.config.enable_metrics:
            return
        
        metric = QueryMetric(
            query=query,
            response_time=response_time,
            sources_retrieved=sources_retrieved,
            timestamp=time.time(),
            success=success,
            error_message=error_message
        )
        
        self.query_metrics.append(metric)
        
        # Update performance stats
        self._update_query_stats(metric)
    
    async def record_ingestion_metrics(
        self,
        file_path: str,
        processing_time: float,
        chunks_created: int,
        file_size: int = 0,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Record document ingestion metrics"""
        if not self.config.enable_metrics:
            return
        
        metric = IngestionMetric(
            file_path=file_path,
            processing_time=processing_time,
            chunks_created=chunks_created,
            file_size=file_size,
            timestamp=time.time(),
            success=success,
            error_message=error_message
        )
        
        self.ingestion_metrics.append(metric)
        
        # Update performance stats
        self._update_ingestion_stats(metric)
    
    async def record_error(self, error_type: str, error_message: str) -> None:
        """Record error occurrence"""
        if not self.config.enable_metrics:
            return
        
        self.error_counts[error_type] += 1
        
        # Log error for debugging
        if self.config.log_level == "DEBUG":
            print(f"Error recorded: {error_type} - {error_message}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        return {
            "query_metrics": len(self.query_metrics),
            "ingestion_metrics": len(self.ingestion_metrics),
            "error_counts": dict(self.error_counts),
            "performance_stats": self.performance_stats
        }
    
    async def get_query_metrics(self) -> List[QueryMetric]:
        """Get query metrics"""
        return self.query_metrics.copy()
    
    async def get_ingestion_metrics(self) -> List[IngestionMetric]:
        """Get ingestion metrics"""
        return self.ingestion_metrics.copy()
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.query_metrics and not self.ingestion_metrics:
            return {"status": "no_data"}
        
        summary = {
            "total_queries": len(self.query_metrics),
            "total_ingestions": len(self.ingestion_metrics),
            "successful_queries": len([m for m in self.query_metrics if m.success]),
            "successful_ingestions": len([m for m in self.ingestion_metrics if m.success]),
            "error_counts": dict(self.error_counts)
        }
        
        # Query performance stats
        if self.query_metrics:
            response_times = [m.response_time for m in self.query_metrics if m.success]
            if response_times:
                summary["query_performance"] = {
                    "avg_response_time": sum(response_times) / len(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "p95_response_time": self._calculate_percentile(response_times, 95),
                    "p99_response_time": self._calculate_percentile(response_times, 99)
                }
        
        # Ingestion performance stats
        if self.ingestion_metrics:
            processing_times = [m.processing_time for m in self.ingestion_metrics if m.success]
            if processing_times:
                summary["ingestion_performance"] = {
                    "avg_processing_time": sum(processing_times) / len(processing_times),
                    "min_processing_time": min(processing_times),
                    "max_processing_time": max(processing_times),
                    "p95_processing_time": self._calculate_percentile(processing_times, 95)
                }
        
        return summary
    
    def _update_query_stats(self, metric: QueryMetric) -> None:
        """Update query performance statistics"""
        if "query_stats" not in self.performance_stats:
            self.performance_stats["query_stats"] = {
                "total_queries": 0,
                "successful_queries": 0,
                "total_response_time": 0.0,
                "avg_sources_retrieved": 0.0
            }
        
        stats = self.performance_stats["query_stats"]
        stats["total_queries"] += 1
        
        if metric.success:
            stats["successful_queries"] += 1
            stats["total_response_time"] += metric.response_time
        
        # Update average sources retrieved
        total_sources = sum(m.sources_retrieved for m in self.query_metrics)
        stats["avg_sources_retrieved"] = total_sources / len(self.query_metrics)
    
    def _update_ingestion_stats(self, metric: IngestionMetric) -> None:
        """Update ingestion performance statistics"""
        if "ingestion_stats" not in self.performance_stats:
            self.performance_stats["ingestion_stats"] = {
                "total_ingestions": 0,
                "successful_ingestions": 0,
                "total_processing_time": 0.0,
                "total_chunks_created": 0,
                "total_file_size": 0
            }
        
        stats = self.performance_stats["ingestion_stats"]
        stats["total_ingestions"] += 1
        
        if metric.success:
            stats["successful_ingestions"] += 1
            stats["total_processing_time"] += metric.processing_time
            stats["total_chunks_created"] += metric.chunks_created
            stats["total_file_size"] += metric.file_size
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    async def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.query_metrics.clear()
        self.ingestion_metrics.clear()
        self.error_counts.clear()
        self.performance_stats.clear()
    
    async def export_metrics(self, file_path: str) -> None:
        """Export metrics to file"""
        import json
        
        export_data = {
            "query_metrics": [self._metric_to_dict(m) for m in self.query_metrics],
            "ingestion_metrics": [self._metric_to_dict(m) for m in self.ingestion_metrics],
            "error_counts": dict(self.error_counts),
            "performance_stats": self.performance_stats,
            "export_timestamp": time.time()
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _metric_to_dict(self, metric) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            "timestamp": metric.timestamp,
            "success": metric.success,
            "error_message": metric.error_message,
            **{k: v for k, v in metric.__dict__.items() if k not in ["timestamp", "success", "error_message"]}
        }
    
    async def close(self) -> None:
        """Close metrics collector"""
        # Export final metrics if enabled
        if self.config.enable_metrics and self.query_metrics:
            try:
                await self.export_metrics("final_metrics.json")
            except Exception:
                pass 