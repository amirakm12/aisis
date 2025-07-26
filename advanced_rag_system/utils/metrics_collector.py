"""
Metrics Collector - Monitors system performance and usage
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from loguru import logger

from .config import MonitoringConfig

@dataclass
class QueryMetric:
    """Metric for a query operation"""
    timestamp: float
    query: str
    response_time: float
    confidence: float
    sources_count: int
    success: bool
    error: Optional[str] = None

@dataclass
class IngestionMetric:
    """Metric for a document ingestion operation"""
    timestamp: float
    document_path: str
    processing_time: float
    chunks_created: int
    embeddings_created: int
    success: bool
    error: Optional[str] = None

class MetricsCollector:
    """
    Metrics Collector for monitoring RAG system performance
    
    Tracks:
    - Query performance and success rates
    - Document ingestion metrics
    - System resource usage
    - Error rates and types
    - Response quality indicators
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.is_initialized = False
        
        # Metric storage
        self.query_metrics = deque(maxlen=10000)  # Keep last 10k queries
        self.ingestion_metrics = deque(maxlen=1000)  # Keep last 1k ingestions
        
        # Aggregated metrics
        self.query_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.response_times = deque(maxlen=1000)
        self.confidence_scores = deque(maxlen=1000)
        
        # System metrics
        self.system_start_time = time.time()
        self.total_queries = 0
        self.total_ingestions = 0
        self.total_errors = 0
        
        logger.info("Metrics Collector initialized")
    
    async def initialize(self):
        """Initialize the metrics collector"""
        if self.is_initialized:
            return
        
        try:
            # Start background tasks if needed
            if self.config.enable_metrics:
                # Could start background metric aggregation tasks here
                pass
            
            self.is_initialized = True
            logger.info("Metrics Collector initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize Metrics Collector: {str(e)}")
            raise
    
    async def record_query_metrics(
        self,
        query: str,
        response: Any,
        processing_time: float
    ):
        """Record metrics for a query operation"""
        if not self.config.enable_metrics:
            return
        
        try:
            # Extract metrics from response
            confidence = getattr(response, 'confidence', 0.0)
            sources_count = len(getattr(response, 'sources', []))
            success = True
            error = None
            
            # Create metric record
            metric = QueryMetric(
                timestamp=time.time(),
                query=query[:100],  # Truncate for privacy/storage
                response_time=processing_time,
                confidence=confidence,
                sources_count=sources_count,
                success=success,
                error=error
            )
            
            # Store metric
            self.query_metrics.append(metric)
            
            # Update aggregated metrics
            self.total_queries += 1
            self.response_times.append(processing_time)
            self.confidence_scores.append(confidence)
            self.query_counts[datetime.now().strftime("%Y-%m-%d %H")] += 1
            
            logger.debug(f"Recorded query metric: {processing_time:.2f}s, confidence: {confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording query metrics: {str(e)}")
    
    async def record_ingestion_metrics(
        self,
        document_path: str,
        result: Any,
        processing_time: float
    ):
        """Record metrics for a document ingestion operation"""
        if not self.config.enable_metrics:
            return
        
        try:
            # Extract metrics from result
            success = getattr(result, 'success', True)
            chunks_created = getattr(result, 'chunks_created', 0)
            embeddings_created = getattr(result, 'embeddings_created', 0)
            error = None if success else getattr(result, 'error', 'Unknown error')
            
            # Create metric record
            metric = IngestionMetric(
                timestamp=time.time(),
                document_path=document_path,
                processing_time=processing_time,
                chunks_created=chunks_created,
                embeddings_created=embeddings_created,
                success=success,
                error=error
            )
            
            # Store metric
            self.ingestion_metrics.append(metric)
            
            # Update aggregated metrics
            self.total_ingestions += 1
            if not success:
                self.total_errors += 1
                self.error_counts["ingestion"] += 1
            
            logger.debug(f"Recorded ingestion metric: {document_path}, {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error recording ingestion metrics: {str(e)}")
    
    async def record_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Record an error occurrence"""
        if not self.config.enable_metrics:
            return
        
        try:
            self.total_errors += 1
            self.error_counts[error_type] += 1
            
            logger.debug(f"Recorded error: {error_type} - {error_message}")
            
        except Exception as e:
            logger.error(f"Error recording error metric: {str(e)}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            current_time = time.time()
            uptime = current_time - self.system_start_time
            
            # Calculate averages
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0
            
            # Calculate rates
            queries_per_hour = self.total_queries / (uptime / 3600) if uptime > 0 else 0
            error_rate = self.total_errors / max(self.total_queries + self.total_ingestions, 1)
            
            # Recent metrics (last hour)
            one_hour_ago = current_time - 3600
            recent_queries = [m for m in self.query_metrics if m.timestamp > one_hour_ago]
            recent_ingestions = [m for m in self.ingestion_metrics if m.timestamp > one_hour_ago]
            
            metrics = {
                "system": {
                    "uptime_seconds": uptime,
                    "total_queries": self.total_queries,
                    "total_ingestions": self.total_ingestions,
                    "total_errors": self.total_errors,
                    "error_rate": error_rate
                },
                "performance": {
                    "avg_response_time": avg_response_time,
                    "avg_confidence": avg_confidence,
                    "queries_per_hour": queries_per_hour,
                    "p95_response_time": self._calculate_percentile(list(self.response_times), 95),
                    "p99_response_time": self._calculate_percentile(list(self.response_times), 99)
                },
                "recent_activity": {
                    "queries_last_hour": len(recent_queries),
                    "ingestions_last_hour": len(recent_ingestions),
                    "avg_response_time_last_hour": sum(m.response_time for m in recent_queries) / len(recent_queries) if recent_queries else 0,
                    "avg_confidence_last_hour": sum(m.confidence for m in recent_queries) / len(recent_queries) if recent_queries else 0
                },
                "errors": dict(self.error_counts),
                "query_distribution": dict(self.query_counts),
                "timestamp": current_time
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {"error": str(e)}
    
    async def get_query_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query metrics"""
        try:
            recent_metrics = list(self.query_metrics)[-limit:]
            return [asdict(metric) for metric in recent_metrics]
            
        except Exception as e:
            logger.error(f"Error getting query metrics: {str(e)}")
            return []
    
    async def get_ingestion_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent ingestion metrics"""
        try:
            recent_metrics = list(self.ingestion_metrics)[-limit:]
            return [asdict(metric) for metric in recent_metrics]
            
        except Exception as e:
            logger.error(f"Error getting ingestion metrics: {str(e)}")
            return []
    
    async def get_performance_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for a specific time window"""
        try:
            current_time = time.time()
            window_start = current_time - (time_window_hours * 3600)
            
            # Filter metrics by time window
            window_queries = [m for m in self.query_metrics if m.timestamp > window_start]
            window_ingestions = [m for m in self.ingestion_metrics if m.timestamp > window_start]
            
            if not window_queries and not window_ingestions:
                return {"message": f"No activity in the last {time_window_hours} hours"}
            
            # Calculate summary statistics
            successful_queries = [m for m in window_queries if m.success]
            failed_queries = [m for m in window_queries if not m.success]
            
            summary = {
                "time_window_hours": time_window_hours,
                "total_queries": len(window_queries),
                "successful_queries": len(successful_queries),
                "failed_queries": len(failed_queries),
                "success_rate": len(successful_queries) / len(window_queries) if window_queries else 0,
                "total_ingestions": len(window_ingestions),
                "avg_response_time": sum(m.response_time for m in window_queries) / len(window_queries) if window_queries else 0,
                "avg_confidence": sum(m.confidence for m in window_queries) / len(window_queries) if window_queries else 0,
                "avg_sources_per_query": sum(m.sources_count for m in window_queries) / len(window_queries) if window_queries else 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of a list of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        
        return sorted_values[index]
    
    async def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        try:
            self.query_metrics.clear()
            self.ingestion_metrics.clear()
            self.query_counts.clear()
            self.error_counts.clear()
            self.response_times.clear()
            self.confidence_scores.clear()
            
            self.system_start_time = time.time()
            self.total_queries = 0
            self.total_ingestions = 0
            self.total_errors = 0
            
            logger.info("Metrics reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting metrics: {str(e)}")
    
    async def export_metrics(self, format: str = "json") -> Dict[str, Any]:
        """Export metrics in specified format"""
        try:
            if format.lower() == "json":
                return await self.get_metrics()
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the metrics collector"""
        try:
            # Could save metrics to persistent storage here
            self.is_initialized = False
            logger.info("Metrics Collector closed")
            
        except Exception as e:
            logger.error(f"Error closing Metrics Collector: {str(e)}")
            raise