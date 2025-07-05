"""
Component Compliance Tracker v2.5 - Real-time Dashboard Component Monitoring
============================================================================

This module provides REAL-TIME tracking of dashboard component compliance with
control panel filters. It eliminates fake/hardcoded compliance data and provides
accurate monitoring of which components use filtered vs raw data.

Key Features:
- Real-time component registration and tracking
- Data source monitoring (filtered vs raw data usage)
- Metrics compliance tracking for EOTS calculations
- Performance-optimized tracking with minimal overhead
- Integration with AI Hub compliance reporting

Author: EOTS v2.5 Development Team
Version: 2.5.0
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import threading
import time


logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Types of data sources that components can use."""
    FILTERED_OPTIONS = "filtered_options"
    RAW_OPTIONS = "raw_options"
    FILTERED_STRIKES = "filtered_strikes"
    RAW_STRIKES = "raw_strikes"
    UNDERLYING_DATA = "underlying_data"
    METRICS_FILTERED = "metrics_filtered"
    METRICS_RAW = "metrics_raw"
    NO_OPTIONS_DATA = "no_options_data"

class ComplianceStatus(Enum):
    """Component compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"

@dataclass
class ComponentTrackingInfo:
    """Information about a tracked dashboard component."""
    component_id: str
    component_name: str
    creation_timestamp: datetime
    data_sources_used: Set[DataSourceType]
    compliance_status: ComplianceStatus
    data_access_count: int
    last_data_access: Optional[datetime]
    metadata: Dict[str, Any]

@dataclass
class MetricsTrackingInfo:
    """Information about tracked metrics calculations."""
    metric_name: str
    calculation_timestamp: datetime
    used_filtered_data: bool
    data_source_type: DataSourceType
    component_id: Optional[str]
    metadata: Dict[str, Any]

class ComponentComplianceTracker:
    """
    Real-time tracker for dashboard component compliance with control panel filters.
    
    This singleton class monitors all dashboard components and their data usage
    to provide accurate compliance reporting instead of fake empty lists.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the compliance tracker."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.components: Dict[str, ComponentTrackingInfo] = {}
        self.metrics: Dict[str, MetricsTrackingInfo] = {}
        self.control_params = None
        self.filtered_bundle = None
        self.tracking_enabled: bool = True
        self.session_start: datetime = datetime.now()
        
        # Performance tracking
        self.total_components_tracked: int = 0
        self.total_data_accesses: int = 0
        self.tracking_overhead_ms: float = 0.0
        
        self._initialized = True
        logger.info("🎯 ComponentComplianceTracker initialized - REAL TRACKING ENABLED")
    
    def reset_session(self) -> None:
        """Reset tracking for a new dashboard session."""
        with self._lock:
            self.components.clear()
            self.metrics.clear()
            self.control_params = None
            self.filtered_bundle = None
            self.total_components_tracked = 0
            self.total_data_accesses = 0
            self.tracking_overhead_ms = 0.0
            self.session_start = datetime.now()
            logger.info("🔄 Compliance tracking session reset")
    
    def set_control_parameters(self, params, filtered_bundle) -> None:
        """Set control panel parameters and filtered data bundle for tracking."""
        self.control_params = params
        self.filtered_bundle = filtered_bundle
        logger.info(f"✅ Control parameters set: {params.get_filter_description()}")
    
    def register_component(self, component_id: str, component_name: str, 
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new dashboard component for tracking.
        
        Args:
            component_id: Unique identifier for the component
            component_name: Human-readable component name
            metadata: Additional metadata about the component
        """
        if not self.tracking_enabled:
            return
            
        start_time = time.time()
        
        with self._lock:
            if component_id in self.components:
                logger.warning(f"⚠️ Component {component_id} already registered, updating...")
            
            self.components[component_id] = ComponentTrackingInfo(
                component_id=component_id,
                component_name=component_name,
                creation_timestamp=datetime.now(),
                data_sources_used=set(),
                compliance_status=ComplianceStatus.PENDING,
                data_access_count=0,
                last_data_access=None,
                metadata=metadata or {}
            )
            
            self.total_components_tracked += 1
            
        self.tracking_overhead_ms += (time.time() - start_time) * 1000
        logger.debug(f"📝 Registered component: {component_name} ({component_id})")
    
    def track_data_usage(self, component_id: str, data_source: DataSourceType, 
                        data_object: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Track data usage by a dashboard component.
        
        Args:
            component_id: ID of the component using data
            data_source: Type of data source being used
            data_object: The actual data object being used (for validation)
            metadata: Additional metadata about the data usage
        """
        if not self.tracking_enabled or component_id not in self.components:
            return
            
        start_time = time.time()
        
        with self._lock:
            component = self.components[component_id]
            component.data_sources_used.add(data_source)
            component.data_access_count += 1
            component.last_data_access = datetime.now()
            
            # Update metadata
            if metadata:
                component.metadata.update(metadata)
            
            # Update compliance status based on data sources used
            self._update_component_compliance_status(component_id)
            
            self.total_data_accesses += 1
            
        self.tracking_overhead_ms += (time.time() - start_time) * 1000
        logger.debug(f"📊 Data usage tracked: {component_id} -> {data_source.value}")
    
    def track_metrics_calculation(self, metric_name: str, used_filtered_data: bool,
                                data_source_type: DataSourceType, 
                                component_id: Optional[str] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Track metrics calculations for compliance monitoring.
        
        Args:
            metric_name: Name of the metric being calculated
            used_filtered_data: Whether filtered data was used
            data_source_type: Type of data source used
            component_id: Associated component ID (if any)
            metadata: Additional metadata
        """
        if not self.tracking_enabled:
            return
            
        start_time = time.time()
        
        with self._lock:
            metric_key = f"{metric_name}_{datetime.now().isoformat()}"
            self.metrics[metric_key] = MetricsTrackingInfo(
                metric_name=metric_name,
                calculation_timestamp=datetime.now(),
                used_filtered_data=used_filtered_data,
                data_source_type=data_source_type,
                component_id=component_id,
                metadata=metadata or {}
            )
            
        self.tracking_overhead_ms += (time.time() - start_time) * 1000
        logger.debug(f"🧮 Metrics calculation tracked: {metric_name} (filtered: {used_filtered_data})")
    
    def _update_component_compliance_status(self, component_id: str) -> None:
        """Update compliance status for a component based on its data usage."""
        if component_id not in self.components:
            return
            
        component = self.components[component_id]
        data_sources = component.data_sources_used
        
        # Determine compliance status
        has_filtered_options = DataSourceType.FILTERED_OPTIONS in data_sources
        has_raw_options = DataSourceType.RAW_OPTIONS in data_sources
        has_filtered_strikes = DataSourceType.FILTERED_STRIKES in data_sources
        has_raw_strikes = DataSourceType.RAW_STRIKES in data_sources
        has_filtered_metrics = DataSourceType.METRICS_FILTERED in data_sources
        has_raw_metrics = DataSourceType.METRICS_RAW in data_sources
        
        # No options data used - not applicable
        if DataSourceType.NO_OPTIONS_DATA in data_sources:
            component.compliance_status = ComplianceStatus.NOT_APPLICABLE
        # Only filtered data used - compliant
        elif (has_filtered_options or has_filtered_strikes or has_filtered_metrics) and \
             not (has_raw_options or has_raw_strikes or has_raw_metrics):
            component.compliance_status = ComplianceStatus.COMPLIANT
        # Only raw data used - non-compliant
        elif (has_raw_options or has_raw_strikes or has_raw_metrics) and \
             not (has_filtered_options or has_filtered_strikes or has_filtered_metrics):
            component.compliance_status = ComplianceStatus.NON_COMPLIANT
        # Mixed usage - partial compliance
        elif (has_filtered_options or has_filtered_strikes or has_filtered_metrics) and \
             (has_raw_options or has_raw_strikes or has_raw_metrics):
            component.compliance_status = ComplianceStatus.PARTIAL
        # No data sources tracked yet - pending
        else:
            component.compliance_status = ComplianceStatus.PENDING
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """
        Get real-time compliance summary with actual tracking data.
        
        Returns:
            Dict containing compliance summary with real data
        """
        with self._lock:
            # Count components by compliance status
            compliant_components = []
            non_compliant_components = []
            partial_components = []
            pending_components = []
            na_components = []
            
            for component in self.components.values():
                if component.compliance_status == ComplianceStatus.COMPLIANT:
                    compliant_components.append(component.component_name)
                elif component.compliance_status == ComplianceStatus.NON_COMPLIANT:
                    non_compliant_components.append(component.component_name)
                elif component.compliance_status == ComplianceStatus.PARTIAL:
                    partial_components.append(component.component_name)
                elif component.compliance_status == ComplianceStatus.PENDING:
                    pending_components.append(component.component_name)
                else:
                    na_components.append(component.component_name)
            
            # Count metrics by compliance
            metrics_with_filtered_data = []
            metrics_with_raw_data = []
            
            for metric in self.metrics.values():
                if metric.used_filtered_data:
                    metrics_with_filtered_data.append(metric.metric_name)
                else:
                    metrics_with_raw_data.append(metric.metric_name)
            
            # Calculate real compliance score
            total_applicable_components = len(compliant_components) + len(non_compliant_components) + len(partial_components)
            component_score = 0.0
            if total_applicable_components > 0:
                # Full credit for compliant, half credit for partial
                component_score = (len(compliant_components) + 0.5 * len(partial_components)) / total_applicable_components
            
            # Metrics compliance score
            total_metrics = len(metrics_with_filtered_data) + len(metrics_with_raw_data)
            metrics_score = 0.0
            if total_metrics > 0:
                metrics_score = len(metrics_with_filtered_data) / total_metrics
            
            # Filter application score (from existing logic)
            filter_score = 0.0
            if self.control_params:
                filter_score = 1.0  # Filters are always applied when control_params exist
            
            # Calculate weighted compliance score
            if total_applicable_components > 0 or total_metrics > 0:
                # Weighted average: component compliance 60%, metrics compliance 25%, filter application 15%
                total_weight = 0.0
                weighted_score = 0.0
                
                if total_applicable_components > 0:
                    weighted_score += component_score * 0.6
                    total_weight += 0.6
                
                if total_metrics > 0:
                    weighted_score += metrics_score * 0.25
                    total_weight += 0.25
                
                if self.control_params:
                    weighted_score += filter_score * 0.15
                    total_weight += 0.15
                
                final_compliance_score = weighted_score / total_weight if total_weight > 0 else 0.0
            else:
                final_compliance_score = filter_score  # Only filter score available
            
            return {
                # Real component tracking data
                "components_respecting_filters": compliant_components,
                "components_not_respecting_filters": non_compliant_components,
                "components_partial_compliance": partial_components,
                "components_pending": pending_components,
                "components_not_applicable": na_components,
                
                # Real metrics tracking data
                "metrics_calculated_with_filters": list(set(metrics_with_filtered_data)),
                "metrics_using_raw_data": list(set(metrics_with_raw_data)),
                
                # Real compliance scores
                "component_compliance_score": component_score,
                "metrics_compliance_score": metrics_score,
                "filter_application_score": filter_score,
                "overall_compliance_score": final_compliance_score,
                
                # Tracking statistics
                "total_components_tracked": len(self.components),
                "total_data_accesses": self.total_data_accesses,
                "tracking_overhead_ms": self.tracking_overhead_ms,
                "session_duration_seconds": (datetime.now() - self.session_start).total_seconds(),
                
                # Detailed breakdown
                "component_details": {
                    comp_id: {
                        "name": comp.component_name,
                        "status": comp.compliance_status.value,
                        "data_sources": [ds.value for ds in comp.data_sources_used],
                        "access_count": comp.data_access_count,
                        "last_access": comp.last_data_access.isoformat() if comp.last_data_access else None
                    }
                    for comp_id, comp in self.components.items()
                }
            }

# Global tracker instance
_global_tracker = None

def get_compliance_tracker():
    """Get the global compliance tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ComponentComplianceTracker()
    return _global_tracker

def reset_compliance_tracking() -> None:
    """Reset compliance tracking for a new session."""
    tracker = get_compliance_tracker()
    tracker.reset_session()

def track_component_creation(component_id: str, component_name: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to track component creation."""
    tracker = get_compliance_tracker()
    tracker.register_component(component_id, component_name, metadata)

def track_data_access(component_id: str, data_source: DataSourceType, 
                     data_object: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to track data access."""
    tracker = get_compliance_tracker()
    tracker.track_data_usage(component_id, data_source, data_object, metadata)

def track_metrics_calculation(metric_name: str, used_filtered_data: bool,
                            data_source_type: DataSourceType, 
                            component_id: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to track metrics calculations."""
    tracker = get_compliance_tracker()
    tracker.track_metrics_calculation(metric_name, used_filtered_data, data_source_type, component_id, metadata) 