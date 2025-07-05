"""
HuiHui Base Expert Class
========================

Base class for all HuiHui AI experts providing common functionality:
- Pydantic-first architecture with EOTS schema validation
- Standardized expert interface and communication protocols
- Performance tracking and learning integration
- Database connectivity and caching

Author: EOTS v2.5 AI Architecture Division
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# EOTS v2.5 Pydantic schemas - ALWAYS validate against these
from data_models import (
    HuiHuiExpertConfigV2_5,
    HuiHuiAnalysisRequestV2_5,
    HuiHuiAnalysisResponseV2_5,
    HuiHuiUsageRecordV2_5,
    HuiHuiPerformanceMetricsV2_5,
    HuiHuiExpertType
)

logger = logging.getLogger(__name__)

class BaseHuiHuiExpert(ABC):
    """
    Base class for all HuiHui AI experts.
    
    Provides standardized interface, performance tracking, and Pydantic-first validation.
    All expert implementations must inherit from this class.
    """
    
    def __init__(self, expert_config: HuiHuiExpertConfigV2_5, db_manager=None):
        """
        Initialize base expert with Pydantic-validated configuration.
        
        Args:
            expert_config: Validated HuiHui expert configuration
            db_manager: Database manager for storage and caching
        """
        self.logger = logger.getChild(f"{self.__class__.__name__}")
        self.config = expert_config
        self.db_manager = db_manager
        
        # Performance tracking
        self.usage_history: List[HuiHuiUsageRecordV2_5] = []
        self.performance_metrics: Optional[HuiHuiPerformanceMetricsV2_5] = None
        
        # Expert state
        self.is_initialized = False
        self.last_analysis_time: Optional[datetime] = None
        
        self.logger.info(f"🧠 Initializing {self.config.expert_name}")
    
    @abstractmethod
    async def analyze(self, request: HuiHuiAnalysisRequestV2_5) -> HuiHuiAnalysisResponseV2_5:
        """
        Perform expert analysis on provided data.
        
        Args:
            request: Validated analysis request with market data
            
        Returns:
            HuiHuiAnalysisResponseV2_5: Validated analysis response
        """
        pass
    
    @abstractmethod
    def get_specialization_keywords(self) -> List[str]:
        """Get keywords that define this expert's specialization."""
        pass
    
    @abstractmethod
    def validate_input_data(self, data: Dict[str, Any]) -> bool:
        """Validate that input data contains required fields for this expert."""
        pass
    
    def record_usage(self, request: HuiHuiAnalysisRequestV2_5, response: HuiHuiAnalysisResponseV2_5, 
                    processing_time_ms: float) -> None:
        """Record usage statistics for performance tracking."""
        try:
            usage_record = HuiHuiUsageRecordV2_5(
                expert_used=HuiHuiExpertType.ORCHESTRATOR,  # TODO: Replace with correct mapping if available
                symbol=request.symbol,
                processing_time=processing_time_ms,
                success=True,
                error_message=None,
                market_condition="normal"
            )
            
            self.usage_history.append(usage_record)
            
            # Store in database if available
            if self.db_manager:
                self._store_usage_record(usage_record)
                
        except Exception as e:
            self.logger.error(f"Failed to record usage: {e}")
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for the last N days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_usage = [
                record for record in self.usage_history 
                if record.timestamp >= cutoff_date
            ]
            
            if not recent_usage:
                return {"status": "no_data", "days": days}
            
            total_requests = len(recent_usage)
            avg_processing_time = sum(r.processing_time for r in recent_usage) / total_requests
            success_rate = sum(1 for r in recent_usage if r.success) / total_requests
            
            return {
                "status": "active",
                "days": days,
                "total_requests": total_requests,
                "avg_processing_time": avg_processing_time,
                "success_rate": success_rate,
                "requests_per_day": total_requests / days
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {"status": "error", "error": str(e)}
    
    def _store_usage_record(self, usage_record: HuiHuiUsageRecordV2_5) -> None:
        """Store usage record in database."""
        try:
            if self.db_manager and hasattr(self.db_manager, 'insert_record'):
                record_dict = usage_record.model_dump()
                self.db_manager.insert_record('huihui_usage_records', record_dict)
        except Exception as e:
            self.logger.error(f"Failed to store usage record: {e}")
    
    def initialize(self) -> bool:
        """Initialize the expert. Override in subclasses for custom initialization."""
        try:
            self.is_initialized = True
            self.logger.info(f"✅ {self.config.expert_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize expert: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the expert gracefully."""
        try:
            self.logger.info(f"🛑 Shutting down {self.config.expert_name}")
            self.is_initialized = False
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

class ExpertRegistry:
    """Registry for managing HuiHui experts."""
    
    def __init__(self):
        self.experts: Dict[str, BaseHuiHuiExpert] = {}
        self.logger = logger.getChild("ExpertRegistry")
    
    def register_expert(self, expert: BaseHuiHuiExpert) -> None:
        """Register an expert in the registry."""
        expert_name = expert.config.expert_name
        self.experts[expert_name] = expert
        self.logger.info(f"📝 Registered expert: {expert_name}")
    
    def get_expert(self, expert_id: str) -> Optional[BaseHuiHuiExpert]:
        """Get expert by ID."""
        return self.experts.get(expert_id)
    
    def get_all_experts(self) -> Dict[str, BaseHuiHuiExpert]:
        """Get all registered experts."""
        return self.experts.copy()
    
    def get_expert_status(self) -> Dict[str, Any]:
        """Get status of all registered experts."""
        return {
            expert_id: {
                "name": expert.config.expert_name,
                "initialized": expert.is_initialized,
                "last_analysis": expert.last_analysis_time
            }
            for expert_id, expert in self.experts.items()
        }

# Global expert registry
_expert_registry = ExpertRegistry()

def get_expert_registry() -> ExpertRegistry:
    """Get the global expert registry."""
    return _expert_registry

def register_expert(expert: BaseHuiHuiExpert) -> None:
    """Register an expert globally."""
    _expert_registry.register_expert(expert)

def get_expert(expert_id: str) -> Optional[BaseHuiHuiExpert]:
    """Get expert by ID from global registry."""
    return _expert_registry.get_expert(expert_id)
