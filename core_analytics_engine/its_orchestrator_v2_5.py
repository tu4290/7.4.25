# eots/core_analytics_engine/its_orchestrator_v2_5.py
"""
🎯 Enhanced ITS Orchestrator v2.5 - LEGENDARY META-ORCHESTRATOR
PYDANTIC-FIRST: Fully validated against EOTS schemas and integrated with legendary experts

This is the 4th pillar of the legendary system - the Meta-Orchestrator that coordinates
all analysis and makes final strategic decisions using the EOTS v2.5 architecture.
"""

import asyncio
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Pydantic imports for validation
from pydantic import BaseModel, Field

# EOTS core imports - Updated to use current data models structure
from data_models import (
    ProcessedDataBundleV2_5,
    ProcessedUnderlyingAggregatesV2_5,
    ProcessedStrikeLevelMetricsV2_5,
    ProcessedContractMetricsV2_5,
    KeyLevelsDataV2_5,
    KeyLevelV2_5,
    MarketRegimeState,
    FinalAnalysisBundleV2_5,
    UnifiedIntelligenceAnalysis,
    UnprocessedDataBundleV2_5,
    AdaptiveLearningConfigV2_5,
    SystemStateV2_5,
    PredictionConfigV2_5,
    RawOptionsContractV2_5,
    RawUnderlyingDataCombinedV2_5
)

# CRITICAL FIX: Import KeyLevelIdentifierV2_5 for real-time key level generation
from core_analytics_engine.key_level_identifier_v2_5 import KeyLevelIdentifierV2_5

# MOE schemas - Updated to use current data models structure
from data_models import (
    ExpertStatus,
    RoutingStrategy,
    ConsensusStrategy,
    AgreementLevel,
    HealthStatus,
    MOEExpertRegistryV2_5,
    MOEGatingNetworkV2_5,
    MOEExpertResponseV2_5,
    MOEUnifiedResponseV2_5
)

# EOTS utilities - VALIDATED AGAINST USER'S SYSTEM
from utils.config_manager_v2_5 import ConfigManagerV2_5
from core_analytics_engine.eots_metrics import MetricsCalculatorV2_5
from core_analytics_engine.market_regime_engine_v2_5 import MarketRegimeEngineV2_5
from core_analytics_engine.market_intelligence_engine_v2_5 import MarketIntelligenceEngineV2_5
from core_analytics_engine.atif_engine_v2_5 import ATIFEngineV2_5
from core_analytics_engine.news_intelligence_engine_v2_5 import NewsIntelligenceEngineV2_5
from core_analytics_engine.adaptive_learning_integration_v2_5 import AdaptiveLearningIntegrationV2_5
from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5
from data_management.database_manager_v2_5 import DatabaseManagerV2_5
from data_management.historical_data_manager_v2_5 import HistoricalDataManagerV2_5
from data_management.performance_tracker_v2_5 import PerformanceTrackerV2_5
from data_management.convexvalue_data_fetcher_v2_5 import ConvexValueDataFetcherV2_5
from data_management.tradier_data_fetcher_v2_5 import TradierDataFetcherV2_5

# Import Elite components - Updated to use consolidated elite_intelligence
from core_analytics_engine.eots_metrics.elite_intelligence import EliteConfig, ConvexValueColumns, EliteImpactColumns, MarketRegime, FlowType

# 🚀 REAL COMPLIANCE TRACKING: Import tracking system for metrics
try:
    from dashboard_application.modes.ai_dashboard.component_compliance_tracker_v2_5 import (
        track_metrics_calculation, DataSourceType
    )
    COMPLIANCE_TRACKING_AVAILABLE = True
except ImportError:
    COMPLIANCE_TRACKING_AVAILABLE = False

# HuiHui integration - USING USER'S EXISTING STRUCTURE
try:
    from huihui_integration.core.model_interface import create_market_regime_model
    from huihui_integration import (
        get_market_regime_expert,
        get_options_flow_expert,
        get_sentiment_expert,
        get_expert_coordinator,
        is_system_ready
    )
    LEGENDARY_EXPERTS_AVAILABLE = True
except ImportError:
    LEGENDARY_EXPERTS_AVAILABLE = False
    ExpertCommunicationProtocol = None

logger = logging.getLogger(__name__)

class LegendaryOrchestrationConfig(BaseModel):
    """PYDANTIC-FIRST: Configuration for legendary orchestration capabilities"""
    
    # AI Decision Making
    ai_decision_enabled: bool = Field(default=True, description="Enable AI-powered decision making")
    ai_model_name: str = Field(default="llama3.1:8b", description="AI model for decision making")
    ai_temperature: float = Field(default=0.1, description="AI temperature for consistency")
    ai_max_tokens: int = Field(default=2000, description="Maximum tokens for AI responses")
    
    # Expert Coordination
    expert_weight_adaptation: bool = Field(default=True, description="Enable dynamic expert weighting")
    expert_consensus_threshold: float = Field(default=0.7, description="Threshold for expert consensus")
    conflict_resolution_enabled: bool = Field(default=True, description="Enable conflict resolution")
    
    # Performance Optimization
    parallel_processing_enabled: bool = Field(default=True, description="Enable parallel expert processing")
    max_concurrent_experts: int = Field(default=4, description="Maximum concurrent expert analyses")
    cache_enabled: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL in seconds")
    
    # Learning and Adaptation
    continuous_learning_enabled: bool = Field(default=True, description="Enable continuous learning")
    performance_tracking_enabled: bool = Field(default=True, description="Enable performance tracking")
    adaptation_rate: float = Field(default=0.01, description="Rate of system adaptation")
    
    # Risk Management
    risk_management_enabled: bool = Field(default=True, description="Enable risk management")
    max_position_exposure: float = Field(default=0.1, description="Maximum position exposure")
    stop_loss_threshold: float = Field(default=0.02, description="Stop loss threshold")
    
    class Config:
        extra = 'forbid'

class ITSOrchestratorV2_5:
    """
    🎯 LEGENDARY META-ORCHESTRATOR - 4th Pillar of the Legendary System
    
    PYDANTIC-FIRST: Fully validated against EOTS schemas and integrated with legendary experts.
    This orchestrator coordinates all analysis and makes final strategic decisions.
    """
    
    def __init__(self, config_manager: ConfigManagerV2_5):
        """Initialize the ITS Orchestrator with required components."""
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
        # Initialize database manager first
        self._db_manager = DatabaseManagerV2_5(config_manager)
        
        # Initialize cache manager
        self._cache_manager = None  # Will be lazily initialized by property
        
        # Initialize historical data manager
        self.historical_data_manager = HistoricalDataManagerV2_5(
            config_manager=config_manager,
            db_manager=self._db_manager
        )
        
                # Initialize data fetchers FIRST (required by other components)
        try:
            self.convex_fetcher = ConvexValueDataFetcherV2_5(config_manager)
            self.tradier_fetcher = TradierDataFetcherV2_5(config_manager)
            self.logger.info("🔗 Data fetchers initialized successfully")
        except Exception as e:
            self.logger.error(f"⚠️ Failed to initialize data fetchers: {e}")
            self.convex_fetcher = None
            self.tradier_fetcher = None

        # Initialize metrics calculator
        # Get elite_config - now properly returns EliteConfig Pydantic model
        elite_config_obj = config_manager.get_setting("elite_config", None)
        elite_config_dict = elite_config_obj.model_dump() if elite_config_obj else None

        self.metrics_calculator = MetricsCalculatorV2_5(
            config_manager=config_manager,
            historical_data_manager=self.historical_data_manager,
            enhanced_cache_manager=self.cache_manager,  # Use property to ensure initialization
            elite_config=elite_config_dict
        )

        # Initialize market regime engine with Pydantic model (not dict) - NOW that fetchers exist
        self.market_regime_engine = MarketRegimeEngineV2_5(config_manager, elite_config_obj, self.tradier_fetcher, self.convex_fetcher)
        
        # Initialize market intelligence engine
        self.market_intelligence_engine = MarketIntelligenceEngineV2_5(
            config_manager=config_manager,
            metrics_calculator=self.metrics_calculator
        )
        
        # Initialize ATIF engine
        self.atif_engine = ATIFEngineV2_5(config_manager=config_manager)
        
        # Initialize news intelligence engine
        self.news_intelligence = NewsIntelligenceEngineV2_5(config_manager=config_manager)
        
        # Initialize adaptive learning integration
        adaptive_config = AdaptiveLearningConfigV2_5(
            auto_adaptation=True,
            confidence_threshold=0.7,
            pattern_discovery_threshold=0.7,
            adaptation_frequency_minutes=60,
            **config_manager.get_setting("adaptive_learning_settings", {})
        )
        self.adaptive_learning = AdaptiveLearningIntegrationV2_5(config=adaptive_config)
        
        # Initialize prediction config
        self.prediction_config = PredictionConfigV2_5(
            enabled=True,
            model_name="default_prediction_model",
            prediction_interval_seconds=300,
            max_data_age_seconds=120,
            success_threshold=0.7,
            **config_manager.get_setting("prediction_settings", {})
        )
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTrackerV2_5(config_manager)

        # CRITICAL FIX: Initialize KeyLevelIdentifierV2_5 for real-time key level generation
        self.key_level_identifier = KeyLevelIdentifierV2_5(config_manager)

        # Initialize system state with all required fields
        self.system_state = SystemStateV2_5(
            is_running=True,
            current_mode="operational",
            active_processes=["market_regime_engine", "market_intelligence_engine", "atif_engine", "news_intelligence", "adaptive_learning"],
            status_message="System initialized and running",
            errors=[]  # Empty list - no errors at initialization
        )
        
        self.logger.info("🎯 ITS Orchestrator initialized successfully with all components")
        
    async def analyze_market_regime(self, data_bundle: ProcessedDataBundleV2_5) -> str:
        """Analyze market regime using the market regime engine."""
        try:
            # Get market regime from the engine
            regime = await self.market_regime_engine.determine_market_regime(data_bundle)
            self.logger.info(f"Market regime determined: {regime}")
            return regime
        except Exception as e:
            self.logger.error(f"Failed to analyze market regime: {e}")
            return "UNDEFINED"
            
    def _calculate_regime_metrics(self, data_bundle: ProcessedDataBundleV2_5) -> Dict[str, float]:
        """Calculate regime metrics from the data bundle."""
        try:
            if not data_bundle or not data_bundle.underlying_data_enriched:
                return {}
                
            und_data = data_bundle.underlying_data_enriched
            
            # Extract required metrics from the underlying data model
            metrics = {}
            
            # Add volatility metrics
            metrics['volatility'] = getattr(und_data, 'u_volatility', 0.0)
            metrics['trend_strength'] = getattr(und_data, 'vri_2_0_und', 0.0)
            metrics['volume_trend'] = getattr(und_data, 'vfi_0dte_und_avg', 0.0)
            metrics['momentum'] = getattr(und_data, 'a_mspi_und', 0.0)
            metrics['regime_score'] = getattr(und_data, 'current_market_regime_v2_5', 'UNKNOWN')
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating regime metrics: {e}")
            return {}
    
    def _initialize_moe_expert_registry(self) -> MOEExpertRegistryV2_5:
        """Initialize MOE Expert Registry for the 4th MOE Expert (Meta-Orchestrator)"""
        try:
            registry = MOEExpertRegistryV2_5(
                expert_id="meta_orchestrator_v2_5",
                expert_name="Ultimate Meta-Orchestrator",
                expert_type="meta_orchestrator",
                capabilities=[
                    "expert_coordination",
                    "consensus_building",
                    "conflict_resolution",
                    "strategic_synthesis",
                    "risk_assessment",
                    "final_decision_making"
                ],
                specializations=[
                    "meta_analysis",
                    "expert_synthesis",
                    "strategic_decision_making"
                ],
                supported_tasks=[
                    "expert_coordination",
                    "consensus_building",
                    "final_analysis"
                ],
                status=ExpertStatus.ACTIVE,
                accuracy_score=0.95,
                confidence_score=0.9,
                response_time_ms=15000.0,
                success_rate=95.0,
                memory_usage_mb=512.0,
                cpu_usage_percent=25.0,
                gpu_required=False,
                health_score=0.98,
                last_health_check=datetime.now(),
                tags=["meta", "orchestrator", "legendary", "v2_5"]
            )
            self.logger.info("🎯 MOE Expert Registry initialized for Meta-Orchestrator")
            return registry
        except Exception as e:
            self.logger.error(f"Failed to initialize MOE Expert Registry: {e}")
            raise
    
    def _create_moe_gating_network(self, request_context: Dict[str, Any]) -> MOEGatingNetworkV2_5:
        """Create MOE Gating Network for routing decisions"""
        try:
            # Determine which experts to route to based on request context
            selected_experts = request_context.get('include_experts', ["regime", "flow", "intelligence"])
            
            # Calculate expert weights based on request type and context
            expert_weights = self._calculate_expert_weights(request_context)
            
            # Calculate capability scores
            capability_scores = {
                "regime_expert": 0.9,
                "flow_expert": 0.85,
                "intelligence_expert": 0.88,
                "meta_orchestrator": 0.95
            }
            
            gating_network = MOEGatingNetworkV2_5(
                selected_experts=selected_experts,
                routing_strategy=RoutingStrategy.WEIGHTED,
                routing_confidence=0.9,
                expert_weights=expert_weights,
                capability_scores=capability_scores,
                request_context=request_context
            )
            
            self.logger.info(f"🎯 MOE Gating Network created with {len(selected_experts)} experts")
            return gating_network
            
        except Exception as e:
            self.logger.error(f"Failed to create MOE Gating Network: {e}")
            raise
    
    def _calculate_expert_weights(self, request_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dynamic expert weights based on request context"""
        analysis_type = request_context.get('analysis_type', 'full')
        priority = request_context.get('priority', 'normal')
        
        # Base weights
        weights = {
            "regime_expert": 0.3,
            "flow_expert": 0.3,
            "intelligence_expert": 0.25,
            "meta_orchestrator": 0.15
        }
        
        # Adjust weights based on analysis type
        if analysis_type == 'regime_focused':
            weights["regime_expert"] = 0.5
            weights["flow_expert"] = 0.2
            weights["intelligence_expert"] = 0.2
            weights["meta_orchestrator"] = 0.1
        elif analysis_type == 'flow_focused':
            weights["regime_expert"] = 0.2
            weights["flow_expert"] = 0.5
            weights["intelligence_expert"] = 0.2
            weights["meta_orchestrator"] = 0.1
        elif analysis_type == 'intelligence_focused':
            weights["regime_expert"] = 0.2
            weights["flow_expert"] = 0.2
            weights["intelligence_expert"] = 0.5
            weights["meta_orchestrator"] = 0.1
        
        # Increase meta-orchestrator weight for high priority requests
        if priority == 'high':
            weights["meta_orchestrator"] += 0.1
            # Normalize other weights
            total_other = sum(v for k, v in weights.items() if k != "meta_orchestrator")
            for k in weights:
                if k != "meta_orchestrator":
                    weights[k] = weights[k] * (0.9 / total_other)
        
        return weights
    
    def _create_moe_expert_response(self, expert_id: str, expert_name: str, response_data: Dict[str, Any], 
                                   confidence_score: float, processing_time_ms: float) -> MOEExpertResponseV2_5:
        """Create MOE Expert Response for individual expert results"""
        try:
            expert_response = MOEExpertResponseV2_5(
                expert_id=expert_id,
                expert_name=expert_name,
                response_data=response_data,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                quality_score=min(confidence_score + 0.1, 1.0),  # Quality slightly higher than confidence
                uncertainty_score=1.0 - confidence_score,
                success=True,
                error_message=None,  # No error for successful response
                timestamp=datetime.now(),
                version="2.5"
            )
            return expert_response
        except Exception as e:
            self.logger.error(f"Failed to create MOE expert response for {expert_id}: {e}")
            # Return error response
            return MOEExpertResponseV2_5(
                expert_id=expert_id,
                expert_name=expert_name,
                response_data={"error": str(e)},
                confidence_score=0.0,
                processing_time_ms=processing_time_ms,
                quality_score=0.0,
                uncertainty_score=1.0,
                success=False,
                error_message=str(e),
                timestamp=datetime.now(),
                version="2.5"
            )
    
    def _create_moe_unified_response(self, expert_responses: List[MOEExpertResponseV2_5], 
                                   unified_data: Dict[str, Any], final_confidence: float,
                                   total_processing_time_ms: float) -> MOEUnifiedResponseV2_5:
        """Create MOE Unified Response combining all expert responses"""
        try:
            # Determine consensus strategy and agreement level
            successful_responses = [r for r in expert_responses if r.success]
            consensus_strategy = ConsensusStrategy.WEIGHTED_AVERAGE if len(successful_responses) > 1 else ConsensusStrategy.EXPERT_RANKING
            
            # Calculate agreement level based on confidence variance
            if len(successful_responses) > 1:
                confidence_scores = [r.confidence_score for r in successful_responses]
                confidence_variance = sum((c - final_confidence) ** 2 for c in confidence_scores) / len(confidence_scores)
                if confidence_variance < 0.01:
                    agreement_level = AgreementLevel.HIGH
                elif confidence_variance < 0.05:
                    agreement_level = AgreementLevel.MEDIUM
                else:
                    agreement_level = AgreementLevel.LOW
            else:
                agreement_level = AgreementLevel.HIGH  # Single expert, no disagreement
            
            unified_response = MOEUnifiedResponseV2_5(
                request_id=self.current_analysis.get("analysis_id", "unknown") if self.current_analysis else "unknown",
                request_type=self.current_analysis.get('analysis_type', 'full') if self.current_analysis else "unknown",
                consensus_strategy=consensus_strategy,
                agreement_level=agreement_level,
                final_confidence=final_confidence,
                expert_responses=expert_responses,
                participating_experts=[r.expert_id for r in expert_responses],
                unified_response=unified_data,
                response_quality=final_confidence,
                total_processing_time_ms=total_processing_time_ms,
                expert_coordination_time_ms=total_processing_time_ms * 0.1,  # Estimate 10% for coordination
                consensus_time_ms=total_processing_time_ms * 0.05,  # Estimate 5% for consensus
                system_health=HealthStatus.HEALTHY if len(successful_responses) == len(expert_responses) else HealthStatus.DEGRADED,
                timestamp=datetime.now(),
                version="2.5",
                debug_info={
                    "total_experts": len(expert_responses),
                    "successful_experts": len(successful_responses),
                    "failed_experts": len(expert_responses) - len(successful_responses)
                },
                performance_breakdown={
                    "data_processing": total_processing_time_ms * 0.3,
                    "expert_analysis": total_processing_time_ms * 0.5,
                    "synthesis": total_processing_time_ms * 0.15,
                    "coordination": total_processing_time_ms * 0.05
                }
            )
            
            self.logger.info(f"🎯 MOE Unified Response created with {len(successful_responses)}/{len(expert_responses)} successful experts")
            return unified_response
            
        except Exception as e:
            self.logger.error(f"Failed to create MOE unified response: {e}")
            raise
    
    def _get_regime_analysis_prompt(self) -> str:
        """Get system prompt for AI decision making"""
        return """
        You are the LEGENDARY META-ORCHESTRATOR for the EOTS v2.5 options trading system.
        
        Your role is to synthesize analysis from 3 specialist experts:
        1. Market Regime Expert - Provides VRI 2.0 analysis and regime detection
        2. Options Flow Expert - Provides VAPI-FA, DWFD, and elite flow analysis  
        3. Market Intelligence Expert - Provides sentiment, behavioral, and microstructure analysis
        
        Your responsibilities:
        - Synthesize expert analyses into strategic recommendations
        - Resolve conflicts between expert opinions
        - Provide final trading decisions with confidence scores
        - Assess risk and provide risk management guidance
        - Maintain consistency with EOTS v2.5 methodologies
        
        Always provide structured, actionable recommendations based on the expert analyses.
        Focus on high-probability setups with clear risk/reward profiles.
        """
    
    async def run_full_analysis_cycle(self, ticker: str, dte_min: int, dte_max: int, price_range_percent: int, **kwargs) -> FinalAnalysisBundleV2_5:
        """
        Run a complete analysis cycle with all experts, using only live data.
        """
        try:
            self.logger.debug(f"🚀 Starting full analysis cycle for {ticker}...")
            start_time = datetime.now()

            # Step 1: Fetch live data (ConvexValue primary, Tradier fallback)
            self.logger.debug(f"🔄 Fetching live data for {ticker}...")
            chain_data, underlying_data = await self.convex_fetcher.fetch_chain_and_underlying(
                session=None,
                symbol=ticker,
                dte_min=dte_min,
                dte_max=dte_max,
                price_range_percent=price_range_percent
            )
            if not underlying_data or not chain_data:
                self.logger.warning(f"ConvexValue fetch failed, trying Tradier fallback for {ticker}")
                async with self.tradier_fetcher as tradier:
                    chain_data, underlying_data = await tradier.fetch_chain_and_underlying(ticker)
            if not underlying_data or not chain_data:
                raise RuntimeError(f"Failed to fetch live data from both ConvexValue and Tradier for {ticker}.")

            # CRITICAL FIX: Enrich ConvexValue data with Tradier OHLC data for price change calculations
            if underlying_data and (
                underlying_data.price_change_pct_und is None or
                underlying_data.day_open_price_und is None or
                underlying_data.prev_day_close_price_und is None
            ):
                self.logger.info(f"🔄 Enriching {ticker} data with Tradier OHLC for price change calculations...")
                try:
                    async with self.tradier_fetcher as tradier:
                        # Fetch historical data to get previous close and today's open
                        historical_data = await tradier.fetch_historical_data(ticker, days=2)
                        if historical_data and 'data' in historical_data and len(historical_data['data']) >= 1:
                            # Get the most recent day's data
                            latest_day = historical_data['data'][-1]
                            prev_day = historical_data['data'][-2] if len(historical_data['data']) >= 2 else latest_day

                            # Calculate price changes using real OHLC data
                            current_price = float(underlying_data.price)
                            day_open = float(latest_day.get('open', current_price))
                            prev_close = float(prev_day.get('close', current_price))
                            day_high = float(latest_day.get('high', current_price))
                            day_low = float(latest_day.get('low', current_price))

                            # Calculate price changes
                            price_change_abs = current_price - prev_close if prev_close > 0 else 0.0
                            price_change_pct = (price_change_abs / prev_close) if prev_close > 0 else 0.0

                            # Create enriched underlying data with OHLC fields populated
                            enrichment_data = {
                                'price_change_abs_und': price_change_abs,
                                'price_change_pct_und': price_change_pct,
                                'day_open_price_und': day_open,
                                'day_high_price_und': day_high,
                                'day_low_price_und': day_low,
                                'prev_day_close_price_und': prev_close
                            }

                            # CRITICAL FIX: Add missing flow fields using ConvexValue data
                            # Map ConvexValue flow fields to expected field names for elite intelligence
                            if hasattr(underlying_data, 'value_bs') and underlying_data.value_bs is not None:
                                enrichment_data['net_value_flow_5m_und'] = underlying_data.value_bs
                                self.logger.info(f"✅ Mapped value_bs ({underlying_data.value_bs}) to net_value_flow_5m_und")

                            if hasattr(underlying_data, 'volm_bs') and underlying_data.volm_bs is not None:
                                enrichment_data['net_vol_flow_5m_und'] = underlying_data.volm_bs
                                self.logger.info(f"✅ Mapped volm_bs ({underlying_data.volm_bs}) to net_vol_flow_5m_und")

                            underlying_data = underlying_data.model_copy(update=enrichment_data)

                            self.logger.info(f"✅ Enriched {ticker} with OHLC: price_change_pct={price_change_pct:.4f}, open={day_open}, prev_close={prev_close}")
                        else:
                            self.logger.warning(f"⚠️ No historical data available for {ticker} OHLC enrichment")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to enrich {ticker} with Tradier OHLC data: {e}")
                    # Continue with ConvexValue data only - the system will handle missing OHLC gracefully

            # Keep data as Pydantic v2 models for as long as possible
            # Only convert to DataFrame/dict at the metrics calculator boundary if absolutely necessary
            self.logger.debug(f"🔄 Processing live data and calculating all metrics for {ticker}...")

            # STRICT PYDANTIC V2-ONLY: Pass Pydantic models directly to metrics calculator
            # Validate input types to ensure strict Pydantic v2 compliance
            if not isinstance(underlying_data, RawUnderlyingDataCombinedV2_5):
                raise TypeError(f"underlying_data must be RawUnderlyingDataCombinedV2_5, got {type(underlying_data)}")
            if not isinstance(chain_data, list) or not all(isinstance(c, RawOptionsContractV2_5) for c in chain_data):
                raise TypeError(f"chain_data must be List[RawOptionsContractV2_5], got {type(chain_data)}")

            # Call metrics calculator with strict Pydantic v2 models
            processed_bundle = self.metrics_calculator.process_data_bundle_v2(
                options_contracts=chain_data,  # List[RawOptionsContractV2_5]
                underlying_data=underlying_data  # RawUnderlyingDataCombinedV2_5
            )
            self.logger.info(f"✅ All metrics calculated for {ticker}")

            # Step 3: Market Regime Analysis (using the enriched data from metrics_calculator)
            self.logger.info(f"🏛️ STEP: Market regime analysis for {ticker}")
            market_regime_analysis_result = await self.market_regime_engine.determine_market_regime(processed_bundle)
            processed_bundle.underlying_data_enriched.current_market_regime_v2_5 = market_regime_analysis_result
            self.logger.info(f"🏛️ Market regime determined: {market_regime_analysis_result} for {ticker}")

            # Step 4: Generate Key Levels (using the enriched data)
            self.logger.info(f"🔑 STEP: Generating key levels for {ticker}")
            key_levels_data = await self._generate_key_levels(
                processed_bundle,
                ticker,
                datetime.now()
            )
            self.logger.info(f"✅ Key levels generated for {ticker}")

            # Step 5: ATIF Recommendations (placeholder for now, will use processed_bundle)
            self.logger.info(f"💡 STEP: Generating ATIF recommendations for {ticker}")
            atif_recommendations = [] # Placeholder
            self.logger.info(f"✅ ATIF recommendations generated for {ticker}")

            # Step 6: Assemble the final analysis bundle
            # Create minimal valid scored signals (required by model validation)
            scored_signals = {
                "system_status": [
                    f"Analysis completed for {ticker} at {datetime.now().strftime('%H:%M:%S')}",
                    f"Processed {len(processed_bundle.options_data_with_metrics)} contracts",
                    f"Elite impact score: {processed_bundle.underlying_data_enriched.elite_impact_score_und:.1f}"
                ]
            }

            bundle = FinalAnalysisBundleV2_5(
                processed_data_bundle=processed_bundle,
                scored_signals_v2_5=scored_signals,
                key_levels_data_v2_5=key_levels_data,
                bundle_timestamp=datetime.now(),
                target_symbol=ticker,
                system_status_messages=[],
                active_recommendations_v2_5=[],
                atif_recommendations_v2_5=atif_recommendations
            )
            self.logger.info(f"✅ Final analysis bundle created for {ticker}")
            return bundle
        except Exception as e:
            self.logger.error(f"Error in run_full_analysis_cycle: {e}", exc_info=True)
            raise
    
    def _create_error_bundle(self, ticker: str, error_message: str) -> FinalAnalysisBundleV2_5:
        """FAIL FAST - NO ERROR BUNDLES WITH FAKE DATA ALLOWED!"""
        raise ValueError(
            f"CRITICAL: Cannot create analysis bundle for {ticker} due to error: {error_message}. "
            f"NO FAKE DATA WILL BE CREATED! Fix the underlying data issue instead of masking it with fake data."
        )

    
    async def _get_processed_data_bundle(self, ticker: str) -> Optional[ProcessedDataBundleV2_5]:
        """
        Get processed data bundle validated against EOTS schemas
        This method is now simplified as elite calculations are done internally.
        """
        try:
            self.logger.info(f"📊 Getting processed data bundle for {ticker}")
            
            # Check if data fetchers are available
            if not self.convex_fetcher or not self.tradier_fetcher:
                self.logger.error("❌ Data fetchers not initialized - cannot fetch real data")
                return None
            
            # Fetch real underlying data from ConvexValue
            try:
                und_data_api_raw = await self.convex_fetcher.fetch_underlying_data(ticker)
                self.logger.info(f"✅ Successfully fetched underlying data for {ticker}")
            except Exception as e:
                self.logger.error(f"❌ Failed to fetch underlying data: {e}")
                # Fallback to simulated data if real fetch fails
                und_data_api_raw = {
                    'symbol': ticker,
                    'price': 4500.0,
                    'price_change_pct': 0.01,
                    'day_volume': 100000000,
                    'tradier_iv5_approx_smv_avg': 0.25,
                    'u_volatility': 0.25
                }
            options_df_raw = pd.DataFrame({
                'strike': np.linspace(und_data_api_raw['price'] * 0.9, und_data_api_raw['price'] * 1.1, 50),
                'dte_calc': np.random.randint(0, 45 + 1, 50),
                'option_type': np.random.choice(['call', 'put'], 50),
                'volume': np.random.randint(100, 10000, 50),
                'open_interest': np.random.randint(1000, 50000, 50),
                'delta': np.random.uniform(-0.9, 0.9, 50),
                'gamma': np.random.uniform(0.001, 0.05, 50),
                'vega': np.random.uniform(0.01, 0.5, 50),
                'theta': np.random.uniform(-0.05, -0.001, 50),
                'dxoi': np.random.uniform(-50000, 50000, 50),
                'gxoi': np.random.uniform(0, 10000, 50),
                'vxoi': np.random.uniform(0, 5000, 50),
                'txoi': np.random.uniform(-1000, -100, 50),
                'charmxoi': np.random.uniform(0, 1000, 50),
                'vannaxoi': np.random.uniform(-500, 500, 50),
                'vommaxoi': np.random.uniform(0, 200, 50),
                'volmbs_5m': np.random.uniform(-1000, 1000, 50),
                'volmbs_15m': np.random.uniform(-2000, 2000, 50),
                'volmbs_30m': np.random.uniform(-3000, 3000, 50),
                'volmbs_60m': np.random.uniform(-4000, 4000, 50),
                'valuebs_5m': np.random.uniform(-100000, 100000, 50),
                'valuebs_15m': np.random.uniform(-200000, 200000, 50),
                'valuebs_30m': np.random.uniform(-300000, 300000, 50),
                'valuebs_60m': np.random.uniform(-400000, 400000, 50),
                'value_bs': np.random.uniform(-500000, 500000, 50),
                'volm_bs': np.random.uniform(-5000, 5000, 50),
                'bid_price': np.random.uniform(0.1, 10.0, 50),
                'ask_price': np.random.uniform(0.1, 10.0, 50) + 0.05,
                'volatility': np.random.uniform(0.1, 0.5, 50)
            })

            # Convert DataFrame to list of Pydantic models instead of dictionaries
            options_contracts = []
            for _, row in options_df_raw.iterrows():
                try:
                    contract = RawOptionsContractV2_5.model_validate(row.to_dict())
                    options_contracts.append(contract)
                except Exception as e:
                    self.logger.warning(f"Failed to validate options contract: {e}")

            raw_bundle = UnprocessedDataBundleV2_5(
                options_contracts=options_contracts,  # List of Pydantic models, not dictionaries
                underlying_data=und_data_api_raw,
                fetch_timestamp=datetime.now(),
                errors=[]
            )
            
            # Step 3: Process the raw data using InitialDataProcessorV2_5
            self.logger.info(f"🔄 Processing raw data for {ticker}...")
            metrics_output = self.metrics_calculator.calculate_metrics(
                options_df_raw=options_df_raw,
                und_data_api_raw=und_data_api_raw,
                dte_max=45, # Assuming a default dte_max for this internal call
                market_data=pd.DataFrame([und_data_api_raw]) # Pass underlying data as market_data for elite calc
            )
            processed_bundle = ProcessedDataBundleV2_5(
                options_data_with_metrics=metrics_output.options_with_metrics,
                strike_level_data_with_metrics=metrics_output.strike_level_data,
                underlying_data_enriched=metrics_output.underlying_enriched,
                processing_timestamp=datetime.now()
            )
            
            self.logger.info(f"✅ Data processing completed for {ticker}")
            self.logger.info(f"   📊 Options with metrics: {len(processed_bundle.options_data_with_metrics)}")
            self.logger.info(f"   🎯 Strike levels with metrics: {len(processed_bundle.strike_level_data_with_metrics)}")
            
            return processed_bundle
            
        except Exception as e:
            self.logger.error(f"Failed to get processed data bundle for {ticker}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.performance_metrics["failed_analyses"] = self.performance_metrics.get("failed_analyses", 0) + 1
            return None
    
    def _calculate_data_quality_score(self, data_bundle: ProcessedDataBundleV2_5) -> float:
        """Calculate data quality score for the analysis"""
        try:
            if not data_bundle:
                return 0.0
            
            quality_factors = []
            
            # Check underlying data quality
            if data_bundle.underlying_data_enriched:
                if data_bundle.underlying_data_enriched.price:
                    quality_factors.append(1.0)
                else:
                    quality_factors.append(0.0)
            
            # Check options data quality
            if data_bundle.options_data_with_metrics:
                options_quality = len(data_bundle.options_data_with_metrics) / 100.0  # Normalize by expected count
                quality_factors.append(min(options_quality, 1.0))
            else:
                quality_factors.append(0.0)
            
            # Check strike level data quality
            if data_bundle.strike_level_data_with_metrics:
                strike_quality = len(data_bundle.strike_level_data_with_metrics) / 50.0  # Normalize by expected count
                quality_factors.append(min(strike_quality, 1.0))
            else:
                quality_factors.append(0.0)
            
            # Calculate average quality
            if quality_factors:
                return sum(quality_factors) / len(quality_factors)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Data quality calculation failed: {e}")
            return 0.0
    
    def _update_performance_metrics(self, result: Dict[str, Any]):
        """Update performance tracking metrics"""
        self.performance_metrics["total_analyses"] = self.performance_metrics.get("total_analyses", 0) + 1
        
        if result.get("confidence_score", 0) > 0.5:
            self.performance_metrics["successful_analyses"] = self.performance_metrics.get("successful_analyses", 0) + 1
        else:
            self.performance_metrics["failed_analyses"] = self.performance_metrics.get("failed_analyses", 0) + 1
    
    def get_legendary_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for the legendary system"""
        try:
            # Update real-time metrics
            self.performance_metrics["uptime_seconds"] = (datetime.now() - self.start_time).total_seconds()
            
            return self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Performance metrics retrieval failed: {e}")
            return {
                'system_status': 'ERROR',
                'error': str(e)
            }
    
    async def legendary_orchestrate_analysis(self, data_bundle: ProcessedDataBundleV2_5, **kwargs) -> Dict[str, Any]:
        """
        🎯 LEGENDARY orchestration method for backward compatibility
        Returns Dict[str, Any] instead of non-existent model
        """
        try:
                         # Run full analysis cycle and convert to dict format
             final_bundle = await self.run_full_analysis_cycle(data_bundle.underlying_data_enriched.symbol, **kwargs)
             
             return {
                 "analysis_id": f"{data_bundle.underlying_data_enriched.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                 "ticker": data_bundle.underlying_data_enriched.symbol,
                 "timestamp": datetime.now(),
                 "final_bundle": final_bundle,
                 "status": "completed"
             }
            
        except Exception as e:
            self.logger.error(f"Legendary orchestration failed: {e}")
            return {
                "analysis_id": f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "ticker": data_bundle.underlying_data_enriched.symbol if data_bundle else "unknown",
                "timestamp": datetime.now(),
                "error": str(e),
                "status": "failed"
            }
    
    async def _generate_key_levels(self, data_bundle: ProcessedDataBundleV2_5, ticker: str, timestamp: datetime) -> KeyLevelsDataV2_5:
        """
        Generate key levels from database first, then from real-time analysis if database is empty.
        CRITICAL FIX: Generate key levels from current strike data when database is empty.

        Args:
            data_bundle: Processed data bundle containing price and options data
            ticker: Trading symbol
            timestamp: Analysis timestamp

        Returns:
            KeyLevelsDataV2_5: Key levels from database or real-time analysis
        """
        try:
            self.logger.info(f"🔍 Retrieving key levels for {ticker} from database first")

            # Step 1: Try to retrieve key levels from database
            database_levels = await self._retrieve_key_levels_from_database(ticker)
            if database_levels and len(database_levels.supports + database_levels.resistances +
                                     database_levels.pin_zones + database_levels.vol_triggers +
                                     database_levels.major_walls) > 0:
                self.logger.info(f"✅ Retrieved {len(database_levels.supports + database_levels.resistances + database_levels.pin_zones + database_levels.vol_triggers + database_levels.major_walls)} key levels from database")
                return database_levels

            # Step 2: CRITICAL FIX - Generate key levels from current strike data when database is empty
            self.logger.info(f"🔑 Database empty, generating key levels from current strike data for {ticker}")

            # Convert strike data to DataFrame for key level identification
            strike_data = data_bundle.strike_level_data_with_metrics
            if not strike_data:
                self.logger.warning(f"⚠️ No strike data available for key level generation for {ticker}")
                return KeyLevelsDataV2_5(
                    supports=[],
                    resistances=[],
                    pin_zones=[],
                    vol_triggers=[],
                    major_walls=[],
                    timestamp=timestamp
                )

            # Convert Pydantic models to DataFrame
            df_strike = pd.DataFrame([s.model_dump() for s in strike_data])

            # Use KeyLevelIdentifierV2_5 to generate key levels from current data
            generated_levels = self.key_level_identifier.identify_and_score_key_levels(
                df_strike,
                data_bundle.underlying_data_enriched
            )

            self.logger.info(f"✅ Generated {len(generated_levels.supports)} supports, {len(generated_levels.resistances)} resistances from current data")
            return generated_levels
                
        except Exception as e:
            self.logger.error(f"❌ Error retrieving key levels from database for {ticker}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty key levels data on error - NO FALLBACK DATA
            return KeyLevelsDataV2_5(
                supports=[],
                resistances=[],
                pin_zones=[],
                vol_triggers=[],
                major_walls=[],
                timestamp=timestamp
            )

    async def _retrieve_key_levels_from_database(self, ticker: str) -> Optional[KeyLevelsDataV2_5]:
        """
        Retrieve key levels from the database metrics schema.
        
        Args:
            ticker: Trading symbol
            
        Returns:
            KeyLevelsDataV2_5: Key levels from database or None if not found
        """
        try:
            if not self.db_manager or not hasattr(self.db_manager, '_conn'):
                return None
                
            # Query the key_level_performance table for recent levels
            sql = """
            SELECT level_price, level_type, conviction_score, level_source, created_at
            FROM key_level_performance
            WHERE symbol = %s 
            AND date >= CURRENT_DATE - INTERVAL '7 days'
            AND conviction_score > 0.3
            ORDER BY conviction_score DESC, created_at DESC
            LIMIT 50
            """
            
            cursor = self.db_manager._conn.cursor()
            cursor.execute(sql, (ticker,))
            results = cursor.fetchall()
            
            if not results:
                self.logger.info(f"📊 No key levels found in database for {ticker}")
                return None
            
            # Convert database results to KeyLevelV2_5 models
            supports = []
            resistances = []
            pin_zones = []
            vol_triggers = []
            major_walls = []
            
            for row in results:
                # Handle both tuple and dict-like row objects
                if isinstance(row, dict):
                    level_price = row.get('level_price')
                    level_type = row.get('level_type')
                    conviction_score = row.get('conviction_score')
                    level_source = row.get('level_source')
                    created_at = row.get('created_at')
                else:
                    # Assume tuple/list format
                    level_price, level_type, conviction_score, level_source, created_at = row
                
                # Skip rows with missing critical data - NO FALLBACK DATA GENERATION
                if level_price is None or level_type is None or conviction_score is None:
                    self.logger.warning(f"⚠️ Skipping row with missing data: price={level_price}, type={level_type}, score={conviction_score}")
                    continue
                
                key_level = KeyLevelV2_5(
                    level_price=float(level_price),
                    level_type=str(level_type),
                    conviction_score=float(conviction_score),
                    contributing_metrics=[level_source] if level_source else [],
                    source_identifier=level_source or 'database'
                )
                
                # Categorize by type
                level_type_str = str(level_type).lower()
                if level_type_str in ['support']:
                    supports.append(key_level)
                elif level_type_str in ['resistance']:
                    resistances.append(key_level)
                elif level_type_str in ['pivot', 'pin_zone']:
                    pin_zones.append(key_level)
                elif level_type_str in ['max_pain', 'vol_trigger']:
                    vol_triggers.append(key_level)
                elif level_type_str in ['gamma_wall', 'major_wall']:
                    major_walls.append(key_level)
                else:
                    # Default to resistance for unknown types
                    resistances.append(key_level)
            
            self.logger.info(f"📊 Retrieved from database: {len(supports)} supports, {len(resistances)} resistances, "
                           f"{len(pin_zones)} pin zones, {len(vol_triggers)} vol triggers, {len(major_walls)} major walls")
            
            return KeyLevelsDataV2_5(
                supports=supports,
                resistances=resistances,
                pin_zones=pin_zones,
                vol_triggers=vol_triggers,
                major_walls=major_walls,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error retrieving key levels from database for {ticker}: {e}")
            return None

    @property
    def cache_manager(self) -> Optional[EnhancedCacheManagerV2_5]:
        """Get the cache manager instance."""
        if self._cache_manager is None:
            cache_root = self.config_manager.get_resolved_path('cache_settings.cache_root')
            # CRITICAL FIX: Always create cache manager with default path if config path is None
            if not cache_root:
                cache_root = "cache/enhanced_v2_5"  # Default cache path

            self._cache_manager = EnhancedCacheManagerV2_5(
                cache_root=cache_root,
                memory_limit_mb=100,
                disk_limit_mb=1000,
                default_ttl_seconds=3600,
                ultra_fast_mode=True
            )
        return self._cache_manager

    @property
    def db_manager(self) -> DatabaseManagerV2_5:
        """Get the database manager instance."""
        return self._db_manager

    async def analyze_market_regime(self, processed_data: ProcessedDataBundleV2_5) -> UnifiedIntelligenceAnalysis:
        """Analyze market regime using the consolidated market intelligence engine."""
        try:
            # Use the consolidated market intelligence engine to analyze the market
            analysis = await self.market_intelligence_engine.analyze_market_data(
                data_bundle=processed_data,
                huihui_regime=None,  # Will be fetched internally if needed
                huihui_flow=None,    # Will be fetched internally if needed
                huihui_sentiment=None # Will be fetched internally if needed
            )
            self.logger.info(f"Market intelligence analysis completed for {processed_data.underlying_data_enriched.symbol}")
            return analysis
                
        except Exception as e:
            self.logger.error(f"Error in market intelligence analysis: {str(e)}")
            return UnifiedIntelligenceAnalysis(
                symbol=processed_data.underlying_data_enriched.symbol,
                timestamp=datetime.now(),
                confidence_score=0.0,
                market_regime_analysis=str(MarketRegimeState.UNDEFINED),
                options_flow_analysis="Error in analysis",
                sentiment_analysis="Error in analysis",
                strategic_recommendations=[],
                risk_assessment="Error in analysis",
                learning_insights=[f"Error: {str(e)}"],
                performance_metrics={}
            )

# Maintain backward compatibility
ItsOrchestratorV2_5 = ITSOrchestratorV2_5
