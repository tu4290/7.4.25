"""Detailed Pydantic models for configuration schemas.

This module defines specific Pydantic models to replace Dict[str, Any] patterns
found in configuration_schemas.py, providing better type safety and validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union


# --- Data Processor Settings ---
class DataProcessorFactors(BaseModel):
    """Various numerical factors used in metric calculations."""
    tdpi_gaussian_width: Optional[float] = Field(None, description="Gaussian width for TDPI calculations")
    flow_smoothing_factor: Optional[float] = Field(None, description="Factor for smoothing flow calculations")
    volatility_adjustment_factor: Optional[float] = Field(None, description="Factor for volatility adjustments")
    momentum_decay_factor: Optional[float] = Field(None, description="Decay factor for momentum calculations")
    regime_transition_sensitivity: Optional[float] = Field(None, description="Sensitivity for regime transitions")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class IVContextParameters(BaseModel):
    """Parameters for IV contextualization."""
    vol_trend_avg_days: Optional[int] = Field(None, description="Days for volatility trend averaging")
    iv_rank_lookback_days: Optional[int] = Field(None, description="Lookback days for IV rank calculation")
    iv_percentile_window: Optional[int] = Field(None, description="Window for IV percentile calculation")
    term_structure_analysis_enabled: Optional[bool] = Field(None, description="Enable term structure analysis")
    skew_analysis_enabled: Optional[bool] = Field(None, description="Enable skew analysis")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


# --- Market Regime Engine Settings ---
class RegimeRuleConditions(BaseModel):
    """Conditions for a specific market regime rule."""
    vix_threshold: Optional[float] = Field(None, description="VIX threshold for regime")
    flow_alignment_threshold: Optional[float] = Field(None, description="Flow alignment threshold")
    volatility_regime: Optional[str] = Field(None, description="Volatility regime condition")
    momentum_condition: Optional[str] = Field(None, description="Momentum condition")
    structure_condition: Optional[str] = Field(None, description="Structure condition")
    confidence_threshold: Optional[float] = Field(None, description="Confidence threshold")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class RegimeRules(BaseModel):
    """Dictionary of rules defining conditions for each market regime."""
    # Basic regime types
    BULLISH_MOMENTUM: Optional[RegimeRuleConditions] = Field(None, description="Rules for bullish momentum regime")
    BEARISH_MOMENTUM: Optional[RegimeRuleConditions] = Field(None, description="Rules for bearish momentum regime")
    CONSOLIDATION: Optional[RegimeRuleConditions] = Field(None, description="Rules for consolidation regime")
    HIGH_VOLATILITY: Optional[RegimeRuleConditions] = Field(None, description="Rules for high volatility regime")
    LOW_VOLATILITY: Optional[RegimeRuleConditions] = Field(None, description="Rules for low volatility regime")
    REGIME_UNCLEAR_OR_TRANSITIONING: Optional[RegimeRuleConditions] = Field(None, description="Rules for unclear/transitioning regime")
    
    # Specific advanced regime types from config
    REGIME_SPX_0DTE_FRIDAY_EOD_VANNA_CASCADE_POTENTIAL_BULLISH: Optional[RegimeRuleConditions] = Field(None, description="Rules for SPX 0DTE Friday EOD Vanna cascade bullish regime")
    REGIME_SPY_PRE_FOMC_VOL_COMPRESSION_WITH_DWFD_ACCUMULATION: Optional[RegimeRuleConditions] = Field(None, description="Rules for SPY pre-FOMC vol compression with DWFD accumulation regime")
    REGIME_HIGH_VAPI_FA_BULLISH_MOMENTUM_UNIVERSAL: Optional[RegimeRuleConditions] = Field(None, description="Rules for high VAPI-FA bullish momentum universal regime")
    REGIME_ADAPTIVE_STRUCTURE_BREAKDOWN_WITH_DWFD_CONFIRMATION_BEARISH_UNIVERSAL: Optional[RegimeRuleConditions] = Field(None, description="Rules for adaptive structure breakdown with DWFD confirmation bearish universal regime")
    REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BULLISH: Optional[RegimeRuleConditions] = Field(None, description="Rules for vol expansion imminent VRI 0DTE bullish regime")
    REGIME_VOL_EXPANSION_IMMINENT_VRI0DTE_BEARISH: Optional[RegimeRuleConditions] = Field(None, description="Rules for vol expansion imminent VRI 0DTE bearish regime")
    REGIME_NVP_STRONG_BUY_IMBALANCE_AT_KEY_STRIKE: Optional[RegimeRuleConditions] = Field(None, description="Rules for NVP strong buy imbalance at key strike regime")
    REGIME_EOD_HEDGING_PRESSURE_BUY: Optional[RegimeRuleConditions] = Field(None, description="Rules for EOD hedging pressure buy regime")
    REGIME_EOD_HEDGING_PRESSURE_SELL: Optional[RegimeRuleConditions] = Field(None, description="Rules for EOD hedging pressure sell regime")
    REGIME_SIDEWAYS_MARKET: Optional[RegimeRuleConditions] = Field(None, description="Rules for sideways market regime")
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for regime, conditions in self.dict().items():
            if conditions is not None:
                if isinstance(conditions, RegimeRuleConditions):
                    result[regime] = conditions.to_dict()
                else:
                    result[regime] = conditions
        return result


# --- Enhanced Flow Metric Settings ---
class VAPIFAParameters(BaseModel):
    """Parameters specific to VAPI-FA calculation."""
    primary_flow_interval: Optional[str] = Field(None, description="Primary flow interval for VAPI-FA")
    iv_source_key: Optional[str] = Field(None, description="IV source key for calculations")
    flow_acceleration_window: Optional[int] = Field(None, description="Window for flow acceleration")
    volatility_adjustment_enabled: Optional[bool] = Field(None, description="Enable volatility adjustment")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class DWFDParameters(BaseModel):
    """Parameters specific to DWFD calculation."""
    flow_interval: Optional[str] = Field(None, description="Flow interval for DWFD")
    fvd_weight_factor: Optional[float] = Field(None, description="FVD weight factor")
    divergence_threshold: Optional[float] = Field(None, description="Divergence threshold")
    smoothing_factor: Optional[float] = Field(None, description="Smoothing factor")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class TWLAFParameters(BaseModel):
    """Parameters specific to TW-LAF calculation."""
    time_weights_for_intervals: Optional[List[float]] = Field(None, description="Time weights for intervals")
    spread_calculation_params: Optional[Dict[str, float]] = Field(None, description="Spread calculation parameters")
    liquidity_adjustment_factor: Optional[float] = Field(None, description="Liquidity adjustment factor")
    flow_normalization_method: Optional[str] = Field(None, description="Flow normalization method")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


# --- ATIF Settings ---
class SignalIntegrationParameters(BaseModel):
    """Parameters for how ATIF integrates and weights raw signals."""
    base_signal_weights: Optional[Dict[str, float]] = Field(None, description="Base weights for different signals")
    performance_weighting_sensitivity: Optional[float] = Field(None, description="Sensitivity for performance-based weighting")
    signal_decay_factor: Optional[float] = Field(None, description="Decay factor for signal strength")
    correlation_adjustment_enabled: Optional[bool] = Field(None, description="Enable correlation adjustments")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class ConvictionMappingParameters(BaseModel):
    """Rules and thresholds for mapping ATIF's internal assessment to final conviction score."""
    strong_signal_threshold: Optional[float] = Field(None, description="Threshold for strong signals")
    moderate_signal_threshold: Optional[float] = Field(None, description="Threshold for moderate signals")
    weak_signal_threshold: Optional[float] = Field(None, description="Threshold for weak signals")
    conviction_scaling_factor: Optional[float] = Field(None, description="Scaling factor for conviction scores")
    max_conviction_score: Optional[float] = Field(None, description="Maximum conviction score")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class StrategySpecificRule(BaseModel):
    """Individual rule for strategy specificity."""
    assessment_condition: Optional[str] = Field(None, description="Assessment condition")
    conviction_range: Optional[List[float]] = Field(None, description="Conviction range")
    regime_context: Optional[str] = Field(None, description="Regime context")
    iv_context: Optional[str] = Field(None, description="IV context")
    recommended_strategy: Optional[str] = Field(None, description="Recommended strategy")
    target_dte_range: Optional[List[int]] = Field(None, description="Target DTE range")
    target_delta_range: Optional[List[float]] = Field(None, description="Target delta range")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class IntelligentRecommendationManagementRules(BaseModel):
    """Rules for adaptive exits, parameter adjustments, and partial position management."""
    adaptive_exit_enabled: Optional[bool] = Field(None, description="Enable adaptive exits")
    profit_target_adjustment_factor: Optional[float] = Field(None, description="Profit target adjustment factor")
    stop_loss_adjustment_factor: Optional[float] = Field(None, description="Stop loss adjustment factor")
    partial_position_thresholds: Optional[List[float]] = Field(None, description="Thresholds for partial position management")
    time_decay_adjustment_enabled: Optional[bool] = Field(None, description="Enable time decay adjustments")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


# --- Ticker Context Analyzer Settings ---
class TickerSpecificParameters(BaseModel):
    """Specific context analysis parameters for a ticker."""
    volatility_lookback_days: Optional[int] = Field(None, description="Volatility lookback days")
    correlation_threshold: Optional[float] = Field(None, description="Correlation threshold")
    volume_profile_enabled: Optional[bool] = Field(None, description="Enable volume profile analysis")
    earnings_impact_factor: Optional[float] = Field(None, description="Earnings impact factor")
    sector_correlation_weight: Optional[float] = Field(None, description="Sector correlation weight")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


# --- Heatmap Generation Settings ---
class UGCHParameters(BaseModel):
    """Parameters for UGCH data generation."""
    greek_weights: Optional[Dict[str, float]] = Field(None, description="Weights for each Greek")
    normalization_method: Optional[str] = Field(None, description="Normalization method")
    smoothing_factor: Optional[float] = Field(None, description="Smoothing factor")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class SGDHPParameters(BaseModel):
    """Parameters for SGDHP data generation."""
    proximity_sensitivity_param: Optional[float] = Field(None, description="Price proximity sensitivity")
    decay_factor: Optional[float] = Field(None, description="Decay factor")
    clustering_threshold: Optional[float] = Field(None, description="Clustering threshold")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class IVSDHParameters(BaseModel):
    """Parameters for IVSDH data generation."""
    time_decay_sensitivity_factor: Optional[float] = Field(None, description="Time decay sensitivity factor")
    volatility_surface_smoothing: Optional[bool] = Field(None, description="Enable volatility surface smoothing")
    skew_adjustment_factor: Optional[float] = Field(None, description="Skew adjustment factor")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


# --- Adaptive Metric Parameters ---
class ADAGSettings(BaseModel):
    """Settings for Adaptive Delta Adjusted Gamma Exposure (A-DAG)."""
    calculation_method: Optional[str] = Field("standard", description="Method for A-DAG calculation")
    flow_alignment_threshold: Optional[float] = Field(0.1, description="Threshold for flow alignment detection")
    smoothing_factor: Optional[float] = Field(0.2, description="Smoothing factor for A-DAG calculation")
    lookback_periods: Optional[int] = Field(20, description="Lookback periods for A-DAG analysis")
    custom_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom A-DAG parameters")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ESDAGSettings(BaseModel):
    """Settings for Enhanced Skew and Delta Adjusted Gamma Exposure (E-SDAG)."""
    skew_calculation_method: Optional[str] = Field("enhanced", description="Method for skew calculation")
    delta_adjustment_factor: Optional[float] = Field(1.0, description="Factor for delta adjustment")
    gamma_exposure_weight: Optional[float] = Field(0.5, description="Weight for gamma exposure component")
    volatility_surface_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Volatility surface parameters")
    custom_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom E-SDAG parameters")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class DTDPISettings(BaseModel):
    """Settings for Dynamic Time Decay Pressure Indicator (D-TDPI)."""
    time_decay_model: Optional[str] = Field("black_scholes", description="Model for time decay calculation")
    pressure_threshold: Optional[float] = Field(0.05, description="Threshold for pressure detection")
    charm_flow_weight: Optional[float] = Field(0.3, description="Weight for charm flow component")
    dynamic_adjustment_factor: Optional[float] = Field(1.2, description="Factor for dynamic adjustments")
    custom_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom D-TDPI parameters")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class VRI2Settings(BaseModel):
    """Settings for Volatility Regime Indicator Version 2.0 (VRI 2.0)."""
    vanna_flow_proxy_weight: Optional[float] = Field(None, description="Vanna flow proxy weight")
    volatility_regime_threshold: Optional[float] = Field(None, description="Volatility regime threshold")
    regime_transition_smoothing: Optional[bool] = Field(None, description="Enable regime transition smoothing")
    historical_volatility_window: Optional[int] = Field(None, description="Historical volatility window")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class EnhancedHeatmapSettingsDetailed(BaseModel):
    """Detailed settings for enhanced heatmap generation."""
    color_scheme: Optional[str] = Field(None, description="Color scheme for heatmap visualization")
    grid_resolution: Optional[int] = Field(None, description="Resolution of the heatmap grid")
    interpolation_method: Optional[str] = Field(None, description="Interpolation method for heatmap data")
    show_annotations: Optional[bool] = Field(None, description="Whether to show value annotations on heatmap")
    transparency_level: Optional[float] = Field(None, description="Transparency level for heatmap overlay")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class SymbolOverrideSettings(BaseModel):
    """Settings for symbol-specific configuration overrides."""
    strategy_multiplier: Optional[float] = Field(None, description="Multiplier for strategy parameters")
    risk_adjustment: Optional[float] = Field(None, description="Risk adjustment factor for the symbol")
    volume_threshold: Optional[int] = Field(None, description="Minimum volume threshold for trading")
    volatility_adjustment: Optional[float] = Field(None, description="Volatility adjustment factor")
    custom_parameters: Optional[Dict[str, Any]] = Field(None, description="Custom symbol-specific parameters")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class ApiKeysSettings(BaseModel):
    """Settings for API keys configuration."""
    tradier_api_key: Optional[str] = Field(None, description="Tradier API key")
    convexvalue_api_key: Optional[str] = Field(None, description="ConvexValue API key")
    polygon_api_key: Optional[str] = Field(None, description="Polygon API key")
    alpha_vantage_api_key: Optional[str] = Field(None, description="Alpha Vantage API key")
    custom_api_keys: Optional[Dict[str, str]] = Field(None, description="Custom API keys")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class EnhancedHeatmapSettings(BaseModel):
    """Settings for enhanced heatmap generation."""
    resolution: Optional[str] = Field(None, description="Heatmap resolution")
    color_scheme: Optional[str] = Field(None, description="Color scheme")
    interpolation_method: Optional[str] = Field(None, description="Interpolation method")
    data_aggregation_method: Optional[str] = Field(None, description="Data aggregation method")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


# --- Trade Parameter Optimizer Settings ---
class ContractSelectionFilters(BaseModel):
    """Filters for selecting optimal option contracts."""
    min_volume_threshold_pct_of_max_for_chain: Optional[float] = Field(None, description="Minimum volume threshold percentage")
    min_open_interest_threshold_pct_of_max_for_chain: Optional[float] = Field(None, description="Minimum open interest threshold percentage")
    max_allowable_relative_spread_pct: Optional[float] = Field(None, description="Maximum allowable relative spread percentage")
    delta_matching_tolerance_abs: Optional[float] = Field(None, description="Delta matching tolerance")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class StopLossCalculationRules(BaseModel):
    """Rules for calculating adaptive stop-losses."""
    base_atr_multiplier_standard_risk: Optional[float] = Field(None, description="Base ATR multiplier for standard risk")
    atr_risk_posture_adjustment_factors: Optional[Dict[str, float]] = Field(None, description="ATR risk posture adjustment factors")
    key_level_buffer_pct_for_sl: Optional[float] = Field(None, description="Key level buffer percentage for stop loss")
    max_allowable_initial_stop_loss_pct_of_price: Optional[float] = Field(None, description="Maximum allowable initial stop loss percentage")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class ProfitTargetCalculationRules(BaseModel):
    """Rules for calculating multi-tiered profit targets."""
    num_profit_targets_to_generate: Optional[int] = Field(None, description="Number of profit targets to generate")
    pt_level_selection_hierarchy: Optional[str] = Field(None, description="Profit target level selection hierarchy")
    min_reward_to_risk_ratio_pt1: Optional[float] = Field(None, description="Minimum reward to risk ratio for PT1")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


# --- Prediction Configuration ---
class ConfidenceCalibration(BaseModel):
    """Parameters for calibrating confidence scores based on signal strength."""
    strong_signal_threshold: Optional[float] = Field(None, description="Strong signal threshold")
    moderate_signal_threshold: Optional[float] = Field(None, description="Moderate signal threshold")
    max_confidence: Optional[float] = Field(None, description="Maximum confidence")
    min_confidence: Optional[float] = Field(None, description="Minimum confidence")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


# --- Elite Configuration ---
class EliteConfigurationSettings(BaseModel):
    """Elite regime detection configuration settings."""
    enable_elite_regime_detection: Optional[bool] = Field(None, description="Enable elite regime detection")
    elite_regime_threshold: Optional[float] = Field(None, description="Elite regime threshold")
    advanced_pattern_recognition: Optional[bool] = Field(None, description="Enable advanced pattern recognition")
    machine_learning_integration: Optional[bool] = Field(None, description="Enable machine learning integration")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


# --- Time of Day Definitions ---
class TimeOfDayDefinitions(BaseModel):
    """Time of day definitions for market analysis."""
    market_open: Optional[str] = Field(None, description="Market open time")
    market_close: Optional[str] = Field(None, description="Market close time")
    pre_market_start: Optional[str] = Field(None, description="Pre-market start time")
    after_hours_end: Optional[str] = Field(None, description="After hours end time")
    lunch_break_start: Optional[str] = Field(None, description="Lunch break start time")
    lunch_break_end: Optional[str] = Field(None, description="Lunch break end time")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class StrategyParameters(BaseModel):
    """Extensible strategy parameters for trading configuration."""
    position_sizing_method: Optional[str] = Field(None, description="Method for position sizing")
    risk_management_rules: Optional[Dict[str, Any]] = Field(None, description="Risk management rules")
    entry_conditions: Optional[Dict[str, Any]] = Field(None, description="Entry condition parameters")
    exit_conditions: Optional[Dict[str, Any]] = Field(None, description="Exit condition parameters")
    portfolio_allocation: Optional[float] = Field(None, description="Portfolio allocation percentage")
    max_concurrent_positions: Optional[int] = Field(None, description="Maximum concurrent positions")
    custom_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom strategy parameters")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}

class RegimeContextWeightMultipliers(BaseModel):
    """Weight multipliers applied to signal weights based on current market regime."""
    bullish: Optional[float] = Field(1.2, description="Multiplier for bullish market regime")
    bearish: Optional[float] = Field(0.8, description="Multiplier for bearish market regime")
    neutral: Optional[float] = Field(1.0, description="Multiplier for neutral market regime")
    high_volatility: Optional[float] = Field(0.9, description="Multiplier for high volatility regime")
    low_volatility: Optional[float] = Field(1.1, description="Multiplier for low volatility regime")
    trending: Optional[float] = Field(1.3, description="Multiplier for trending market regime")
    range_bound: Optional[float] = Field(0.7, description="Multiplier for range-bound market regime")
    custom_multipliers: Optional[Dict[str, float]] = Field(default_factory=dict, description="Custom regime multipliers")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}