"""
Pydantic models for representing generated trading signals and identified
key price levels within the EOTS v2.5 system.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from .signal_level_config_schemas import SupportingMetrics

class SignalPayloadV2_5(BaseModel):
    """
    Represents a single, discrete trading signal generated by SignalGeneratorV2_5.
    It encapsulates all relevant information about a specific signal event,
    including its nature, strength, and the context in which it occurred.
    These payloads are primary inputs to the Adaptive Trade Idea Framework (ATIF).
    """
    signal_id: str = Field(..., description="A unique identifier for this specific signal instance (e.g., UUID).")
    signal_name: str = Field(..., description="A human-readable name for the signal (e.g., 'VAPI_FA_Bullish_Surge', 'A_MSPI_Support_Confirmed').")
    symbol: str = Field(..., description="The ticker symbol for which the signal was generated.")
    timestamp: datetime = Field(..., description="Timestamp of when the signal was generated.")
    signal_type: str = Field(..., description="Categorization of the signal (e.g., 'Directional', 'Volatility', 'Structural', 'Flow_Divergence', 'Warning').")
    direction: Optional[str] = Field(None, description="If applicable, the directional bias of the signal (e.g., 'Bullish', 'Bearish', 'Neutral').")
    strength_score: float = Field(..., ge=-5.0, le=5.0, description="A numerical score (e.g., -1.0 to +1.0 or 0 to 1.0) indicating the intensity or confidence of this raw signal, generated by SignalGeneratorV2_5 before ATIF weighting.")
    strike_impacted: Optional[float] = Field(None, description="The specific strike price most relevant to this signal, if applicable (e.g., for a structural signal).")
    regime_at_signal_generation: Optional[str] = Field(None, description="The market regime active when this signal was triggered.")
    supporting_metrics: SupportingMetrics = Field(default_factory=SupportingMetrics, description="Key metric values and their states that contributed to triggering this signal, providing context for ATIF evaluation.")

    class Config:
        extra = 'forbid' # Internal model, structure should be strictly defined


class KeyLevelV2_5(BaseModel):
    """
    Represents a single identified key price level (e.g., support, resistance,
    pin zone, volatility trigger, major wall). It includes the level's price,
    type, a conviction score, and the metrics that contributed to its identification.
    """
    level_price: float = Field(..., description="The price of the identified key level.")
    level_type: str = Field(..., description="The type of key level (e.g., 'Support', 'Resistance', 'PinZone', 'VolTrigger', 'MajorWall').")
    conviction_score: float = Field(..., description="A score (e.g., 0.0 to 1.0 or 1-5) indicating the system's confidence in this level's significance, based on confluence and source strength.")
    contributing_metrics: List[str] = Field(default_factory=list, description="A list of metric names or source identifiers that flagged or support this level (e.g., ['A-MSPI', 'NVP_Peak', 'SGDHP_Zone']).")
    source_identifier: Optional[str] = Field(None, description="An optional string to identify the primary method or specific source that generated this level (e.g., 'A-MSPI_Daily', 'UGCH_Strike_4500').")

    class Config:
        extra = 'forbid' # Internal model


class KeyLevelsDataV2_5(BaseModel):
    """
    A container object that aggregates all identified key levels for a given
    analysis cycle, categorized by type. It provides a structured overview
    of critical price zones identified by KeyLevelIdentifierV2_5.
    """
    supports: List[KeyLevelV2_5] = Field(default_factory=list, description="A list of identified support levels.")
    resistances: List[KeyLevelV2_5] = Field(default_factory=list, description="A list of identified resistance levels.")
    pin_zones: List[KeyLevelV2_5] = Field(default_factory=list, description="A list of identified potential pinning zones (often from D-TDPI + vci_0dte).")
    vol_triggers: List[KeyLevelV2_5] = Field(default_factory=list, description="A list of identified volatility trigger levels (often from E-SDAG_VF or IVSDH data).")
    major_walls: List[KeyLevelV2_5] = Field(default_factory=list, description="A list of identified major structural walls (often from strong SGDHP/UGCH scores).")
    timestamp: datetime = Field(..., description="Timestamp of when this set of key levels was generated.")

    class Config:
        extra = 'forbid' # Internal model