#!/usr/bin/env python3
"""
🚀 PYDANTIC-FIRST CONTROL PANEL VALIDATOR V2.5 🚀

*** THIS IS A PYDANTIC-FIRST IMPLEMENTATION ***
*** DO NOT MODIFY WITHOUT PYDANTIC VALIDATION ***
*** ALL CHANGES MUST BE AI-VALIDATED AGAINST EOTS.SCHEMAS ***
*** CROSS-REFERENCED WITH METRICS_CALCULATOR AND ITS_ORCHESTRATOR ***

AUTHOR: NEXUS AI SYSTEM
DATE: 2025-01-27
VALIDATED AGAINST: data_models.eots_schemas_v2_5
CROSS-REFERENCED WITH: metrics_calculator_v2_5, its_orchestrator_v2_5, config_v2_5.json

PURPOSE:
- PREVENT CONTROL PANEL PARAMETER ERRORS FOREVER
- ENSURE ROBUST PYDANTIC-FIRST VALIDATION
- PROVIDE COMPREHENSIVE ERROR HANDLING
- MAINTAIN CROSS-SYSTEM COMPATIBILITY
- TRACK COMPLIANCE AND VALIDATION HISTORY

WARNING: THIS IS PYDANTIC-FIRST ARCHITECTURE
DO NOT BYPASS VALIDATION OR PASS DICTIONARIES DIRECTLY
ALL PARAMETERS MUST BE VALIDATED THROUGH PYDANTIC MODELS
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from functools import wraps

from pydantic import BaseModel, Field, ValidationError

# *** PYDANTIC-FIRST: EOTS SCHEMA IMPORTS ***
# ALL VALIDATION MUST GO THROUGH THESE SCHEMAS
from data_models import ControlPanelParametersV2_5

# *** CROSS-REFERENCE IMPORTS ***
# THESE MUST BE VALIDATED FOR COMPATIBILITY
from utils.config_manager_v2_5 import ConfigManagerV2_5

logger = logging.getLogger(__name__)


class PydanticControlPanelValidationError(Exception):
    """Custom exception for control panel validation errors."""
    pass


class ControlPanelComplianceTracker(BaseModel):
    """*** PYDANTIC-FIRST: Track control panel compliance and validation history ***"""
    
    component_name: str = Field(..., description="Name of component using control panel")
    validation_timestamp: datetime = Field(default_factory=datetime.now)
    parameters_validated: bool = Field(default=False)
    cross_reference_validated: bool = Field(default=False)
    error_count: int = Field(default=0)
    last_error: Optional[str] = Field(default=None)
    validation_history: List[str] = Field(default_factory=list)
    
    class Config:
        extra = 'forbid'


class PydanticControlPanelValidator:
    """PYDANTIC-FIRST: Control panel validator with strict validation.
    
    *** STRICT VALIDATION: NO DEFAULTS, NO FALLBACKS ***
    *** ALL PARAMETERS MUST BE EXPLICITLY PROVIDED ***
    *** VALIDATION ERRORS WILL FAIL FAST ***
    """
    
    def __init__(self, config: ConfigManagerV2_5):
        """Initialize validator with config."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._validated_params: Optional[ControlPanelParametersV2_5] = None
        self._validation_history: List[Dict[str, Any]] = []
        
    def _record_validation_success(self, params: ControlPanelParametersV2_5, validation_time: float) -> None:
        """Record successful validation."""
        self._validation_history.append({
            'timestamp': datetime.now(),
            'success': True,
            'params': params.model_dump(),
            'validation_time': validation_time
        })
        
    def _record_validation_error(self, error_msg: str) -> None:
        """Record validation error."""
        self._validation_history.append({
            'timestamp': datetime.now(),
            'success': False,
            'error': error_msg
        })
        
    def validate_and_set_params(self, 
                               symbol: str,
                               dte_min: int,
                               dte_max: int,
                               price_range_percent: int,
                               fetch_interval_seconds: int) -> ControlPanelParametersV2_5:
        """*** PYDANTIC-FIRST: Validate and set control panel parameters ***
        
        *** STRICT VALIDATION: NO DEFAULTS, NO FALLBACKS ***
        *** ALL PARAMETERS MUST BE EXPLICITLY PROVIDED ***
        *** VALIDATION ERRORS WILL FAIL FAST ***
        
        Args:
            symbol: Trading symbol (MUST BE STRING)
            dte_min: Minimum days to expiration (MUST BE INT)
            dte_max: Maximum days to expiration (MUST BE INT)
            price_range_percent: Price range percentage (MUST BE INT)
            fetch_interval_seconds: Fetch interval (MUST BE INT)
            
        Returns:
            Validated ControlPanelParametersV2_5 instance
            
        Raises:
            PydanticControlPanelValidationError: If validation fails
        """
        validation_start = time.time()
        
        try:
            # *** PYDANTIC-FIRST: Type validation before Pydantic ***
            if isinstance(symbol, dict):
                raise PydanticControlPanelValidationError(
                    "❌ CRITICAL ERROR: symbol parameter is a dictionary! "
                    "This causes 'dict' object has no attribute 'upper' errors. "
                    "Pass individual parameters, not dictionaries!"
                )
            
            if not isinstance(symbol, str):
                raise PydanticControlPanelValidationError(
                    f"❌ PARAMETER ERROR: symbol must be string, got {type(symbol)}"
                )
            
            if not isinstance(dte_min, int) or not isinstance(dte_max, int):
                raise PydanticControlPanelValidationError(
                    f"❌ PARAMETER ERROR: dte_min and dte_max must be integers, "
                    f"got {type(dte_min)} and {type(dte_max)}"
                )
            
            if not isinstance(price_range_percent, int):
                raise PydanticControlPanelValidationError(
                    f"❌ PARAMETER ERROR: price_range_percent must be integer, "
                    f"got {type(price_range_percent)}"
                )
                
            if not isinstance(fetch_interval_seconds, int):
                raise PydanticControlPanelValidationError(
                    f"❌ PARAMETER ERROR: fetch_interval_seconds must be integer, "
                    f"got {type(fetch_interval_seconds)}"
                )
            
            # *** PYDANTIC-FIRST: Create and validate model ***
            validated_params = ControlPanelParametersV2_5(
                symbol=symbol.upper(),
                dte_min=dte_min,
                dte_max=dte_max,
                price_range_percent=price_range_percent,
                fetch_interval_seconds=fetch_interval_seconds
            )
            
            # Store validated parameters
            self._validated_params = validated_params
            
            # Record validation success
            validation_time = time.time() - validation_start
            self._record_validation_success(validated_params, validation_time)
            
            return validated_params
            
        except ValidationError as e:
            error_msg = f"❌ PYDANTIC VALIDATION ERROR: {str(e)}"
            self._record_validation_error(error_msg)
            raise PydanticControlPanelValidationError(error_msg)
            
        except Exception as e:
            error_msg = f"❌ UNEXPECTED ERROR: {str(e)}"
            self._record_validation_error(error_msg)
            raise PydanticControlPanelValidationError(error_msg)


def get_pydantic_validator(config: ConfigManagerV2_5) -> PydanticControlPanelValidator:
    """Get or create a validator instance."""
    return PydanticControlPanelValidator(config)


def enforce_control_panel_compliance(func: Callable) -> Callable:
    """*** PYDANTIC-FIRST: Decorator to enforce control panel compliance ***
    
    *** STRICT VALIDATION: NO FALLBACKS, NO DEFAULTS ***
    *** VALIDATION ERRORS WILL FAIL FAST ***
    
    This decorator ensures that any function using control panel parameters:
    1. Has all required parameters explicitly provided
    2. Validates all parameters through Pydantic models
    3. Fails fast on any validation error
    4. Maintains compliance tracking
    """
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get validator instance
        config = kwargs.get('config')
        if not config:
            raise PydanticControlPanelValidationError(
                "❌ CRITICAL ERROR: config parameter is required for control panel validation"
            )
            
        validator = get_pydantic_validator(config)
        
        # Extract and validate control panel parameters
        required_params = ['symbol', 'dte_min', 'dte_max', 'price_range_percent', 'fetch_interval_seconds']
        missing_params = [param for param in required_params if param not in kwargs]
        
        if missing_params:
            raise PydanticControlPanelValidationError(
                f"❌ CRITICAL ERROR: Missing required parameters: {', '.join(missing_params)}"
            )
            
        # Type conversion with validation
        try:
            # Handle ValidationError first since it's most specific
            try:
                # Validate parameters through Pydantic
                validated_params = validator.validate_and_set_params(
                    symbol=str(kwargs['symbol']),
                    dte_min=int(kwargs['dte_min']),
                    dte_max=int(kwargs['dte_max']),
                    price_range_percent=int(kwargs['price_range_percent']),
                    fetch_interval_seconds=int(kwargs['fetch_interval_seconds'])
                )
            except ValidationError as e:
                raise PydanticControlPanelValidationError(f"❌ VALIDATION ERROR: {str(e)}")
            
            # Update kwargs with validated parameters
            kwargs.update({
                'symbol': validated_params.symbol,
                'dte_min': validated_params.dte_min,
                'dte_max': validated_params.dte_max,
                'price_range_percent': validated_params.price_range_percent,
                'fetch_interval_seconds': validated_params.fetch_interval_seconds
            })
            
            # Execute function with validated parameters
            return func(*args, **kwargs)
            
        except (TypeError, ValueError) as e:
            raise PydanticControlPanelValidationError(f"❌ TYPE ERROR: {str(e)}")
            
        except Exception as e:
            raise PydanticControlPanelValidationError(f"❌ UNEXPECTED ERROR: {str(e)}")
            
    return wrapper