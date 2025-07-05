#!/usr/bin/env python
# compliance_decorators_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED, CANONICAL IMPLEMENTATION

"""
Elite Options Trading System v2.5 Compliance Decorators

This module provides decorators for ensuring compliance, validation, error handling,
and performance monitoring throughout the Elite Options Trading System.

Key decorators:
- validate_data: Ensures data meets specified validation criteria using Pydantic models
- track_performance: Monitors and logs execution time and resource usage
- error_handler: Provides standardized error handling and recovery
- audit_log: Records function calls and parameters for audit purposes
- rate_limit: Enforces rate limits on function calls
- compliance_check: Ensures operations comply with trading regulations
- retry: Implements retry logic with exponential backoff
- cache_result: Caches function results for improved performance
- deprecation_warning: Marks functions as deprecated with migration guidance
"""

import functools
import inspect
import json
import logging
import time
import traceback
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import psutil
from filelock import FileLock
from pydantic import BaseModel, ValidationError

# Configure logger
logger = logging.getLogger(__name__)

# Define compliance levels for different operations
class ComplianceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Define performance metrics tracking structure
class PerformanceMetrics(BaseModel):
    """Performance metrics tracking structure."""
    function_name: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_percent: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

# Define audit log entry structure
class AuditLogEntry(BaseModel):
    """Audit log entry structure."""
    function_name: str
    timestamp: datetime
    user_id: Optional[str] = None
    args: List[Any]
    kwargs: Dict[str, Any]
    result_status: str
    execution_time_ms: float
    trace_id: str

# Validation decorator
def validate_data(model: Type[BaseModel], field_name: str = 'data'):
    """
    Validates function input data against a Pydantic model.
    
    Args:
        model: The Pydantic model class to validate against
        field_name: The parameter name containing the data to validate
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract data to validate
            data = kwargs.get(field_name)
            if data is None and args:
                # Check if data is in positional arguments
                sig = inspect.signature(func)
                parameters = list(sig.parameters.keys())
                if field_name in parameters:
                    idx = parameters.index(field_name)
                    if idx < len(args):
                        data = args[idx]
            
            # Validate data
            if data is not None:
                try:
                    if isinstance(data, dict):
                        validated_data = model(**data)
                    elif isinstance(data, model):
                        validated_data = data
                    else:
                        raise ValueError(f"Data must be a dict or {model.__name__} instance")
                    
                    # Replace with validated data
                    if field_name in kwargs:
                        kwargs[field_name] = validated_data
                    else:
                        args_list = list(args)
                        args_list[idx] = validated_data
                        args = tuple(args_list)
                        
                except ValidationError as e:
                    logger.error(f"Validation error in {func.__name__}: {e}")
                    raise ValueError(f"Data validation failed: {e}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Compliance tracking decorator (alias for track_performance)
def track_compliance(component_id: str, component_name: str):
    """
    Tracks compliance for dashboard components (alias for track_performance).

    Args:
        component_id: Unique identifier for the component
        component_name: Human-readable name for the component

    Returns:
        Decorated function with performance tracking
    """
    return track_performance()

# Performance tracking decorator
def track_performance(metrics_file: Optional[str] = None):
    """
    Tracks and logs performance metrics for the decorated function.
    
    Args:
        metrics_file: Optional file path to save metrics (in addition to logging)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            success = True
            error_message = None
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                logger.error(f"Error in {func.__name__}: {e}")
                raise
            finally:
                end_time = time.time()
                execution_time_ms = (end_time - start_time) * 1000
                end_memory = process.memory_info().rss / (1024 * 1024)  # MB
                memory_usage_mb = end_memory - start_memory
                cpu_percent = process.cpu_percent()
                
                # Log performance metrics
                metrics = PerformanceMetrics(
                    function_name=func.__name__,
                    execution_time_ms=execution_time_ms,
                    memory_usage_mb=memory_usage_mb,
                    cpu_percent=cpu_percent,
                    timestamp=datetime.now(),
                    success=success,
                    error_message=error_message
                )
                
                logger.info(f"Performance metrics for {func.__name__}: "
                           f"time={metrics.execution_time_ms:.2f}ms, "
                           f"memory={metrics.memory_usage_mb:.2f}MB, "
                           f"CPU={metrics.cpu_percent:.1f}%")
                
                # Save metrics to file if specified
                if metrics_file:
                    try:
                        metrics_path = Path(metrics_file)
                        metrics_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with FileLock(f"{metrics_file}.lock"):
                            with open(metrics_file, 'a') as f:
                                f.write(metrics.json() + '\n')
                    except Exception as e:
                        logger.error(f"Failed to save metrics to file: {e}")
                
        return wrapper
    return decorator

# Error handling decorator
def error_handler(retry_count: int = 0, retry_delay: float = 1.0, 
                 fallback_function: Optional[Callable] = None):
    """
    Provides standardized error handling with optional retry logic.
    
    Args:
        retry_count: Number of retries on failure
        retry_delay: Delay between retries in seconds
        fallback_function: Function to call if all retries fail
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts <= retry_count:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    logger.warning(
                        f"Error in {func.__name__} (attempt {attempts}/{retry_count+1}): {e}"
                    )
                    
                    if attempts <= retry_count:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"All {retry_count+1} attempts failed for {func.__name__}: {e}\n"
                            f"Traceback: {traceback.format_exc()}"
                        )
                        
                        if fallback_function:
                            logger.info(f"Executing fallback function: {fallback_function.__name__}")
                            return fallback_function(*args, **kwargs)
                        
                        raise
        return wrapper
    return decorator

# Audit logging decorator
def audit_log(log_file: Optional[str] = None, log_args: bool = True, 
             log_result: bool = False, compliance_level: ComplianceLevel = ComplianceLevel.MEDIUM):
    """
    Records function calls and parameters for audit purposes.
    
    Args:
        log_file: Optional file path to save audit logs
        log_args: Whether to log function arguments
        log_result: Whether to log function results
        compliance_level: Compliance level for the operation
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Sanitize args and kwargs for logging
            safe_args = []
            if log_args:
                for arg in args:
                    if isinstance(arg, (str, int, float, bool, list, dict)):
                        safe_args.append(arg)
                    else:
                        safe_args.append(f"{type(arg).__name__} instance")
                
                safe_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, (str, int, float, bool, list, dict)):
                        safe_kwargs.append(v)
                    else:
                        safe_kwargs[k] = f"{type(v).__name__} instance"
            else:
                safe_args = ["<args not logged>"]
                safe_kwargs = {"<kwargs>": "not logged"}
            
            # Log function call
            logger.info(
                f"[AUDIT:{compliance_level.value}] Function call: {func.__name__}, "
                f"TraceID: {trace_id}"
            )
            
            result = None
            status = "SUCCESS"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = f"ERROR: {type(e).__name__}"
                logger.error(
                    f"[AUDIT:{compliance_level.value}] Error in {func.__name__}: {e}, "
                    f"TraceID: {trace_id}"
                )
                raise
            finally:
                end_time = time.time()
                execution_time_ms = (end_time - start_time) * 1000
                
                # Create audit log entry
                entry = AuditLogEntry(
                    function_name=func.__name__,
                    timestamp=datetime.now(),
                    user_id=kwargs.get('user_id'),
                    args=safe_args,
                    kwargs=safe_kwargs,
                    result_status=status,
                    execution_time_ms=execution_time_ms,
                    trace_id=trace_id
                )
                
                # Log result if requested
                if log_result and result is not None:
                    if isinstance(result, (str, int, float, bool)):
                        logger.info(f"[AUDIT:{compliance_level.value}] Result: {result}")
                    elif isinstance(result, (list, dict)):
                        try:
                            logger.info(
                                f"[AUDIT:{compliance_level.value}] Result: "
                                f"{json.dumps(result, default=str)[:1000]}"
                            )
                        except:
                            logger.info(f"[AUDIT:{compliance_level.value}] Result: <complex object>")
                    else:
                        logger.info(
                            f"[AUDIT:{compliance_level.value}] Result type: {type(result).__name__}"
                        )
                
                # Save to audit log file if specified
                if log_file:
                    try:
                        log_path = Path(log_file)
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with FileLock(f"{log_file}.lock"):
                            with open(log_file, 'a') as f:
                                f.write(entry.json() + '\n')
                    except Exception as e:
                        logger.error(f"Failed to save audit log to file: {e}")
                
        return wrapper
    return decorator

# Rate limiting decorator
def rate_limit(max_calls: int, time_period: float = 60.0):
    """
    Enforces rate limits on function calls.
    
    Args:
        max_calls: Maximum number of calls allowed in the time period
        time_period: Time period in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        # Store call history per function
        call_history = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Remove expired calls from history
            nonlocal call_history
            call_history = [t for t in call_history if current_time - t < time_period]
            
            # Check if rate limit exceeded
            if len(call_history) >= max_calls:
                wait_time = time_period - (current_time - call_history[0])
                logger.warning(
                    f"Rate limit exceeded for {func.__name__}. "
                    f"Maximum {max_calls} calls per {time_period} seconds. "
                    f"Please wait {wait_time:.2f} seconds."
                )
                raise Exception(
                    f"Rate limit exceeded. Please wait {wait_time:.2f} seconds."
                )
            
            # Add current call to history
            call_history.append(current_time)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Compliance check decorator
def compliance_check(rules: List[Callable], level: ComplianceLevel = ComplianceLevel.MEDIUM):
    """
    Ensures operations comply with trading regulations and business rules.
    
    Args:
        rules: List of rule functions that return (passed, message)
        level: Compliance level for the operation
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Apply all compliance rules
            for rule_func in rules:
                passed, message = rule_func(*args, **kwargs)
                if not passed:
                    logger.error(
                        f"[COMPLIANCE:{level.value}] Rule violation in {func.__name__}: {message}"
                    )
                    raise ValueError(f"Compliance check failed: {message}")
            
            logger.info(
                f"[COMPLIANCE:{level.value}] All compliance checks passed for {func.__name__}"
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Retry decorator with exponential backoff
def retry(max_retries: int = 3, backoff_factor: float = 2.0, 
         exceptions: tuple = (Exception,)):
    """
    Implements retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Factor to increase delay between retries
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(
                            f"All {max_retries} retry attempts failed for {func.__name__}: {e}"
                        )
                        raise
                    
                    delay = backoff_factor ** (retry_count - 1)
                    logger.warning(
                        f"Retry {retry_count}/{max_retries} for {func.__name__} after error: {e}. "
                        f"Waiting {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
        return wrapper
    return decorator

# Result caching decorator
def cache_result(ttl: float = 300.0, max_size: int = 100):
    """
    Caches function results for improved performance.
    
    Args:
        ttl: Time-to-live for cached results in seconds
        max_size: Maximum cache size
        
    Returns:
        Decorated function
    """
    def decorator(func):
        cache = {}  # {key: (result, timestamp)}
        cache_keys = []  # For LRU tracking
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function args and kwargs
            key_parts = [func.__name__]
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))
                else:
                    key_parts.append(str(id(arg)))
            
            for k, v in sorted(kwargs.items()):
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}:{v}")
                else:
                    key_parts.append(f"{k}:{id(v)}")
            
            cache_key = "|".join(key_parts)
            current_time = time.time()
            
            # Check if result is in cache and not expired
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if current_time - timestamp < ttl:
                    # Move key to end of LRU list
                    cache_keys.remove(cache_key)
                    cache_keys.append(cache_key)
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = (result, current_time)
            cache_keys.append(cache_key)
            
            # Enforce max cache size (LRU eviction)
            if len(cache_keys) > max_size:
                oldest_key = cache_keys.pop(0)
                del cache[oldest_key]
            
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            return result
        
        # Add cache management methods
        def clear_cache():
            nonlocal cache, cache_keys
            cache = {}
            cache_keys = []
            logger.info(f"Cache cleared for {func.__name__}")
        
        wrapper.clear_cache = clear_cache
        
        return wrapper
    return decorator

# Deprecation warning decorator
def deprecation_warning(message: str, removal_version: str):
    """
    Marks functions as deprecated with migration guidance.
    
    Args:
        message: Migration guidance message
        removal_version: Version when the function will be removed
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.warning(
                f"DEPRECATED: {func.__name__} is deprecated and will be removed in "
                f"version {removal_version}. {message}"
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Transaction logging decorator for financial operations
def transaction_log(operation_type: str, log_file: Optional[str] = None):
    """
    Records detailed logs for financial transactions.
    
    Args:
        operation_type: Type of financial operation
        log_file: Optional file path to save transaction logs
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            transaction_id = str(uuid.uuid4())
            start_time = time.time()
            
            logger.info(
                f"[TRANSACTION:{transaction_id}] Starting {operation_type} operation in {func.__name__}"
            )
            
            result = None
            status = "COMPLETED"
            try:
                result = func(*args, **kwargs)
                logger.info(
                    f"[TRANSACTION:{transaction_id}] Successfully completed {operation_type} "
                    f"operation in {func.__name__}"
                )
                return result
            except Exception as e:
                status = f"FAILED: {type(e).__name__}"
                logger.error(
                    f"[TRANSACTION:{transaction_id}] Failed {operation_type} operation in "
                    f"{func.__name__}: {e}"
                )
                raise
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Log transaction details
                transaction_data = {
                    "transaction_id": transaction_id,
                    "operation_type": operation_type,
                    "function": func.__name__,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": execution_time,
                    "status": status,
                    "parameters": str(kwargs)
                }
                
                if log_file:
                    try:
                        log_path = Path(log_file)
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with FileLock(f"{log_file}.lock"):
                            with open(log_file, 'a') as f:
                                f.write(json.dumps(transaction_data) + '\n')
                    except Exception as e:
                        logger.error(f"Failed to save transaction log to file: {e}")
                
        return wrapper
    return decorator

# Data sanitization decorator
def sanitize_data(fields_to_sanitize: List[str]):
    """
    Sanitizes sensitive data in function arguments.
    
    Args:
        fields_to_sanitize: List of parameter names to sanitize
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create sanitized kwargs
            sanitized_kwargs = kwargs.copy()
            for field in fields_to_sanitize:
                if field in sanitized_kwargs:
                    if isinstance(sanitized_kwargs[field], str):
                        sanitized_kwargs[field] = "****"
                    elif isinstance(sanitized_kwargs[field], dict) and "password" in sanitized_kwargs[field]:
                        sanitized_kwargs[field]["password"] = "****"
            
            # Log sanitized function call
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **sanitized_kwargs)
            bound_args.apply_defaults()
            
            logger.info(f"Calling {func.__name__} with sanitized parameters: {bound_args.arguments}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
