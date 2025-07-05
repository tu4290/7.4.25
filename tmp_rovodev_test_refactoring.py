#!/usr/bin/env python3
"""
Test script to verify the refactored EOTS metrics module works correctly.
"""

import sys
import traceback

def test_basic_imports():
    """Test basic imports work correctly."""
    print("🧪 Testing basic imports...")
    try:
        from core_analytics_engine.eots_metrics import MetricsCalculatorV2_5
        print("✅ MetricsCalculatorV2_5 import successful")
        
        from core_analytics_engine.eots_metrics import CoreCalculator, FlowAnalytics, AdaptiveCalculator
        print("✅ Individual calculator imports successful")
        
        from core_analytics_engine.eots_metrics import ValidationUtils
        print("✅ ValidationUtils import successful")
        
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        traceback.print_exc()
        return False

def test_validation_utils():
    """Test ValidationUtils functionality."""
    print("\n🧪 Testing ValidationUtils functionality...")
    try:
        from core_analytics_engine.eots_metrics.validation_utils import ValidationUtils
        import pandas as pd
        
        validator = ValidationUtils()
        print("✅ ValidationUtils instantiated successfully")
        
        # Test validation methods exist
        methods = ['validate_input_data', 'validate_foundational_metrics', 'validate_elite_results', 'require_column_sum']
        for method in methods:
            if hasattr(validator, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        # Test simple validation
        test_df = pd.DataFrame({'test_col': [1, 2, 3]})
        result = validator.require_column_sum(test_df, 'test_col', 'Test Column')
        print(f"✅ Column sum validation works: {result}")
        
        return True
    except Exception as e:
        print(f"❌ ValidationUtils test failed: {e}")
        traceback.print_exc()
        return False

def test_composite_calculator():
    """Test composite calculator structure."""
    print("\n🧪 Testing composite calculator...")
    try:
        from core_analytics_engine.eots_metrics.metrics_calculator_composite import MetricsCalculatorV2_5
        
        # Check if expected methods exist
        expected_methods = [
            'calculate_all_metrics',
            '_calculate_flow_metric',
            '_calculate_avg_iv_at_strike',
            '_create_initial_model',
            '_calculate_flow_metrics'
        ]
        
        for method in expected_methods:
            if hasattr(MetricsCalculatorV2_5, method):
                print(f"✅ Method {method} exists in MetricsCalculatorV2_5")
            else:
                print(f"❌ Method {method} missing from MetricsCalculatorV2_5")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Composite calculator test failed: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test backward compatibility."""
    print("\n🧪 Testing backward compatibility...")
    try:
        from core_analytics_engine.eots_metrics import __all__
        print(f"✅ __all__ exports: {len(__all__)} items")
        
        expected_exports = [
            'MetricsCalculatorV2_5',
            'CoreCalculator',
            'FlowAnalytics',
            'AdaptiveCalculator',
            'VisualizationMetrics',
            'EliteImpactCalculator',
            'SupplementaryMetrics'
        ]
        
        for item in expected_exports:
            if item in __all__:
                print(f"✅ {item} properly exported")
            else:
                print(f"❌ {item} missing from exports")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all expected files exist."""
    print("\n🧪 Testing file structure...")
    try:
        import os
        
        expected_files = [
            'core_analytics_engine/eots_metrics/__init__.py',
            'core_analytics_engine/eots_metrics/metrics_calculator_composite.py',
            'core_analytics_engine/eots_metrics/validation_utils.py'
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"✅ File exists: {file_path}")
            else:
                print(f"❌ File missing: {file_path}")
                return False
        
        # Check file sizes
        init_size = os.path.getsize('core_analytics_engine/eots_metrics/__init__.py')
        composite_size = os.path.getsize('core_analytics_engine/eots_metrics/metrics_calculator_composite.py')
        validation_size = os.path.getsize('core_analytics_engine/eots_metrics/validation_utils.py')
        
        print(f"📊 File sizes:")
        print(f"  - __init__.py: {init_size} bytes")
        print(f"  - metrics_calculator_composite.py: {composite_size} bytes")
        print(f"  - validation_utils.py: {validation_size} bytes")
        
        if init_size < 5000:  # Should be much smaller now
            print("✅ __init__.py is appropriately sized (< 5KB)")
        else:
            print("⚠️ __init__.py might still be too large")
        
        return True
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting refactoring verification tests...\n")
    
    tests = [
        test_file_structure,
        test_basic_imports,
        test_validation_utils,
        test_composite_calculator,
        test_backward_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
            print("✅ PASSED\n")
        else:
            print("❌ FAILED\n")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Refactoring successful!")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)