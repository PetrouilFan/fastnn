#!/usr/bin/env python3
"""
Test script to verify error handling improvements in fastnn.
This tests that the new error types are properly exported and can be used.
"""

import sys
import os

# Add the fastnn module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import fastnn
    print("✓ Successfully imported fastnn")
    
    # Test that error types are available
    print("\nTesting error type exports:")
    
    # Check if error module is accessible
    if hasattr(fastnn, 'error'):
        print("✓ fastnn.error module is accessible")
    else:
        print("✗ fastnn.error module is not accessible")
        sys.exit(1)
    
    # Check if FastnnError is available
    if hasattr(fastnn.error, 'FastnnError'):
        print("✓ FastnnError type is available")
        
        # Test creating different error types
        from fastnn.error import FastnnError
        
        shape_error = FastnnError.shape("Test shape error")
        print(f"✓ Shape error created: {shape_error}")
        
        overflow_error = FastnnError.overflow("Test overflow error")
        print(f"✓ Overflow error created: {overflow_error}")
        
        io_error = FastnnError.io("Test IO error")
        print(f"✓ IO error created: {io_error}")
        
        computation_error = FastnnError.computation("Test computation error")
        print(f"✓ Computation error created: {computation_error}")
        
    else:
        print("✗ FastnnError type is not available")
        sys.exit(1)
    
    print("\n✓ All error handling tests passed!")
    
except ImportError as e:
    print(f"✗ Failed to import fastnn: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
