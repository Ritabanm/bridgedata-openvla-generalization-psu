#!/usr/bin/env python3
"""
Regenerate 500-sample baseline data with proper error handling
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# Add current directory to path to import openvla-baseline
sys.path.append('.')

def regenerate_baseline():
    """Regenerate 500-sample baseline data"""
    print("🔄 Regenerating 500-sample baseline data...")
    
    try:
        # Import and run the baseline evaluation
        from openvla_baseline import run_evaluation
        
        # Temporarily modify the max_samples in openvla_baseline
        import openvla_baseline
        
        # Backup original function
        original_load_sample_data = openvla_baseline.load_sample_data
        
        def load_500_samples(max_samples=3):
            """Load 500 samples instead of default"""
            return original_load_sample_data(max_samples=500)
        
        # Patch the function
        openvla_baseline.load_sample_data = load_500_samples
        
        # Run evaluation
        print("🚀 Running 500-sample baseline evaluation...")
        run_evaluation()
        
        # Restore original function
        openvla_baseline.load_sample_data = original_load_sample_data
        
        print("✅ 500-sample baseline data regenerated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error regenerating baseline data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    regenerate_baseline()
