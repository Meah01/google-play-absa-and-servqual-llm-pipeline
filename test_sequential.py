# CREATE A TEMPORARY TEST FILE: test_sequential.py

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline.sequential_processor import sequential_processor

def test_sequential():
    """Test sequential processing directly"""
    print("Testing sequential processing directly...")
    
    app_id = "com.amazon.mshop.android.shopping"
    print(f"Starting sequential processing for app: {app_id}")
    
    # Start sequential processing (new job, not resume)
    result = sequential_processor.start_sequential_processing(app_id, None)
    
    if result.success:
        print("SUCCESS!")
        print(f"Job ID: {result.job_id}")
        print(f"Reviews processed: {result.total_reviews_processed}")
    else:
        print(f"FAILED: {result.error_message}")

if __name__ == "__main__":
    test_sequential()