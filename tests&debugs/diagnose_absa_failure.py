"""
Diagnose ABSA Processing Failure
Save as: diagnose_absa.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("🔍 ABSA Processing Failure Diagnostic")
    print("=" * 40)
    
    # Test 1: Basic imports
    print("\n📦 Testing Imports...")
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow: {tf.__version__}")
    except Exception as e:
        print(f"  ❌ TensorFlow: {e}")
    
    try:
        import keras
        print(f"  ✅ Keras: {keras.__version__}")
    except Exception as e:
        print(f"  ❌ Keras: {e}")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        print("  ✅ Transformers available")
    except Exception as e:
        print(f"  ❌ Transformers: {e}")
    
    # Test 2: ABSA engine import
    print("\n🧠 Testing ABSA Engine...")
    try:
        from src.absa.engine import ABSAEngine
        print("  ✅ ABSA Engine imported")
        
        # Try to initialize
        engine = ABSAEngine()
        print("  ✅ ABSA Engine initialized")
        
    except Exception as e:
        print(f"  ❌ ABSA Engine failed: {e}")
        return
    
    # Test 3: Test with sample review
    print("\n🧪 Testing ABSA with Sample Review...")
    try:
        sample_review = {
            'review_id': 'test-123',
            'app_id': 'com.test.app',
            'content': 'This app is great but crashes sometimes. Customer service was helpful.',
            'rating': 4
        }
        
        print(f"  Testing with: '{sample_review['content']}'")
        
        # Try deep analysis
        result = engine.analyze_review(
            sample_review['review_id'],
            sample_review['app_id'],
            sample_review['content']
            # Note: analyze_review doesn't take rating parameter
        )

        if result and hasattr(result, 'aspects'):
            print(f"  ✅ ABSA Analysis successful: {len(result.aspects)} aspects found")
            for aspect in result.aspects[:3]:  # Show first 3
                print(f"    - {aspect.aspect}: {aspect.sentiment_score:.2f} sentiment")
        else:
            print(f"  ❌ ABSA Analysis failed: No aspects found or invalid result")
        
    except Exception as e:
        print(f"  ❌ ABSA test failed: {e}")
        print(f"     Full error: {str(e)}")
    
    # Test 4: Check model files
    print("\n📁 Checking Model Files...")
    try:
        # Check if models directory exists
        models_dir = project_root / "models"
        if models_dir.exists():
            print(f"  ✅ Models directory exists: {models_dir}")
            model_files = list(models_dir.glob("*"))
            print(f"  📁 Found {len(model_files)} files/folders in models/")
            for file in model_files[:5]:  # Show first 5
                print(f"    - {file.name}")
        else:
            print("  ⚠️  Models directory not found")
    
    except Exception as e:
        print(f"  ❌ Model files check failed: {e}")
    
    # Test 5: Memory check
    print("\n💾 Memory Check...")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"  RAM: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available")
        print(f"  Usage: {memory.percent}%")
        
        if memory.percent > 80:
            print("  ⚠️  High memory usage - might cause ABSA failures")
        else:
            print("  ✅ Memory usage acceptable")
            
    except Exception as e:
        print(f"  ❌ Memory check failed: {e}")
    
    print("\n" + "=" * 40)
    print("🎯 ABSA DIAGNOSTIC COMPLETE")
    print("\nCommon ABSA failure causes:")
    print("1. TensorFlow/Keras version conflicts")
    print("2. Missing or corrupted model files")
    print("3. Memory exhaustion during model loading")
    print("4. Dependencies conflicts (transformers, torch)")
    print("5. Model download/cache issues")
    print("\nRecommended fixes:")
    print("- Use SERVQUAL-only processing for now")
    print("- Update TensorFlow/Keras to compatible versions")
    print("- Clear model cache and re-download")
    print("- Check memory usage during processing")

if __name__ == "__main__":
    main()
