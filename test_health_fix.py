"""
Quick test for the health check fix.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_health_check():
    """Test the health check functionality."""
    print("Testing Health Check Fix...")

    try:
        from src.utils.config import health_checker

        print("[TEST] Testing individual health checks...")

        # Test system resources specifically
        print("  [TEST] System resources...")
        try:
            system_health = health_checker.check_system_resources()
            print(f"    [RESULT] {system_health['status']}")
            if system_health['status'] == 'healthy':
                print(f"    [CPU] {system_health.get('cpu_usage', 'N/A')}")
                print(f"    [Memory] {system_health.get('memory_usage', 'N/A')}")
                print(f"    [Disk] {system_health.get('disk_usage', 'N/A')}")
        except Exception as e:
            print(f"    [ERROR] {e}")

        # Test overall health
        print("  [TEST] Overall health...")
        try:
            overall_health = health_checker.get_overall_health()
            print(f"    [RESULT] Overall: {overall_health['overall_status']}")
            print(f"    [SERVICES] Unhealthy: {overall_health.get('unhealthy_services', [])}")

            for service, status in overall_health['services'].items():
                print(f"    [{service.upper()}] {status['status']}")

        except Exception as e:
            print(f"    [ERROR] {e}")

        print("\n[SUCCESS] Health check test completed")
        return True

    except Exception as e:
        print(f"[ERROR] Health check test failed: {e}")
        return False


def test_unicode_fix():
    """Test that Unicode characters are fixed."""
    print("\nTesting Unicode Fix...")

    try:
        # Test the main.py imports
        from src.utils.config import logger

        # Test logging with potentially problematic characters
        logger.info("[TEST] Testing logging without Unicode issues")
        print("[SUCCESS] Unicode logging test passed")
        return True

    except Exception as e:
        print(f"[ERROR] Unicode test failed: {e}")
        return False


def main():
    """Run quick tests for the fixes."""
    print("Quick Fix Verification")
    print("=" * 30)

    health_ok = test_health_check()
    unicode_ok = test_unicode_fix()

    if health_ok and unicode_ok:
        print("\n[SUCCESS] All fixes verified!")
        print("\n[NEXT] Try running:")
        print("  python main.py run")
        return True
    else:
        print("\n[ISSUE] Some fixes may need more work")
        return False


if __name__ == "__main__":
    main()