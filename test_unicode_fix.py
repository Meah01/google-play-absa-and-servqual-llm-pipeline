"""
Quick test to verify Unicode encoding fix.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_main_status():
    """Test the main.py status command."""
    print("[TEST] Testing main.py status command...")

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "main.py", "status"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print("[OK] Status command executed successfully")
            print("Output preview:")
            print(result.stdout[:300] + "..." if len(result.stdout) > 300 else result.stdout)
            return True
        else:
            print(f"[ERROR] Status command failed with return code {result.returncode}")
            print("Error output:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("[ERROR] Status command timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Exception during status test: {e}")
        return False


def main():
    """Run the Unicode fix test."""
    print("Unicode Fix Test")
    print("=" * 30)

    success = test_main_status()

    if success:
        print("\n[SUCCESS] Unicode fix verified! Main.py should work properly now.")
        print("\n[NEXT] You can now run:")
        print("   python main.py run")
    else:
        print("\n[ISSUE] There may still be encoding issues.")
        print("[INFO] Try running in Windows Terminal or VS Code terminal instead of Command Prompt")

    return success


if __name__ == "__main__":
    main()