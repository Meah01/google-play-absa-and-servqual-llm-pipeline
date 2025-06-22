"""
Simple dashboard launcher as alternative to main.py.
Use this if main.py has issues starting the dashboard.
"""

import sys
import subprocess
import time
from pathlib import Path


def launch_dashboard_simple():
    """Simple dashboard launcher."""
    project_root = Path(__file__).parent
    dashboard_path = project_root / "dashboard_app.py"

    print("[START] Starting dashboard with simple launcher...")
    print(f"[FILE] Dashboard: {dashboard_path}")

    if not dashboard_path.exists():
        print("[ERROR] Dashboard file not found!")
        return False

    try:
        # Simple streamlit run command
        cmd = [
            "streamlit", "run", str(dashboard_path),
            "--server.port", "8501"
        ]

        print(f"[CMD] Running: {' '.join(cmd)}")
        print("[INFO] This will block the terminal. Press Ctrl+C to stop.")
        print("[URL] Dashboard should be at: http://localhost:8501")
        print("\n" + "=" * 50)

        # Run streamlit (this will block)
        subprocess.run(cmd, cwd=project_root)

    except KeyboardInterrupt:
        print("\n[STOP] Dashboard stopped by user")
    except FileNotFoundError:
        print("[ERROR] 'streamlit' command not found")
        print("[FIX] Install with: pip install streamlit")
        print("[ALT] Or try: python -m streamlit run dashboard_app.py")
    except Exception as e:
        print(f"[ERROR] Failed to start dashboard: {e}")


def try_alternative_methods():
    """Try alternative ways to start the dashboard."""
    project_root = Path(__file__).parent
    dashboard_path = project_root / "dashboard_app.py"

    methods = [
        # Method 1: Direct streamlit command
        ["streamlit", "run", str(dashboard_path), "--server.port", "8501"],

        # Method 2: Python module
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path), "--server.port", "8501"],

        # Method 3: Different port
        ["streamlit", "run", str(dashboard_path), "--server.port", "8502"],

        # Method 4: Local only
        ["streamlit", "run", str(dashboard_path), "--server.port", "8501", "--server.address", "127.0.0.1"]
    ]

    for i, cmd in enumerate(methods, 1):
        print(f"\n[METHOD {i}] Trying: {' '.join(cmd)}")

        try:
            # Test if command exists
            result = subprocess.run(
                cmd + ["--help"],
                capture_output=True,
                timeout=5
            )

            if result.returncode == 0:
                print(f"[OK] Method {i} command is valid")

                # Ask user if they want to try this method
                try:
                    choice = input(f"Try method {i}? (y/n): ").lower().strip()
                    if choice == 'y':
                        print(f"[START] Starting with method {i}...")
                        print("[INFO] Press Ctrl+C to stop")
                        subprocess.run(cmd, cwd=project_root)
                        return True
                except KeyboardInterrupt:
                    print("\n[SKIP] Skipping this method")
                    continue
            else:
                print(f"[FAIL] Method {i} command failed")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"[FAIL] Method {i} command not found")
        except KeyboardInterrupt:
            print(f"\n[STOP] Interrupted method {i}")
            return False
        except Exception as e:
            print(f"[ERROR] Method {i} error: {e}")

    return False


def main():
    """Main launcher function."""
    print("Simple Dashboard Launcher")
    print("=" * 30)

    choice = input("Choose option:\n1. Simple launch\n2. Try alternative methods\n3. Exit\nChoice (1-3): ").strip()

    if choice == "1":
        launch_dashboard_simple()
    elif choice == "2":
        try_alternative_methods()
    elif choice == "3":
        print("[EXIT] Goodbye!")
    else:
        print("[ERROR] Invalid choice")


if __name__ == "__main__":
    main()