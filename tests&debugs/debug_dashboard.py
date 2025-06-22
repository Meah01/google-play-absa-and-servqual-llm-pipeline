"""
Dashboard debugging script to identify Streamlit issues.
"""

import subprocess
import sys
import time
import socket
from pathlib import Path
import requests

def check_port_availability(port):
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            print(f"[OK] Port {port} is available")
            return True
    except OSError:
        print(f"[BUSY] Port {port} is already in use")
        return False

def check_streamlit_installed():
    """Check if Streamlit is properly installed."""
    try:
        result = subprocess.run([sys.executable, "-c", "import streamlit; print(streamlit.__version__)"],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"[OK] Streamlit installed: version {version}")
            return True
        else:
            print("[ERROR] Streamlit import failed")
            return False
    except Exception as e:
        print(f"[ERROR] Error checking Streamlit: {e}")
        return False

def test_dashboard_file():
    """Test if dashboard file is accessible and has no syntax errors."""
    dashboard_path = Path("dashboard_app.py")

    if not dashboard_path.exists():
        print(f"[ERROR] Dashboard file not found: {dashboard_path}")
        return False

    print(f"[OK] Dashboard file exists: {dashboard_path}")

    # Test syntax
    try:
        result = subprocess.run([sys.executable, "-m", "py_compile", str(dashboard_path)],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("[OK] Dashboard file syntax is valid")
            return True
        else:
            print(f"[ERROR] Dashboard syntax error: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] Error checking dashboard syntax: {e}")
        return False

def find_running_streamlit():
    """Find any running Streamlit processes."""
    try:
        # Check for Streamlit processes
        result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"],
                              capture_output=True, text=True, timeout=10)

        if "python.exe" in result.stdout:
            print("[INFO] Python processes found:")
            lines = result.stdout.split('\n')
            for line in lines:
                if "python.exe" in line:
                    print(f"   {line.strip()}")

        # Try to find processes using port 8501
        result = subprocess.run(["netstat", "-ano", "|", "findstr", ":8501"],
                              shell=True, capture_output=True, text=True, timeout=10)

        if result.stdout.strip():
            print("[INFO] Processes using port 8501:")
            print(result.stdout)
            return True
        else:
            print("[INFO] No processes found using port 8501")
            return False

    except Exception as e:
        print(f"[ERROR] Error finding processes: {e}")
        return False

def test_manual_streamlit():
    """Test starting Streamlit manually."""
    print("[TEST] Attempting to start Streamlit manually...")

    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", "dashboard_app.py",
            "--server.port", "8502",  # Use different port
            "--server.address", "127.0.0.1",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "true"
        ]

        print(f"[CMD] Running: {' '.join(cmd)}")

        # Start process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait a few seconds
        time.sleep(5)

        # Check if process is still running
        if process.poll() is None:
            print("[OK] Streamlit started successfully on port 8502")
            print("[INFO] Try accessing: http://localhost:8502")

            # Try to test the connection
            try:
                response = requests.get("http://localhost:8502", timeout=5)
                if response.status_code == 200:
                    print("[OK] Dashboard is accessible!")
                else:
                    print(f"[WARN] Dashboard responded with status: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"[WARN] Could not connect to dashboard: {e}")

            # Kill the test process
            process.terminate()
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"[ERROR] Streamlit failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False

    except Exception as e:
        print(f"[ERROR] Error testing manual Streamlit: {e}")
        return False

def check_network_connectivity():
    """Check basic network connectivity."""
    try:
        # Test localhost connectivity
        response = requests.get("http://localhost:8080", timeout=2)  # Adminer
        print("[OK] Localhost network is accessible (Adminer responding)")
        return True
    except requests.exceptions.RequestException:
        print("[WARN] Localhost network might have issues")

        # Test basic localhost
        try:
            socket.create_connection(("127.0.0.1", 80), timeout=2)
            print("[OK] Basic localhost connectivity works")
            return True
        except:
            print("[ERROR] Localhost connectivity issues")
            return False

def suggest_solutions():
    """Suggest potential solutions."""
    print("\n[SOLUTIONS] Potential fixes:")
    print("1. Try alternative port:")
    print("   python -m streamlit run dashboard_app.py --server.port 8502")
    print("   Then access: http://localhost:8502")
    print()
    print("2. Try alternative address:")
    print("   python -m streamlit run dashboard_app.py --server.address 127.0.0.1")
    print()
    print("3. Check Windows Firewall:")
    print("   - Go to Windows Defender Firewall")
    print("   - Allow an app through firewall")
    print("   - Add Python.exe")
    print()
    print("4. Try different browser:")
    print("   - Chrome/Firefox/Edge")
    print("   - Try incognito/private mode")
    print()
    print("5. Check antivirus:")
    print("   - Temporarily disable real-time protection")
    print("   - Add project folder to exclusions")

def main():
    """Run dashboard diagnostics."""
    print("Dashboard Diagnostic Tool")
    print("=" * 40)

    # Run diagnostics
    tests = [
        ("Port 8501 Available", lambda: check_port_availability(8501)),
        ("Streamlit Installed", check_streamlit_installed),
        ("Dashboard File Valid", test_dashboard_file),
        ("Find Running Processes", find_running_streamlit),
        ("Network Connectivity", check_network_connectivity),
        ("Manual Streamlit Test", test_manual_streamlit)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} failed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 40)
    print("Diagnostic Results:")
    for test_name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"   {status} {test_name}")

    # Suggestions
    suggest_solutions()

    return any(result for _, result in results)

if __name__ == "__main__":
    main()