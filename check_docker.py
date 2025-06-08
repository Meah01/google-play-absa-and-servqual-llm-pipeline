"""
Quick Docker infrastructure check and startup script.
"""

import subprocess
import time
import sys
from pathlib import Path


def check_docker_running():
    """Check if Docker Desktop is running."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print("[OK] Docker is running")
            return True
        else:
            print("[ERROR] Docker not responding")
            return False

    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("[ERROR] Docker not found or not running")
        print("[INFO] Please ensure Docker Desktop is installed and running")
        return False


def check_docker_compose_services():
    """Check if docker-compose services are running."""
    try:
        result = subprocess.run(
            ["docker-compose", "ps"],
            capture_output=True,
            text=True,
            timeout=10
        )

        print("[INFO] Docker Compose Status:")
        print(result.stdout)

        # Check if services are running
        if "Up" in result.stdout:
            print("[OK] Some services are running")
            return True
        else:
            print("[WARN] No services appear to be running")
            return False

    except Exception as e:
        print(f"[ERROR] Error checking docker-compose: {e}")
        return False


def start_docker_services():
    """Start Docker services."""
    print("[START] Starting Docker services...")

    try:
        # Start services
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print("[OK] Docker services started")
            print("[WAIT] Waiting 15 seconds for services to initialize...")
            time.sleep(15)
            return True
        else:
            print(f"[ERROR] Failed to start services: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("[ERROR] Service startup timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Error starting services: {e}")
        return False


def test_database_connection():
    """Test if PostgreSQL is accessible."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="absa_pipeline",
            user="absa_user",
            password="absa_password"
        )

        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()

        print("[OK] PostgreSQL connection successful")
        return True

    except Exception as e:
        print(f"[ERROR] PostgreSQL connection failed: {e}")
        return False


def test_redis_connection():
    """Test if Redis is accessible."""
    try:
        import redis

        client = redis.Redis(host='localhost', port=6379, db=0)
        client.ping()
        client.close()

        print("[OK] Redis connection successful")
        return True

    except Exception as e:
        print(f"[ERROR] Redis connection failed: {e}")
        return False


def main():
    """Main diagnostic and setup function."""
    print("Docker Infrastructure Diagnostic")
    print("=" * 40)

    # Step 1: Check Docker
    if not check_docker_running():
        print("\n[ACTION NEEDED] Please:")
        print("1. Install Docker Desktop if not installed")
        print("2. Start Docker Desktop")
        print("3. Wait for Docker to fully start")
        print("4. Run this script again")
        return False

    # Step 2: Check services
    services_running = check_docker_compose_services()

    # Step 3: Start services if needed
    if not services_running:
        print("\n[ACTION] Starting Docker services...")
        if not start_docker_services():
            print("\n[FAILED] Could not start Docker services")
            return False

        # Check again
        check_docker_compose_services()

    # Step 4: Test connections
    print("\n[TEST] Testing service connections...")

    db_ok = test_database_connection()
    redis_ok = test_redis_connection()

    if db_ok and redis_ok:
        print("\n[SUCCESS] All services are ready!")
        print("\n[NEXT] You can now run:")
        print("   python main.py run")
        return True
    else:
        print("\n[ISSUE] Some services are not responding")
        print("[INFO] Try waiting a bit longer and running:")
        print("   python check_docker.py")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)