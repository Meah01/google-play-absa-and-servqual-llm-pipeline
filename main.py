"""
Main entry point for ABSA Sentiment Pipeline.
Orchestrates infrastructure, data collection, and dashboard launch.
Phase 1 implementation with core workflow coordination.
"""

import os
import sys
import time
import logging
import subprocess
import argparse
from pathlib import Path
from typing import Optional, List
import asyncio
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import config, health_checker, logger
# Import storage and scraper only when needed to avoid connection errors


class ABSAPipelineOrchestrator:
    """Main orchestrator for the ABSA Sentiment Pipeline."""

    def __init__(self):
        self.logger = logging.getLogger("absa_pipeline.main")
        self.project_root = project_root

    def check_infrastructure(self) -> bool:
        """Check if Docker infrastructure is running."""
        try:
            result = subprocess.run(
                ["docker-compose", "ps", "-q"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                self.logger.info("[OK] Docker infrastructure is running")
                return True
            else:
                self.logger.warning("[WARN] Docker infrastructure not detected")
                return False

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.error(f"[ERROR] Error checking Docker infrastructure: {e}")
            return False

    def start_infrastructure(self) -> bool:
        """Start Docker infrastructure if not running."""
        self.logger.info("[START] Starting Docker infrastructure...")

        try:
            # Start services
            result = subprocess.run(
                ["docker-compose", "up", "-d"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                self.logger.info("[OK] Docker infrastructure started successfully")

                # Wait for services to be ready
                self.logger.info("[WAIT] Waiting for services to be ready...")
                time.sleep(10)

                # Verify health
                return self.wait_for_services()
            else:
                self.logger.error(f"[ERROR] Failed to start infrastructure: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("[ERROR] Infrastructure startup timed out")
            return False
        except Exception as e:
            self.logger.error(f"[ERROR] Error starting infrastructure: {e}")
            return False

    def wait_for_services(self, max_attempts: int = 12) -> bool:
        """Wait for all services to be healthy."""
        self.logger.info("[CHECK] Checking service health...")

        for attempt in range(max_attempts):
            try:
                health = health_checker.get_overall_health()

                if health['overall_status'] == 'healthy':
                    self.logger.info("[OK] All services are healthy")
                    return True
                else:
                    unhealthy = health.get('unhealthy_services', [])
                    self.logger.info(f"[WAIT] Waiting for services: {', '.join(unhealthy)} (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(5)

            except Exception as e:
                self.logger.warning(f"[WARN] Health check failed (attempt {attempt + 1}): {e}")
                time.sleep(5)

        self.logger.error("[ERROR] Services failed to become healthy within timeout")
        return False

    def run_sample_data_collection(self) -> bool:
        """Run sample data collection for demonstration."""
        self.logger.info("[DATA] Running sample data collection...")

        # Import scraper here to avoid connection issues
        from src.data.scraper import scrape_app_reviews

        sample_apps = [
            "com.instagram.android",
            "com.whatsapp",
            "com.spotify.music"
        ]

        try:
            results = []

            for app_id in sample_apps:
                self.logger.info(f"[SCRAPE] Scraping {app_id}...")

                result = scrape_app_reviews(app_id, count=20)  # Small sample for demo
                results.append(result)

                if result['success']:
                    stored = result['statistics']['stored']
                    self.logger.info(f"[OK] {app_id}: {stored} reviews stored")
                else:
                    self.logger.warning(f"[WARN] {app_id}: scraping failed")

                # Small delay between apps
                time.sleep(2)

            # Summary
            total_stored = sum(r['statistics']['stored'] for r in results if r['success'])
            successful_apps = sum(1 for r in results if r['success'])

            self.logger.info(f"[SUMMARY] Sample collection complete: {total_stored} reviews from {successful_apps} apps")
            return total_stored > 0

        except Exception as e:
            self.logger.error(f"[ERROR] Sample data collection failed: {e}")
            return False

    def launch_dashboard(self) -> bool:
        """Launch the Streamlit dashboard."""
        self.logger.info("[DASH] Launching dashboard...")

        try:
            dashboard_path = self.project_root / "dashboard_app.py"

            if not dashboard_path.exists():
                self.logger.error(f"[ERROR] Dashboard file not found: {dashboard_path}")
                return False

            # Launch Streamlit dashboard
            cmd = [
                "streamlit", "run", str(dashboard_path),
                "--server.port", str(config.dashboard.port),
                "--server.address", config.dashboard.host,
                "--browser.gatherUsageStats", "false"
            ]

            self.logger.info(f"[START] Starting dashboard on http://{config.dashboard.host}:{config.dashboard.port}")

            # Start dashboard in background
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Give it a moment to start
            time.sleep(3)

            # Check if process is still running
            if process.poll() is None:
                self.logger.info("[OK] Dashboard launched successfully")
                self.logger.info(f"[URL] Access dashboard at: http://localhost:{config.dashboard.port}")
                return True
            else:
                stdout, stderr = process.communicate()
                self.logger.error(f"[ERROR] Dashboard failed to start: {stderr.decode()}")
                return False

        except FileNotFoundError:
            self.logger.error("[ERROR] Streamlit not found. Install with: pip install streamlit")
            return False
        except Exception as e:
            self.logger.error(f"[ERROR] Error launching dashboard: {e}")
            return False

    def run_full_pipeline(self, skip_data_collection: bool = False) -> bool:
        """Run the complete pipeline workflow."""
        self.logger.info("[START] Starting ABSA Pipeline - Phase 1")

        # Step 1: Check/Start Infrastructure
        if not self.check_infrastructure():
            if not self.start_infrastructure():
                self.logger.error("[ERROR] Failed to start infrastructure")
                return False

        # Step 2: Verify storage connectivity
        try:
            from src.data.storage import storage
            health = storage.health_check()
            if health['overall'] != 'healthy':
                self.logger.error(f"[ERROR] Storage not healthy: {health}")
                return False
            self.logger.info("[OK] Storage connectivity verified")
        except Exception as e:
            self.logger.error(f"[ERROR] Storage check failed: {e}")
            return False

        # Step 3: Sample data collection (optional)
        if not skip_data_collection:
            if not self.run_sample_data_collection():
                self.logger.warning("[WARN] Sample data collection failed, but continuing...")
        else:
            self.logger.info("[SKIP] Skipping data collection")

        # Step 4: Launch dashboard
        if not self.launch_dashboard():
            self.logger.error("[ERROR] Failed to launch dashboard")
            return False

        # Success message
        self.logger.info("[SUCCESS] ABSA Pipeline Phase 1 launched successfully!")
        self.logger.info("[INFO] What's running:")
        self.logger.info("   - PostgreSQL database")
        self.logger.info("   - Redis cache")
        self.logger.info("   - Kafka streaming (ready for Phase 2)")
        self.logger.info("   - Streamlit dashboard")
        self.logger.info("   - Review scraping capability")

        self.logger.info(f"[DASHBOARD] Dashboard: http://localhost:{config.dashboard.port}")
        self.logger.info("[INFO] Press Ctrl+C to stop the pipeline")

        return True

    def stop_infrastructure(self) -> bool:
        """Stop Docker infrastructure."""
        self.logger.info("[STOP] Stopping infrastructure...")

        try:
            result = subprocess.run(
                ["docker-compose", "down"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                self.logger.info("[OK] Infrastructure stopped successfully")
                return True
            else:
                self.logger.error(f"[ERROR] Error stopping infrastructure: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"[ERROR] Error stopping infrastructure: {e}")
            return False

    def show_status(self) -> None:
        """Show current pipeline status."""
        print("\n[STATUS] ABSA Pipeline Status")
        print("=" * 50)

        # Infrastructure status
        if self.check_infrastructure():
            print("[INFRA] Infrastructure: Running")

            # Service health
            try:
                health = health_checker.get_overall_health()
                health_status = "Healthy" if health['overall_status'] == 'healthy' else "Issues"
                print(f"[HEALTH] Services: {health_status}")

                for service, status in health['services'].items():
                    status_icon = "[OK]" if status['status'] == 'healthy' else "[ERR]"
                    print(f"   {service.capitalize()}: {status_icon} {status['status']}")

            except Exception as e:
                print(f"[HEALTH] Services: Health check failed - {e}")
        else:
            print("[INFRA] Infrastructure: Not running")

        # Data status
        try:
            from src.data.storage import storage
            apps = storage.apps.get_all_apps()
            reviews = storage.reviews.get_reviews_for_processing(limit=10000)

            print(f"[DATA] Apps in database: {len(apps)}")
            print(f"[DATA] Reviews in database: {len(reviews)}")

            if apps:
                print("   Apps:")
                for app in apps[:5]:  # Show first 5
                    print(f"     - {app['app_name']} ({app.get('total_reviews', 0)} reviews)")

        except Exception as e:
            print(f"[DATA] Data status: Error - {e}")

        # Dashboard status
        try:
            import requests
            response = requests.get(f"http://localhost:{config.dashboard.port}", timeout=5)
            if response.status_code == 200:
                print(f"[DASH] Dashboard: Running on http://localhost:{config.dashboard.port}")
            else:
                print("[DASH] Dashboard: Not responding")
        except:
            print("[DASH] Dashboard: Not running")

        print("\n[INFO] Use 'python main.py run' to start the pipeline")
        print("[INFO] Use 'python main.py stop' to stop infrastructure")


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(description="ABSA Sentiment Pipeline Orchestrator")
    parser.add_argument(
        "command",
        choices=["run", "start", "stop", "status", "scrape", "dashboard"],
        help="Command to execute"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip sample data collection"
    )
    parser.add_argument(
        "--app-id",
        type=str,
        help="Specific app ID to scrape"
    )

    args = parser.parse_args()

    orchestrator = ABSAPipelineOrchestrator()

    try:
        if args.command == "run":
            # Run full pipeline
            success = orchestrator.run_full_pipeline(skip_data_collection=args.skip_data)
            if success:
                # Keep running until interrupted
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("\n[STOP] Shutting down pipeline...")
                    orchestrator.stop_infrastructure()
            sys.exit(0 if success else 1)

        elif args.command == "start":
            # Start infrastructure only
            success = orchestrator.start_infrastructure()
            sys.exit(0 if success else 1)

        elif args.command == "stop":
            # Stop infrastructure
            success = orchestrator.stop_infrastructure()
            sys.exit(0 if success else 1)

        elif args.command == "status":
            # Show status
            orchestrator.show_status()

        elif args.command == "scrape":
            # Run scraping only
            if args.app_id:
                from src.data.scraper import scrape_app_reviews
                result = scrape_app_reviews(args.app_id, count=50)
                print(f"Scraping result: {result}")
            else:
                print("Please specify --app-id for scraping")

        elif args.command == "dashboard":
            # Launch dashboard only
            success = orchestrator.launch_dashboard()
            if success:
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("\n[STOP] Stopping dashboard...")
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n[INFO] Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()