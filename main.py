"""
Main entry point for ABSA Sentiment Pipeline.
Orchestrates infrastructure, data collection, ABSA processing, SERVQUAL analysis, and dashboard launch.
Phase 2 implementation with complete ABSA and SERVQUAL integration.
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
from src.pipeline.sequential_processor import sequential_processor, SequentialProcessingResult
from typing import Optional, List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import config, health_checker, logger
# Import storage and scraper only when needed to avoid connection errors
# Import batch processing for ABSA integration


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

    def run_absa_processing(self, app_id: Optional[str] = None) -> bool:
        """Run ABSA batch processing on collected reviews."""
        self.logger.info("[ABSA] Starting ABSA batch processing...")

        try:
            # Import batch processor here to avoid early connection issues
            from src.pipeline.batch import batch_processor

            # Run daily processing
            if app_id:
                self.logger.info(f"[ABSA] Processing reviews for app: {app_id}")
                result = batch_processor.run_daily_processing(app_id)
            else:
                self.logger.info("[ABSA] Processing reviews for all apps")
                result = batch_processor.run_daily_processing()

            if result.success:
                self.logger.info(f"[OK] ABSA processing completed successfully")
                self.logger.info(f"   - Reviews processed: {result.reviews_processed}")
                self.logger.info(f"   - Aspects extracted: {result.aspects_extracted}")
                self.logger.info(f"   - SERVQUAL dimensions updated: {result.servqual_dimensions_updated}")
                self.logger.info(f"   - Processing time: {result.processing_time_seconds:.1f}s")
                return True
            else:
                self.logger.error(f"[ERROR] ABSA processing failed: {result.error_message}")
                return False

        except Exception as e:
            self.logger.error(f"[ERROR] ABSA processing error: {e}")
            return False

    def process_reviews_sequential(self, app_id: Optional[str] = None,
                                   resume_job_id: Optional[str] = None,
                                   skip_absa: bool = False,
                                   limit: Optional[int] = None) -> bool:
        """
        Process reviews using sequential ABSA → SERVQUAL workflow with checkpoints.

        Args:
            app_id: Process specific app or None for all apps
            resume_job_id: Resume existing job or None for new job

        Returns:
            True if processing successful, False otherwise
        """
        self.logger.info("[SEQUENTIAL] Starting sequential ABSA -> SERVQUAL processing...")

        try:
            from src.pipeline.sequential_processor import sequential_processor

            # Start sequential processing
            if resume_job_id:
                self.logger.info(f"[SEQUENTIAL] Resuming job: {resume_job_id}")
                result = sequential_processor.start_sequential_processing(
                    app_id,
                    resume_job_id,
                    skip_absa=skip_absa,
                    limit=limit
                )
            else:
                if app_id:
                    self.logger.info(f"[SEQUENTIAL] Processing reviews for app: {app_id}")
                else:
                    self.logger.info("[SEQUENTIAL] Processing reviews for all apps")
                result = sequential_processor.start_sequential_processing(
                    app_id,
                    resume_job_id,
                    skip_absa=skip_absa,
                    limit=limit
                )

            if result.success:
                self.logger.info(f"[OK] Sequential processing completed successfully")
                self.logger.info(f"   - Job ID: {result.job_id}")
                self.logger.info(f"   - Total reviews processed: {result.total_reviews_processed}")
                self.logger.info(f"   - ABSA phase: {result.absa_phase.reviews_processed} reviews")
                self.logger.info(f"   - SERVQUAL phase: {result.servqual_phase.reviews_processed} reviews")
                self.logger.info(f"   - Aspects extracted: {result.total_aspects_extracted}")
                self.logger.info(f"   - SERVQUAL dimensions updated: {result.total_servqual_dimensions_updated}")
                self.logger.info(f"   - Processing time: {result.processing_time_seconds:.1f}s")
                self.logger.info(f"   - Checkpoints created: {result.checkpoints_created}")
                if result.failed_reviews > 0:
                    self.logger.warning(f"   - Failed reviews: {result.failed_reviews}")
                return True
            else:
                self.logger.error(f"[ERROR] Sequential processing failed: {result.error_message}")
                self.logger.error(f"   - ABSA phase success: {result.absa_phase.success}")
                self.logger.error(f"   - SERVQUAL phase success: {result.servqual_phase.success}")
                return False

        except Exception as e:
            self.logger.error(f"[ERROR] Sequential processing error: {e}")
            return False

    def get_processing_status(self) -> Dict[str, Any]:
        """
        Get current processing status for dashboard monitoring.

        Returns:
            Dictionary with processing status information
        """
        try:
            from dashboard.data_loader import dashboard_data_loader
            return dashboard_data_loader.load_processing_status()
        except Exception as e:
            self.logger.error(f"Error getting processing status: {e}")
            return {'active_jobs': [], 'unprocessed_counts': [], 'notifications': []}

    def pause_processing_job(self, job_id: str) -> bool:
        """
        Pause an active processing job.

        Args:
            job_id: UUID of job to pause

        Returns:
            True if successfully paused, False otherwise
        """
        try:
            # Validate UUID format
            import uuid
            uuid.UUID(job_id)

            query = """
            UPDATE processing_jobs 
            SET status = 'paused'
            WHERE job_id = :job_id AND status = 'running'
            """

            from src.data.storage import storage
            rows_affected = storage.db.execute_non_query(query, {'job_id': job_id})

            if rows_affected > 0:
                self.logger.info(f"[PAUSE] Processing job paused: {job_id}")
                return True
            else:
                self.logger.error(f"[PAUSE] Job not found or not running: {job_id}")
                return False

        except ValueError:
            self.logger.error(f"Error pausing job - invalid UUID format: {job_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error pausing job {job_id}: {e}")
            return False

    def resume_processing_job(self, job_id: str) -> bool:
        """
        Resume a paused processing job.

        Args:
            job_id: UUID of job to resume

        Returns:
            True if successfully resumed, False otherwise
        """
        try:
            # Validate UUID format
            import uuid
            uuid.UUID(job_id)

            # Check if job can be resumed
            query = """
            SELECT status, app_id FROM processing_jobs 
            WHERE job_id = :job_id
            """

            from src.data.storage import storage
            df = storage.db.execute_query(query, {'job_id': job_id})

            if df.empty:
                self.logger.error(f"[RESUME] Job not found: {job_id}")
                return False

            row = df.iloc[0]
            status = row['status']
            app_id = row['app_id']

            if status != 'paused':
                self.logger.error(f"[RESUME] Job cannot be resumed - status: {status}")
                return False

            # Resume processing
            return self.process_reviews_sequential(app_id, job_id)

        except ValueError:
            self.logger.error(f"Error resuming job - invalid UUID format: {job_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error resuming job {job_id}: {e}")
            return False

    def show_processing_status(self):
        """Show detailed processing status information."""
        try:
            status = self.get_processing_status()

            print("\n=== PROCESSING STATUS ===")
            print(f"Active jobs: {len(status['active_jobs'])}")
            print(f"Unread notifications: {len(status['notifications'])}")

            if status['active_jobs']:
                print("\nActive Jobs:")
                for job in status['active_jobs']:
                    print(f"  • {job['job_id']}: {job['status']} ({job.get('current_phase', 'unknown')})")

            if status['unprocessed_counts']:
                print("\nUnprocessed Reviews by App:")
                for app in status['unprocessed_counts']:
                    print(f"  • {app['app_id']}: ABSA={app['absa_pending']}, SERVQUAL={app['servqual_pending']}")

            if status['notifications']:
                print("\nRecent Notifications:")
                for notif in status['notifications'][:5]:  # Show first 5
                    print(f"  • {notif['message']}")

            print(f"\nLast updated: {status.get('last_updated', 'Unknown')}")

        except Exception as e:
            self.logger.error(f"Error showing processing status: {e}")
            print("Error retrieving processing status")

    def process_servqual_scores(self, app_id: Optional[str] = None) -> bool:
        """Process SERVQUAL scores for business intelligence."""
        self.logger.info("[SERVQUAL] Processing SERVQUAL scores...")

        try:
            from src.absa.servqual_mapper import process_app_servqual
            from src.data.servqual_storage import servqual_storage
            from datetime import date

            target_date = date.today()

            if app_id:
                # Process SERVQUAL for specific app
                self.logger.info(f"[SERVQUAL] Processing scores for app: {app_id}")
                servqual_results = process_app_servqual(app_id, target_date)

                if servqual_results:
                    stored_count, error_count = servqual_storage.store_servqual_scores(servqual_results)
                    self.logger.info(f"[OK] SERVQUAL processing completed: {stored_count} scores stored")
                    return stored_count > 0
                else:
                    self.logger.info("[INFO] No SERVQUAL data generated (no ABSA results yet)")
                    return True
            else:
                # Process SERVQUAL for all apps
                from src.data.storage import storage
                apps = storage.apps.get_all_apps()

                total_scores = 0
                for app in apps:
                    app_id = app['app_id']
                    servqual_results = process_app_servqual(app_id, target_date)

                    if servqual_results:
                        stored_count, error_count = servqual_storage.store_servqual_scores(servqual_results)
                        total_scores += stored_count

                self.logger.info(f"[OK] SERVQUAL processing completed: {total_scores} total scores stored")
                return True

        except Exception as e:
            self.logger.error(f"[ERROR] SERVQUAL processing error: {e}")
            return False

    def run_sample_data_collection(self) -> bool:
        """Run sample data collection for demonstration."""
        self.logger.info("[DATA] Running sample data collection...")

        # Import scraper here to avoid connection issues
        from src.data.scraper import scrape_app_reviews

        sample_apps = [
            "com.amazon.mShop.android.shopping",
            "com.einnovation.temu",
            "com.zzkko",
            "com.ebay.mobile",
            "com.etsy.android"
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

    def run_full_pipeline(self, skip_data_collection: bool = False, skip_absa_processing: bool = False) -> bool:
        """Run the complete pipeline workflow."""
        self.logger.info("[START] Starting ABSA Pipeline - Phase 2 Enhanced")

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

        # Step 4: ABSA batch processing (optional)
        if not skip_absa_processing:
            if not self.run_absa_processing():
                self.logger.warning("[WARN] ABSA processing failed, but continuing...")
        else:
            self.logger.info("[SKIP] Skipping ABSA processing")

        # Step 5: Launch dashboard
        if not self.launch_dashboard():
            self.logger.error("[ERROR] Failed to launch dashboard")
            return False

        # Success message
        self.logger.info("[SUCCESS] ABSA Pipeline Phase 2 launched successfully!")
        self.logger.info("[INFO] What's running:")
        self.logger.info("   - PostgreSQL database")
        self.logger.info("   - Redis cache")
        self.logger.info("   - Kafka streaming (ready for Phase 3)")
        self.logger.info("   - ABSA analysis engine (RoBERTa + spaCy)")
        self.logger.info("   - SERVQUAL business intelligence")
        self.logger.info("   - Streamlit dashboard")

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

        # ABSA processing status
        try:
            from src.absa.engine import get_absa_engine_status
            from src.pipeline.batch import get_batch_processing_stats

            # Get ABSA engine status
            engine_status = get_absa_engine_status()
            print(f"[ABSA] Engine status: {'Loaded' if engine_status.deep_engine_loaded else 'Not loaded'}")
            print(f"[ABSA] Memory usage: {engine_status.models_memory_usage_mb:.0f} MB")
            print(f"[ABSA] Reviews processed: {engine_status.total_reviews_processed}")

            # Get recent processing statistics
            stats = get_batch_processing_stats(days=7)
            job_stats = stats.get('job_statistics', {})
            absa_stats = stats.get('absa_statistics', {})
            servqual_stats = stats.get('servqual_statistics', {})

            if job_stats:
                print(f"[ABSA] Recent jobs (7 days): {job_stats.get('total_jobs', 0)} total, "
                      f"{job_stats.get('successful_jobs', 0)} successful")

            if absa_stats:
                print(f"[ABSA] ABSA records (7 days): {absa_stats.get('total_absa_records', 0)}")
                print(f"[ABSA] Average sentiment: {absa_stats.get('avg_sentiment', 0):.2f}")

            if servqual_stats:
                print(f"[SERVQUAL] Quality records (7 days): {servqual_stats.get('total_servqual_records', 0)}")
                print(f"[SERVQUAL] Average quality score: {servqual_stats.get('avg_quality_score', 0):.1f}/5.0")

        except Exception as e:
            print(f"[ABSA] Processing status: Error - {e}")

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

        print("\n[INFO] Use 'python main.py run' to start the complete pipeline")
        print("[INFO] Use 'python main.py sequential' to run sequential ABSA → SERVQUAL processing")
        print("[INFO] Use 'python main.py absa' to run ABSA processing only")
        print("[INFO] Use 'python main.py process --app-id <app>' for complete app processing")
        print("[INFO] Use 'python main.py stop' to stop infrastructure")


def main():
    """Main entry point with argparse command handling."""
    parser = argparse.ArgumentParser(description="ABSA Sentiment Pipeline Orchestrator")
    parser.add_argument(
        "command",
        choices=[
            "run", "start", "stop", "status", "scrape", "dashboard",
            "absa", "servqual", "process",
            "sequential", "pause", "resume", "process-status"
        ],
        help="Command to execute"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip sample data collection"
    )
    parser.add_argument(
        "--skip-absa",
        action="store_true",
        help="Skip ABSA processing"
    )
    parser.add_argument(
        "--app-id",
        type=str,
        help="Specific app ID to scrape or process"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        help="Job ID (UUID) for pause/resume operations"
    )
    parser.add_argument(
        "--resume-job-id",
        type=str,
        help="Job ID (UUID) to resume for sequential processing"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of reviews to process (for testing)"
    )

    args = parser.parse_args()

    orchestrator = ABSAPipelineOrchestrator()

    try:
        if args.command == "run":
            # Run full pipeline
            success = orchestrator.run_full_pipeline(
                skip_data_collection=args.skip_data,
                skip_absa_processing=args.skip_absa
            )
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

        elif args.command == "absa":
            # Run ABSA processing only
            success = orchestrator.run_absa_processing(args.app_id)
            sys.exit(0 if success else 1)

        elif args.command == "servqual":
            # Run SERVQUAL processing only
            success = orchestrator.process_servqual_scores(args.app_id)
            sys.exit(0 if success else 1)


        elif args.command == "sequential":
            # Run sequential ABSA → SERVQUAL processing
            if args.resume_job_id:
                print(f"Resuming sequential processing job: {args.resume_job_id}")
                success = orchestrator.process_reviews_sequential(
                    args.app_id,
                    args.resume_job_id,
                    skip_absa=args.skip_absa,
                    limit=args.limit
                )
            else:
                if args.app_id:
                    print(f"Starting sequential processing for app: {args.app_id}")
                else:
                    print("Starting sequential processing for all apps")

                success = orchestrator.process_reviews_sequential(
                    args.app_id,
                    skip_absa=args.skip_absa,
                    limit=args.limit
                )

        elif args.command == "pause":
            # Pause active processing job
            if not args.job_id:
                print("Please specify --job-id for pausing")
                sys.exit(1)

            success = orchestrator.pause_processing_job(args.job_id)
            if success:
                print(f"Job {args.job_id} paused successfully")
            else:
                print(f"Failed to pause job {args.job_id}")
                sys.exit(1)

        elif args.command == "resume":
            # Resume paused processing job
            if not args.job_id:
                print("Please specify --job-id for resuming")
                sys.exit(1)

            success = orchestrator.resume_processing_job(args.job_id)
            if success:
                print(f"Job {args.job_id} resumed successfully")
            else:
                print(f"Failed to resume job {args.job_id}")
                sys.exit(1)

        elif args.command == "process-status":
            # Show detailed processing status
            orchestrator.show_processing_status()

        elif args.command == "process":
            # Run complete processing (scrape + ABSA + SERVQUAL) for specific app
            if args.app_id:
                logger.info(f"[PROCESS] Complete processing for {args.app_id}")

                # First scrape reviews
                success = orchestrator.run_sample_data_collection()
                if success:
                    # Then process with ABSA
                    success = orchestrator.run_absa_processing(args.app_id)
                    if success:
                        # Finally process SERVQUAL
                        success = orchestrator.process_servqual_scores(args.app_id)

                sys.exit(0 if success else 1)
            else:
                print("Please specify --app-id for processing")

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







"""
        print("ABSA Pipeline - Available commands:")
        print("  python main.py                                      - Run default pipeline")
        print("  python main.py sequential [app_id] [resume_job_id]  - Run sequential ABSA → SERVQUAL processing")
        print("  python main.py pause <job_id>                       - Pause active processing job")
        print("  python main.py resume <job_id>                      - Resume paused processing job")
        print("  python main.py status                               - Show processing status")
        print("  python main.py absa [app_id]                        - Run ABSA processing only")
        print("  python main.py servqual [app_id]                    - Run SERVQUAL processing only")
        print("  python main.py help                                 - Show this help message")
        print("  http://localhost:8080      - Database")
        print("  docker-compose up -d                                - Docker start-up")
        print("  docker-compose ps                                   - Check if services are running")
        print("  streamlit run dashboard_app.py                      - dashboard")

        print()
        print("Examples:")
        print("  python main.py sequential                           - Process all apps sequentially")
        print("  python main.py sequential com.amazon.mshop.android.shopping  - Process specific app")
        print("  python main.py pause 123e4567-e89b-12d3-a456-426614174000    - Pause specific job")
        print("  python main.py resume 123e4567-e89b-12d3-a456-426614174000   - Resume specific job")
        """