"""
Configuration management, logging setup, and health checks for ABSA Pipeline.
Centralizes all application settings and provides utilities for system monitoring.
"""

import os
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import yaml
from dotenv import load_dotenv
import psutil
import redis
import psycopg2
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable


# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "absa_pipeline")
    username: str = os.getenv("DB_USER", "absa_user")
    password: str = os.getenv("DB_PASSWORD", "absa_password")

    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def psycopg2_params(self) -> Dict[str, Any]:
        """Generate psycopg2 connection parameters."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.username,
            "password": self.password
        }


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")

    @property
    def connection_params(self) -> Dict[str, Any]:
        """Generate Redis connection parameters."""
        params = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "decode_responses": True
        }
        if self.password:
            params["password"] = self.password
        return params


@dataclass
class KafkaConfig:
    """Kafka configuration."""
    bootstrap_servers: List[str] = None
    producer_config: Dict[str, Any] = None
    consumer_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.bootstrap_servers is None:
            self.bootstrap_servers = [os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")]

        if self.producer_config is None:
            self.producer_config = {
                "bootstrap_servers": self.bootstrap_servers,
                "value_serializer": lambda x: x.encode("utf-8") if isinstance(x, str) else x,
                "acks": "all",
                "retries": 3,
                "batch_size": 16384,
                "linger_ms": 10,
                "buffer_memory": 33554432
            }

        if self.consumer_config is None:
            self.consumer_config = {
                "bootstrap_servers": self.bootstrap_servers,
                "auto_offset_reset": "earliest",
                "enable_auto_commit": True,
                "group_id": "absa_pipeline_group",
                "value_deserializer": lambda x: x.decode("utf-8") if x else None
            }


@dataclass
class ABSAConfig:
    """ABSA processing configuration."""
    deep_model_name: str = os.getenv("ABSA_DEEP_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    quick_model_name: str = os.getenv("ABSA_QUICK_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
    aspect_extraction_model: str = os.getenv("ASPECT_MODEL", "en_core_web_sm")
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "512"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    batch_size: int = int(os.getenv("ABSA_BATCH_SIZE", "50"))  # Change 32 → 50
    cache_results: bool = os.getenv("CACHE_RESULTS", "true").lower() == "true"

    # Aspect definitions file
    aspects_file: str = str(CONFIG_DIR / "aspects.yml")


@dataclass
class ScrapingConfig:
    """Web scraping configuration."""
    max_reviews_per_app: int = int(os.getenv("MAX_REVIEWS_PER_APP", "1000"))
    scraping_delay: float = float(os.getenv("SCRAPING_DELAY", "1.0"))
    batch_scraping_interval: int = int(os.getenv("BATCH_SCRAPING_INTERVAL", "86400"))  # 24 hours
    realtime_scraping_interval: int = int(os.getenv("REALTIME_SCRAPING_INTERVAL", "300"))  # 5 minutes
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    user_agent: str = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    port: int = int(os.getenv("DASHBOARD_PORT", "8502"))  # Changed default to 8502 (working port)
    host: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")  # Changed to bind to all interfaces
    auto_refresh_interval: int = int(os.getenv("AUTO_REFRESH_INTERVAL", "30"))  # seconds
    max_data_points: int = int(os.getenv("MAX_DATA_POINTS", "10000"))
    cache_ttl: int = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    file_path: str = str(LOGS_DIR / "absa_pipeline.log")
    max_file_size: int = int(os.getenv("LOG_MAX_FILE_SIZE", "10485760"))  # 10MB
    backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Config:
    """Main configuration class that combines all configuration objects."""

    def __init__(self):
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.kafka = KafkaConfig()
        self.absa = ABSAConfig()
        self.scraping = ScrapingConfig()
        self.dashboard = DashboardConfig()
        self.logging = LoggingConfig()

        # Application settings
        self.app_name = "ABSA Sentiment Pipeline"
        self.version = "1.0.0"
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

        # Processing settings
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        self.processing_timeout = int(os.getenv("PROCESSING_TIMEOUT", "300"))  # 5 minutes

        # Health check settings
        self.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "120"))  # 2 minutes

    def load_aspects_config(self) -> Dict[str, Any]:
        """Load aspect definitions from YAML file."""
        try:
            with open(self.absa.aspects_file, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logging.warning(f"Aspects config file not found: {self.absa.aspects_file}")
            return self._get_default_aspects()
        except yaml.YAMLError as e:
            logging.error(f"Error parsing aspects config: {e}")
            return self._get_default_aspects()

    def _get_default_aspects(self) -> Dict[str, Any]:
        """Return default aspect configuration if file is not available."""
        return {
            "aspects": {
                "ui": {"keywords": ["interface", "design", "layout"], "weight": 1.0},
                "performance": {"keywords": ["speed", "fast", "slow", "lag"], "weight": 1.2},
                "battery": {"keywords": ["battery", "drain", "power"], "weight": 1.1},
                "features": {"keywords": ["feature", "function", "capability"], "weight": 1.0},
                "usability": {"keywords": ["easy", "difficult", "intuitive"], "weight": 1.0}
            }
        }


def setup_logging(config: LoggingConfig = None) -> logging.Logger:
    """Set up logging configuration for the application."""
    if config is None:
        config = LoggingConfig()

    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                config.file_path,
                maxBytes=config.max_file_size,
                backupCount=config.backup_count
            )
        ]
    )

    # Set specific loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("kafka").setLevel(logging.WARNING)

    logger = logging.getLogger("absa_pipeline")
    logger.info(f"Logging initialized. Level: {config.level}")

    return logger


class HealthChecker:
    """System health monitoring and service availability checks."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("absa_pipeline.health")

    def check_database_health(self) -> Dict[str, Any]:
        """Check PostgreSQL database connectivity and basic metrics."""
        try:
            import psycopg2
            conn = psycopg2.connect(**self.config.database.psycopg2_params)
            cursor = conn.cursor()

            # Test query
            cursor.execute("SELECT 1")
            cursor.fetchone()

            # Get database size
            cursor.execute("""
                SELECT pg_size_pretty(pg_database_size(%s))
            """, (self.config.database.database,))
            db_size = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            return {
                "status": "healthy",
                "database_size": db_size,
                "connection_time": "< 1s"
            }

        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and basic metrics."""
        try:
            client = redis.Redis(**self.config.redis.connection_params)

            # Test connection
            client.ping()

            # Get memory usage
            info = client.info("memory")
            used_memory = info.get("used_memory_human", "N/A")

            client.close()

            return {
                "status": "healthy",
                "used_memory": used_memory,
                "connection_time": "< 1s"
            }

        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def check_kafka_health(self) -> Dict[str, Any]:
        """Check Kafka connectivity and basic metrics."""
        try:
            # Test producer
            producer = KafkaProducer(**self.config.kafka.producer_config)
            producer.close()

            return {
                "status": "healthy",
                "brokers": self.config.kafka.bootstrap_servers,
                "connection_time": "< 1s"
            }

        except NoBrokersAvailable:
            return {
                "status": "unhealthy",
                "error": "No Kafka brokers available"
            }
        except Exception as e:
            self.logger.error(f"Kafka health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "status": "healthy",
                "cpu_usage": f"{cpu_percent}%",
                "memory_usage": f"{memory.percent}%",
                "disk_usage": f"{disk.percent}%",
                "available_memory": f"{memory.available / (1024**3):.2f} GB"
            }

        except Exception as e:
            self.logger.error(f"System resource check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def get_overall_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health_checks = {
            "database": self.check_database_health(),
            "redis": self.check_redis_health(),
            "kafka": self.check_kafka_health()
        }

        # Try system check but don't fail if it errors
        try:
            health_checks["system"] = self.check_system_resources()
        except Exception as e:
            self.logger.warning(f"System resource check skipped due to error: {e}")
            health_checks["system"] = {"status": "skipped", "error": "Resource check unavailable"}

        # Determine overall status (ignore system check for overall health)
        core_services = ["database", "redis", "kafka"]
        unhealthy_services = [
            service for service in core_services
            if health_checks[service]["status"] == "unhealthy"
        ]

        overall_status = "unhealthy" if unhealthy_services else "healthy"

        return {
            "overall_status": overall_status,
            "unhealthy_services": unhealthy_services,
            "services": health_checks,
            "timestamp": datetime.now().isoformat()
        }


# Global configuration instance
config = Config()

# Initialize logging
logger = setup_logging(config.logging)

# Health checker instance
health_checker = HealthChecker(config)