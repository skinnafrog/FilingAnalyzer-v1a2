"""
Celery tasks for automated SEC filing ingestion.
Schedules RSS monitoring during market hours (9:30 AM - 4:00 PM ET, Monday-Friday).
"""
import asyncio
from datetime import datetime, time, timedelta
import pytz
import logging
from typing import Optional
from celery import Celery, Task
from celery.schedules import crontab
from kombu import Queue

from src.config.settings import get_settings
from src.main import IngestionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Initialize Celery app
app = Celery('financial_intel')

# Celery configuration
app.conf.update(
    broker_url=settings.CELERY_BROKER_URL,
    result_backend=settings.CELERY_RESULT_BACKEND,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='America/New_York',
    enable_utc=True,

    # Task routing
    task_routes={
        'src.tasks.manual_ingest': {'queue': 'ingestion'},  # Route manual_ingest to ingestion queue
        'src.tasks.ingest_sec_filings': {'queue': 'ingestion'},
        'src.tasks.process_single_filing': {'queue': 'processing'},
    },

    # Queue configuration
    task_queues=(
        Queue('celery', routing_key='celery'),  # Default queue
        Queue('ingestion', routing_key='ingestion.#'),
        Queue('processing', routing_key='processing.#'),
    ),

    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,

    # Beat schedule for periodic tasks
    beat_schedule={
        'ingest-sec-filings-market-hours': {
            'task': 'src.tasks.ingest_sec_filings',
            'schedule': timedelta(seconds=settings.RSS_POLL_INTERVAL),  # Every 10 minutes
            'options': {
                'expires': 300,  # Task expires in 5 minutes if not executed
            }
        },

        # Daily market open check
        'market-open-check': {
            'task': 'src.tasks.check_market_status',
            'schedule': crontab(hour=9, minute=25),  # 9:25 AM ET daily
        },

        # Daily market close check
        'market-close-check': {
            'task': 'src.tasks.check_market_status',
            'schedule': crontab(hour=16, minute=5),  # 4:05 PM ET daily
        },
    }
)


def is_market_hours() -> bool:
    """
    Check if current time is during US market hours.
    Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday

    Returns:
        bool: True if within market hours, False otherwise
    """
    et_tz = pytz.timezone('America/New_York')
    now = datetime.now(et_tz)

    # Check if it's a weekday (Monday=0, Sunday=6)
    if now.weekday() > 4:  # Saturday or Sunday
        return False

    # Check if it's a US holiday (simplified - you may want to add holiday calendar)
    # For now, just check time range

    market_open = time(9, 30)
    market_close = time(16, 0)
    current_time = now.time()

    return market_open <= current_time <= market_close


class CallbackTask(Task):
    """Task with callbacks for success and failure."""

    def on_success(self, retval, task_id, args, kwargs):
        """Called on successful task execution."""
        logger.info(f"Task {task_id} completed successfully")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        logger.error(f"Task {task_id} failed: {exc}")


@app.task(base=CallbackTask, bind=True, max_retries=3)
def ingest_sec_filings(self, limit: Optional[int] = None, force: bool = False):
    """
    Main task for ingesting SEC filings.
    Only runs during market hours unless forced.

    Args:
        limit: Optional limit on number of filings to process
        force: If True, bypass market hours check (for manual triggers)

    Returns:
        dict: Statistics about the ingestion run
    """
    try:
        # Check if we're in market hours (skip check if forced)
        if not force and not is_market_hours():
            logger.info("Outside market hours, skipping ingestion")
            return {
                'status': 'skipped',
                'reason': 'outside_market_hours',
                'timestamp': datetime.utcnow().isoformat()
            }

        logger.info("Starting SEC filing ingestion task")

        # Create pipeline instance
        pipeline = IngestionPipeline()

        # Run async ingestion in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run single poll cycle
            loop.run_until_complete(pipeline.run_once(limit=limit))

            # Get statistics
            stats = pipeline.get_statistics()

            logger.info(f"Ingestion completed: {stats}")

            return {
                'status': 'success',
                'processed': stats.get('processed_count', 0),
                'failed': stats.get('failed_count', 0),
                'timestamp': datetime.utcnow().isoformat()
            }

        finally:
            loop.close()

    except Exception as exc:
        logger.error(f"Ingestion task failed: {exc}")
        # Retry the task
        raise self.retry(exc=exc, countdown=60)


@app.task
def process_single_filing(accession_number: str):
    """
    Process a single SEC filing.

    Args:
        accession_number: SEC accession number

    Returns:
        dict: Processing result
    """
    try:
        logger.info(f"Processing filing: {accession_number}")

        # TODO: Implement single filing processing
        # This would be used for reprocessing or priority processing

        return {
            'status': 'success',
            'accession_number': accession_number,
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as exc:
        logger.error(f"Failed to process filing {accession_number}: {exc}")
        return {
            'status': 'error',
            'accession_number': accession_number,
            'error': str(exc),
            'timestamp': datetime.utcnow().isoformat()
        }


@app.task
def check_market_status():
    """
    Check and log market status.
    Used for monitoring and alerting.

    Returns:
        dict: Market status information
    """
    is_open = is_market_hours()

    et_tz = pytz.timezone('America/New_York')
    now = datetime.now(et_tz)

    status = {
        'is_open': is_open,
        'current_time_et': now.isoformat(),
        'day_of_week': now.strftime('%A'),
        'timestamp': datetime.utcnow().isoformat()
    }

    if is_open:
        logger.info("Market is OPEN - Ingestion tasks are active")
    else:
        logger.info("Market is CLOSED - Ingestion tasks are paused")

    return status


@app.task
def get_ingestion_status():
    """
    Get current ingestion pipeline status.
    Used by API to report status to UI.

    Returns:
        dict: Current pipeline status
    """
    try:
        pipeline = IngestionPipeline()
        stats = pipeline.get_statistics()

        return {
            'status': 'operational',
            'market_hours': is_market_hours(),
            'statistics': stats,
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as exc:
        logger.error(f"Failed to get status: {exc}")
        return {
            'status': 'error',
            'error': str(exc),
            'timestamp': datetime.utcnow().isoformat()
        }


# Manual trigger endpoint for UI
@app.task
def manual_ingest(limit: int = 5):
    """
    Manually trigger ingestion from UI.
    Bypasses market hours check.

    Args:
        limit: Number of filings to process

    Returns:
        dict: Ingestion result
    """
    logger.info(f"Manual ingestion triggered (limit={limit})")

    # Call the main ingestion task with force=True to bypass market hours check
    result = ingest_sec_filings.apply_async(args=[limit], kwargs={'force': True})

    return {
        'task_id': result.id,
        'status': 'queued',
        'limit': limit,
        'forced': True,
        'timestamp': datetime.utcnow().isoformat()
    }


if __name__ == '__main__':
    # For testing
    print("Celery app configured")
    print(f"Broker: {app.conf.broker_url}")
    print(f"Market hours check: {is_market_hours()}")