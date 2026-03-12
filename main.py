#!/usr/bin/env python3
"""WPU Client - Main entry point for the application."""

import argparse
import logging
import os
import signal
import sys

from wpu_client.config.settings import get_settings
from wpu_client.core.events import EventBus, get_event_bus
from wpu_client.core.service_base import ServiceBase
from wpu_client.services.face_recognition.face_service import FaceRecognitionService
from wpu_client.services.slideshow.slideshow_service import SlideshowService

# Ensure log directory exists
LOG_DIR = "/var/log/wpu-client"
LOG_FILE = os.path.join(LOG_DIR, "app.log")
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging — stdout + file for Alloy to tail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)


class ServiceOrchestrator:
    """Orchestrates multiple services and handles graceful shutdown."""

    def __init__(self, services: list[ServiceBase]):
        """
        Initialize the orchestrator.

        Args:
            services: List of services to manage
        """
        self.services = services
        self._shutdown_requested = False

    def start(self) -> None:
        """Start all services."""
        logger.info("Starting services...")

        # Check if we have a slideshow service (runs in main thread)
        has_slideshow = any(s.name == "slideshow" for s in self.services)

        # Start non-GTK services first (they run in background threads)
        for service in self.services:
            if service.name != "slideshow":  # Slideshow runs in main thread
                service.start()

        # Start slideshow service last (runs in main thread)
        for service in self.services:
            if service.name == "slideshow":
                service.start()

        # If no slideshow service, keep main thread alive for background services
        if not has_slideshow:
            logger.info("Running in background mode - press Ctrl+C to stop")
            try:
                # Keep main thread alive
                while self._has_running_services():
                    import time
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass

    def _has_running_services(self) -> bool:
        """Check if any services are still running."""
        return any(service.is_running() for service in self.services)

    def stop(self) -> None:
        """Stop all services."""
        logger.info("Stopping services...")

        # Stop in reverse order
        for service in reversed(self.services):
            service.stop()

        logger.info("All services stopped")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="WPU Client - Face Recognition Slideshow")
    parser.add_argument(
        "--service",
        choices=["slideshow", "face-recognition", "all"],
        default="all",
        help="Which service(s) to start (default: all)",
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load configuration
    settings = get_settings()
    if args.config:
        from wpu_client.config.settings import reload_settings

        settings = reload_settings(args.config)

    # Get event bus
    event_bus = get_event_bus()

    # Create services based on arguments
    services = []
    face_recognition_service = None

    if args.service in ("face-recognition", "all"):
        if settings.services.face_recognition.enabled:
            face_recognition_service = FaceRecognitionService(settings.services.face_recognition, event_bus)
            services.append(face_recognition_service)
            logger.info("Face recognition service added")
        else:
            logger.warning("Face recognition service is disabled in configuration")

    if args.service in ("slideshow", "all"):
        if settings.services.slideshow.enabled:
            services.append(
                SlideshowService(
                    settings.services.slideshow,
                    event_bus,
                    settings.services.face_recognition.wpu_endpoint,
                    face_recognition_service,  # pass for camera preview
                )
            )
            logger.info("Slideshow service added")
        else:
            logger.warning("Slideshow service is disabled in configuration")

    if not services:
        logger.error("No services enabled. Exiting.")
        sys.exit(1)

    # Create orchestrator
    orchestrator = ServiceOrchestrator(services)

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start all services
        orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        orchestrator.stop()
    except Exception as e:
        logger.error(f"Error running services: {e}", exc_info=True)
        orchestrator.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
