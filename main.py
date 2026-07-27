#!/usr/bin/env python3
"""WPU Client - Main entry point for the application."""

import argparse
import logging
import os
import signal
import sys

from wpu_client.config.settings import get_settings
from wpu_client.core.events import get_event_bus
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
        self.services = services
        self._shutdown_requested = False

    def start(self) -> None:
        """Start all services."""
        logger.info("Starting services...")

        has_slideshow = any(s.name == "slideshow" for s in self.services)

        for service in self.services:
            if service.name != "slideshow":
                service.start()

        for service in self.services:
            if service.name == "slideshow":
                service.start()

        if not has_slideshow:
            logger.info("Running in background mode - press Ctrl+C to stop")
            try:
                while self._has_running_services():
                    import time
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass

    def _has_running_services(self) -> bool:
        return any(service.is_running() for service in self.services)

    def stop(self) -> None:
        """Stop all services."""
        logger.info("Stopping services...")
        for service in reversed(self.services):
            service.stop()
        logger.info("All services stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WPU Client - Face Recognition Slideshow")
    parser.add_argument(
        "--service",
        choices=["slideshow", "face-recognition", "all"],
        default="all",
        help="Which service(s) to start (default: all)",
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Offline diagnostic mode: recognise faces against the local seeded "
             "gallery (no server) and show each person's own local sketches",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    settings = get_settings()
    if args.config:
        from wpu_client.config.settings import reload_settings
        settings = reload_settings(args.config)

    # --diagnostic overrides config: offline local recognition + local sketches
    if args.diagnostic:
        settings.services.face_recognition.diagnostic_mode = True
        logger.info("Diagnostic mode enabled via --diagnostic flag")

    event_bus = get_event_bus()

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
                    face_recognition_service,
                    diagnostic_mode=settings.services.face_recognition.diagnostic_mode,
                )
            )
            logger.info("Slideshow service added")
        else:
            logger.warning("Slideshow service is disabled in configuration")

    if not services:
        logger.error("No services enabled. Exiting.")
        sys.exit(1)

    orchestrator = ServiceOrchestrator(services)

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
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
