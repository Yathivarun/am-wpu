#!/usr/bin/env python3
"""WPU Client - Main entry point for the application."""

import argparse
import contextlib
import logging
import os
import signal
import sys
from pathlib import Path

from wpu_client.core.service_base import ServiceBase

# Deliberately NOT imported here: face_service and slideshow_service, which
# pull in picamera2, cv2 and onnxruntime at module scope. `--check` exists to
# diagnose exactly the environments where those imports fail, so it has to be
# reachable without them — they are imported inside main(), after the checks
# have had their chance to explain what is wrong.

LOG_DIR = "/var/log/wpu-client"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO", to_file: bool = True) -> None:
    """Log to stderr, and to LOG_FILE when it is writable.

    The file handler is best-effort on purpose. It used to be set up at import
    with an unconditional makedirs, so a unit whose log dir was missing or
    owned by another user died on import — before any of its own logging could
    say so. Now it degrades to stderr, which systemd captures anyway, and
    `--check` reports the log dir separately.

    `to_file=False` is for --check, which must not create the very directory
    it is about to report on.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if to_file:
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            handlers.append(logging.FileHandler(LOG_FILE))
        except OSError as e:
            print(f"warning: not logging to {LOG_FILE} ({e})", file=sys.stderr)

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        # The hostname is what makes 50 units' logs legible once they are
        # aggregated; without it every line looks the same.
        format=f"%(asctime)s - {os.uname().nodename} - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def run_preflight(args) -> int:
    """`--check`: report whether this unit can do its job, and exit.

    stdout carries the report and nothing else — a fleet parses `--json` off
    it — so the config loader's prints and every library log line are pushed
    to stderr for the duration.
    """
    from wpu_client import health
    from wpu_client.config.settings import CONFIG_DIR, get_settings, reload_settings

    with contextlib.redirect_stdout(sys.stderr):
        settings = reload_settings(args.config) if args.config else get_settings()
        if args.diagnostic:
            settings.services.face_recognition.diagnostic_mode = True
        diagnostic = settings.services.face_recognition.diagnostic_mode
        config_path = Path(args.config) if args.config else CONFIG_DIR / "config.yaml"
        results = health.run_checks(settings, diagnostic, config_path)

    render = health.render_json if args.json else health.render_text
    print(render(results, diagnostic))
    return health.exit_code(results)


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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run pre-flight checks and exit: dependencies, models, config, "
             "scene art, server reachability, camera, log dir, disk. Exits 1 "
             "if any check fails. Starts no services and touches no state.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --check, emit JSON instead of a table (for Ansible/monitoring)",
    )
    args = parser.parse_args()

    if args.check:
        # WARNING silences httpx's per-request INFO line, which would
        # otherwise interleave with the report.
        configure_logging("WARNING", to_file=False)
        sys.exit(run_preflight(args))

    configure_logging(args.log_level)

    from wpu_client.config.settings import get_settings, reload_settings

    settings = reload_settings(args.config) if args.config else get_settings()

    # --diagnostic overrides config: offline local recognition + local sketches
    if args.diagnostic:
        settings.services.face_recognition.diagnostic_mode = True
        logger.info("Diagnostic mode enabled via --diagnostic flag")

    # Imported here rather than at module scope so --check above still runs on
    # a box where picamera2/cv2/onnxruntime are missing or broken.
    from wpu_client.core.events import get_event_bus
    from wpu_client.services.face_recognition.face_service import FaceRecognitionService
    from wpu_client.services.slideshow.slideshow_service import SlideshowService

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
                    use_legacy_final_images=(
                        settings.services.face_recognition.use_legacy_final_images
                    ),
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
