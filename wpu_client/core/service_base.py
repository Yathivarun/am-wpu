"""Base class for all services."""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class ServiceBase(ABC):
    """
    Abstract base class for all services.

    All services must inherit from this class and implement
    the start() and stop() methods.
    """

    def __init__(self, name: str):
        """
        Initialize the service.

        Args:
            name: Unique name for this service
        """
        self.name = name
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @abstractmethod
    def start(self) -> None:
        """Start the service. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the service. Must be implemented by subclasses."""
        pass

    def is_running(self) -> bool:
        """Check if the service is currently running."""
        return self._running

    def _run_in_thread(self, target: callable) -> None:
        """
        Run a function in a background thread.

        Args:
            target: Function to run in the thread
        """
        self._thread = threading.Thread(target=target, name=self.name, daemon=True)
        self._thread.start()

    def _wait_for_thread(self, timeout: Optional[float] = None) -> None:
        """
        Wait for the background thread to finish.

        Args:
            timeout: Maximum time to wait in seconds. None waits indefinitely.
        """
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', running={self._running})>"
