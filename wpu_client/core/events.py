"""Event bus for inter-service communication."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Base event class."""

    event_type: str
    data: dict[str, Any]


class EventBus:
    """Simple event bus for pub/sub communication between services."""

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to listen for
            callback: Function to call when event is published
        """
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to event: {event_type}")

    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event to stop listening for
            callback: Function to remove from subscribers
        """
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            logger.debug(f"Unsubscribed from event: {event_type}")

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        logger.debug(f"Publishing event: {event.event_type}")
        for callback in self._subscribers[event.event_type]:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback for {event.event_type}: {e}")

    def clear(self) -> None:
        """Clear all subscribers."""
        self._subscribers.clear()


# Global event bus instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
