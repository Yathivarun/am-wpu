import threading
import time

from wpu_client.core.service_base import ServiceBase


class DummyService(ServiceBase):
    """Minimal concrete ServiceBase for exercising the shared thread helpers."""

    def __init__(self):
        super().__init__("dummy")
        self.loop_started = threading.Event()

    def start(self) -> None:
        self._running = True
        self._run_in_thread(self._loop)

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        self._wait_for_thread(timeout=2.0)

    def _loop(self) -> None:
        self.loop_started.set()
        while not self._stop_event.is_set():
            time.sleep(0.01)


def test_not_running_before_start():
    service = DummyService()
    assert service.is_running() is False


def test_start_runs_target_in_background_thread():
    service = DummyService()
    service.start()
    try:
        assert service.loop_started.wait(timeout=2.0)
        assert service.is_running() is True
        assert service._thread is not threading.current_thread()
    finally:
        service.stop()


def test_stop_joins_the_background_thread():
    service = DummyService()
    service.start()
    service.loop_started.wait(timeout=2.0)

    service.stop()

    assert service._thread.is_alive() is False


def test_repr_includes_name_and_running_state():
    service = DummyService()
    assert "dummy" in repr(service)
    assert "running=False" in repr(service)
