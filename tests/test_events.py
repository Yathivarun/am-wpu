from wpu_client.core.events import Event, EventBus


def test_publish_delivers_to_subscriber():
    bus = EventBus()
    received = []
    bus.subscribe("person.detected", received.append)

    event = Event(event_type="person.detected", data={"name": "Varun"})
    bus.publish(event)

    assert received == [event]


def test_publish_ignores_other_event_types():
    bus = EventBus()
    received = []
    bus.subscribe("person.detected", received.append)

    bus.publish(Event(event_type="person.left", data={}))

    assert received == []


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    received = []
    callback = received.append
    bus.subscribe("faces.recognized", callback)
    bus.unsubscribe("faces.recognized", callback)

    bus.publish(Event(event_type="faces.recognized", data={}))

    assert received == []


def test_unsubscribe_missing_callback_is_a_noop():
    bus = EventBus()
    # Never subscribed — must not raise.
    bus.unsubscribe("person.left", lambda e: None)


def test_a_raising_subscriber_does_not_block_the_others():
    bus = EventBus()
    received = []

    def bad_callback(event):
        raise RuntimeError("boom")

    bus.subscribe("person.detected", bad_callback)
    bus.subscribe("person.detected", received.append)

    bus.publish(Event(event_type="person.detected", data={}))

    assert len(received) == 1


def test_clear_removes_all_subscribers():
    bus = EventBus()
    received = []
    bus.subscribe("person.detected", received.append)

    bus.clear()
    bus.publish(Event(event_type="person.detected", data={}))

    assert received == []
