"""Pub/Sub pull consumer (local + emulator). On Cloud Run (Week 7) the same
Processor.handle is called from a push endpoint instead."""
import logging
import threading
import time
from google.api_core.exceptions import AlreadyExists
from google.cloud import pubsub_v1

log = logging.getLogger("consumer")


class PubSubConsumer:
    def __init__(self, project: str, topic: str, subscription: str, handler):
        self.handler = handler
        self.topic_path = pubsub_v1.PublisherClient.topic_path(project, topic)
        self.sub_path = pubsub_v1.SubscriberClient.subscription_path(project, subscription)
        self.subscriber = pubsub_v1.SubscriberClient()
        self.future = None

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _ensure_resources(self) -> None:
        publisher = pubsub_v1.PublisherClient()
        for _ in range(60):  # emulator may still be starting
            try:
                try:
                    publisher.create_topic(request={"name": self.topic_path})
                except AlreadyExists:
                    pass
                try:
                    self.subscriber.create_subscription(request={
                        "name": self.sub_path, "topic": self.topic_path, "ack_deadline_seconds": 30})
                except AlreadyExists:
                    pass
                return
            except Exception as exc:
                log.info("waiting for pubsub: %s", exc)
                time.sleep(1)
        raise RuntimeError("Pub/Sub not reachable")

    def _run(self) -> None:
        self._ensure_resources()
        flow = pubsub_v1.types.FlowControl(max_messages=100)
        self.future = self.subscriber.subscribe(self.sub_path, callback=self._callback, flow_control=flow)
        log.info("consuming from %s", self.sub_path)
        try:
            self.future.result()
        except Exception:
            log.exception("subscriber stopped")

    def _callback(self, message) -> None:
        try:
            self.handler(message.data)
            message.ack()
        except Exception:
            log.exception("processing failed, will retry")
            message.nack()

    def stop(self) -> None:
        if self.future:
            self.future.cancel()
