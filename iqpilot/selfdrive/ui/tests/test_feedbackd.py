import time

import pytest

import iqpilot.cereal.messaging as messaging
from iqpilot.system.manager.process_config import managed_processes


@pytest.mark.linux
def test_feedbackd_publishes_bookmark():
  publisher = messaging.PubMaster(["bookmarkButton"])
  subscriber = messaging.SubMaster(["userBookmark"])
  process = managed_processes["feedbackd"]
  process.start()
  try:
    assert publisher.wait_for_readers_to_update("bookmarkButton", timeout=5)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not subscriber.updated["userBookmark"]:
      publisher.send("bookmarkButton", messaging.new_message("bookmarkButton"))
      subscriber.update(100)
    assert subscriber.updated["userBookmark"]
  finally:
    process.stop()
