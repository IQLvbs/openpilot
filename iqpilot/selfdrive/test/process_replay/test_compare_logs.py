import iqpilot.cereal.messaging as messaging
from iqpilot.selfdrive.test.process_replay.compare_logs import remove_ignored_fields


def test_remove_ignored_text_field():
  msg = messaging.new_message("selfdriveState")
  msg.selfdriveState.alertText1 = "IQ.Pilot"
  cleared = remove_ignored_fields(msg.as_reader(), ["selfdriveState.alertText1"])
  assert cleared.selfdriveState.alertText1 == ""


def test_remove_ignored_enum_field():
  msg = messaging.new_message("selfdriveState")
  msg.selfdriveState.alertStatus = "userPrompt"
  cleared = remove_ignored_fields(msg.as_reader(), ["selfdriveState.alertStatus"])
  assert str(cleared.selfdriveState.alertStatus) == "normal"
