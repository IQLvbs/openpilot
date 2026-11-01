# Copyright (c) 2026 IQ.Lvbs. All rights reserved.

from iqpilot.cereal import messaging


class PhoneGPSMux:
  def __init__(self, service, publish):
    self.service = service
    self.publish = publish
    self.latest_device = None
    self.last_device_sent = None
    self.last_phone_sent = None
    self.last_published_at = 0.0

  def device(self, packet, phone_fix):
    self.latest_device = packet
    if phone_fix is None:
      self.publish(self.service, packet)
      self.last_device_sent = packet
      self.last_published_at = messaging.log_from_bytes(packet).logMonoTime / 1e9

  def update(self, phone_fix, now):
    if phone_fix is None:
      if self.latest_device is not None and self.latest_device is not self.last_device_sent:
        measured_at = messaging.log_from_bytes(self.latest_device).logMonoTime / 1e9
        if self.last_published_at < measured_at <= now and now - measured_at <= 2:
          self.device(self.latest_device, None)
      return
    if phone_fix is self.last_phone_sent:
      return
    message = messaging.new_message(self.service, valid=True, logMonoTime=int(phone_fix.measured_at * 1e9))
    setattr(message, self.service, phone_fix.gps)
    self.publish(self.service, message)
    self.last_phone_sent = phone_fix
    self.last_published_at = phone_fix.measured_at
