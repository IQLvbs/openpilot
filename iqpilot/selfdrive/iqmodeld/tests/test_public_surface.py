"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""
from iqpilot.selfdrive.iqmodeld import metadata, messaging, parser
from iqpilot.selfdrive.iqmodeld.daemon import CaptureStamp, NeuralEngineState


def test_public_module_surface():
  assert hasattr(messaging, "DrivePacketMemory")
  assert hasattr(messaging, "pick_curvature")
  assert hasattr(messaging, "populate_drive_messages")
  assert hasattr(messaging, "populate_odometry_message")

  assert hasattr(parser, "ArchiveParser")
  assert hasattr(parser, "PhaseParser")

  assert hasattr(metadata, "select_meta_layout")
  assert hasattr(metadata, "build_metadata_record")

  assert CaptureStamp.__name__ == "CaptureStamp"
  assert NeuralEngineState.__name__ == "NeuralEngineState"
