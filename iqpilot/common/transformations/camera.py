import itertools
import numpy as np
from dataclasses import dataclass

import iqpilot.common.transformations.orientation as orient

## -- hardcoded hardware params --
@dataclass(frozen=True)
class CameraConfig:
  width: int
  height: int
  focal_length: float

  @property
  def size(self):
    return (self.width, self.height)

  @property
  def intrinsics(self):
    # aka 'K' aka camera_frame_from_view_frame
    return np.array([
      [self.focal_length,  0.0, float(self.width)/2],
      [0.0, self.focal_length, float(self.height)/2],
      [0.0,  0.0, 1.0]
    ])

  @property
  def intrinsics_inv(self):
    # aka 'K_inv' aka view_frame_from_camera_frame
    return np.linalg.inv(self.intrinsics)

@dataclass(frozen=True)
class _NoneCameraConfig(CameraConfig):
  width: int = 0
  height: int = 0
  focal_length: float = 0

@dataclass(frozen=True)
class DeviceCameraConfig:
  fcam: CameraConfig
  dcam: CameraConfig
  ecam: CameraConfig

  def all_cams(self):
    for cam in ['fcam', 'dcam', 'ecam']:
      if not isinstance(getattr(self, cam), _NoneCameraConfig):
        yield cam, getattr(self, cam)

_ar_ox_fisheye = CameraConfig(1928, 1208, 567.0)  # focal length probably wrong? magnification is not consistent across frame
_os_fisheye = CameraConfig(2688 // 2, 1520 // 2, 567.0 / 4 * 3)
_ar_ox_config = DeviceCameraConfig(CameraConfig(1928, 1208, 2648.0), _ar_ox_fisheye, _ar_ox_fisheye)
_os_config = DeviceCameraConfig(CameraConfig(2688 // 2, 1520 // 2, 1522.0 * 3 / 4), _os_fisheye, _os_fisheye)
_neo_config = DeviceCameraConfig(CameraConfig(1164, 874, 910.0), CameraConfig(816, 612, 650.0), _NoneCameraConfig())

DEVICE_CAMERAS = {
  # A "device camera" is defined by a device type and sensor

  # sensor type was never set on eon/neo/two
  ("neo", "unknown"): _neo_config,
  # unknown here is AR0231, field was added with OX03C10 support
  ("tici", "unknown"): _ar_ox_config,

  # before deviceState.deviceType was set, assume tici AR config
  ("unknown", "ar0231"): _ar_ox_config,
  ("unknown", "ox03c10"): _ar_ox_config,

  # simulator (emulates a tici)
  ("pc", "unknown"): _ar_ox_config,
}
prods = itertools.product(('tici', 'tizi', 'mici'), (('ar0231', _ar_ox_config), ('ox03c10', _ar_ox_config), ('os04c10', _os_config)))
DEVICE_CAMERAS.update({(d, c[0]): c[1] for d, c in prods})

# device/mesh : x->forward, y-> right, z->down
# view : x->right, y->down, z->forward
device_frame_from_view_frame = np.array([
  [ 0.,  0.,  1.],
  [ 1.,  0.,  0.],
  [ 0.,  1.,  0.]
])
view_frame_from_device_frame = device_frame_from_view_frame.T


# aka 'extrinsic_matrix'
def get_view_frame_from_calib_frame(roll, pitch, yaw, height):
  device_from_calib= orient.rot_from_euler([roll, pitch, yaw])
  view_from_calib = view_frame_from_device_frame.dot(device_from_calib)
  return np.hstack((view_from_calib, [[0], [height], [0]]))
