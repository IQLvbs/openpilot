import math
import time

import pyray as rl

from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.ui.lib.application import FontWeight, gui_app
from iqpilot.system.ui.lib.text_measure import measure_text_cached
from iqpilot.system.ui.lib.wrap_text import wrap_text
from iqpilot.system.ui.lib.multilang import tr

RETURN_DISTANCE_M = 0.9 * 1609.344
DOUBLE_TAP_SECONDS = 0.35


class MiciNavigation:
  def __init__(self):
    self.active = False
    self._destination = None
    self._distance = ""
    self._street = ""
    self._icon = None
    self._dismissed = False
    self._dismissed_maneuver = None
    self._maneuver = None
    self._maneuver_distance = math.inf
    self._last_tap = -math.inf
    self._layout_key = None
    self._lines = []
    self._distance_font = gui_app.font(FontWeight.BOLD)
    self._street_font = gui_app.font(FontWeight.MEDIUM)

  def update(self):
    self.active = False
    navigation_active = (ui_state.started and ui_state.params.get_bool("NavigationEnabled") and
                         ui_state.params.get_bool("NavigationActive"))
    destination = ui_state.params.get("NavigationDestination")
    try:
      lat, lon = float(destination['latitude']), float(destination['longitude'])
      destination = (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None
    except (KeyError, TypeError, ValueError, OverflowError):
      destination = None
    if destination != self._destination or not navigation_active:
      self._dismissed = False
      self._dismissed_maneuver = None
      self._last_tap = -math.inf
    self._destination = destination
    if destination is None or not navigation_active:
      return

    sm = ui_state.sm
    nav = sm['iqNavState']
    fresh = sm.alive['iqNavState'] and sm.valid['iqNavState']
    matches = (nav.destinationValid and
               abs(nav.destinationLatitude - destination[0]) < 1e-5 and
               abs(nav.destinationLongitude - destination[1]) < 1e-5)
    distance = nav.nextManeuverDistance
    if not (fresh and nav.active and matches and nav.nextManeuverValid and math.isfinite(distance) and distance >= 0):
      self._last_tap = -math.inf
      return

    instruction = (nav.nextManeuverType.raw, nav.nextManeuverDirection.raw, nav.nextManeuverDescription)
    position = None
    # Coordinates distinguish consecutive turns with the same instruction. Guidance can
    # still display if the optional map-render stream is unavailable.
    if sm.alive['iqNavRenderState'] and sm.valid['iqNavRenderState']:
      render = sm['iqNavRenderState']
      if (render.active and abs(render.destinationLatitude - destination[0]) < 1e-5 and
          abs(render.destinationLongitude - destination[1]) < 1e-5 and
          math.isfinite(render.nextManeuverLatitude) and math.isfinite(render.nextManeuverLongitude) and
          (render.nextManeuverLatitude or render.nextManeuverLongitude)):
        position = (round(render.nextManeuverLatitude, 5), round(render.nextManeuverLongitude, 5))
    maneuver = (instruction, position)
    if distance > self._maneuver_distance + 100:
      self._dismissed_maneuver = None
    self._maneuver, self._maneuver_distance = maneuver, distance
    if self._dismissed:
      same_maneuver = (self._dismissed_maneuver is not None and instruction == self._dismissed_maneuver[0] and
                       (position is None or self._dismissed_maneuver[1] is None or position == self._dismissed_maneuver[1]))
      if distance > RETURN_DISTANCE_M or same_maneuver:
        return
      self._dismissed = False
    self.active = True

    if ui_state.is_metric:
      self._distance = f"{distance / 1000:.1f} km" if distance >= 1000 else f"{round(distance / 10) * 10} m"
    else:
      feet = distance / 0.3048
      self._distance = f"{distance / 1609.344:.1f} mi" if feet >= 1000 else f"{round(feet / 10) * 10} ft"
    self._street = " ".join(nav.nextManeuverDescription.split()) or "Continue on route"
    direction = 'left' if nav.nextManeuverDirection.raw == 1 else 'right'
    kind = {1: 'turn', 2: 'off_ramp', 3: 'merge', 4: 'fork', 7: 'turn'}.get(nav.nextManeuverType.raw, 'continue')
    self._icon = 'direction_arrive.png' if nav.nextManeuverType.raw == 6 else f'direction_{kind}_{direction}.png'

  def tap(self):
    if not self.active:
      return
    now = time.monotonic()
    if now - self._last_tap <= DOUBLE_TAP_SECONDS:
      self._dismissed = True
      self._dismissed_maneuver = self._maneuver if self._maneuver_distance <= RETURN_DISTANCE_M else None
      self.active = False
      self._last_tap = -math.inf
    else:
      self._last_tap = now

  def render(self, rect):
    rl.draw_rectangle_rec(rect, rl.BLACK)
    text_x = rect.x + (140 if self._icon else 24)
    width = max(1, rect.width - (text_x - rect.x) - 20)
    key = (self._street, width)
    if key != self._layout_key:
      self._layout_key = key
      lines = wrap_text(self._street_font, self._street, 30, width)
      self._lines = lines[:2]
      if len(lines) > 2:
        line = self._lines[-1]
        while line and measure_text_cached(self._street_font, line + '...', 30).x > width:
          line = line[:-1]
        self._lines[-1] = line + '...'
    if self._icon:
      icon = gui_app.texture('navigation/' + self._icon, 96, 96)
      rl.draw_texture(icon, int(rect.x + 24), int(rect.y + (rect.height - 96) / 2), rl.WHITE)
    top = rect.y + (rect.height - 28 - (64 if self._distance else 0) - 36 * len(self._lines)) / 2
    if self._distance:
      rl.draw_text_ex(self._distance_font, self._distance, rl.Vector2(text_x, top), 54, 0, rl.WHITE)
      top += 64
    for line in self._lines:
      rl.draw_text_ex(self._street_font, line, rl.Vector2(text_x, top), 30, 0, rl.WHITE)
      top += 36
    hint = tr("double tap to dismiss")
    hint_width = measure_text_cached(self._street_font, hint, 18).x
    rl.draw_text_ex(self._street_font, hint, rl.Vector2(rect.x + (rect.width - hint_width) / 2, rect.y + rect.height - 26),
                    18, 0, rl.Color(160, 160, 160, 255))
