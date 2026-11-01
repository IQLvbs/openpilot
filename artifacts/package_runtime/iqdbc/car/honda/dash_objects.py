"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import math
from dataclasses import dataclass

from iqdbc.can.parser import CANParser
from iqdbc.car.honda import dash_lane

NUM_SLOTS = 10
LONG_DIST_CAP_M = 195.0

# byte-faithful empty-slot payload decoded from stock HUD_OBJECTS; an inconsistent frame risks the dash rejecting it
INACTIVE = {
  "OBJECT_ID": 0,
  "IS_LEAD_CAR": 0,
  "CAR_TYPE": -1,
  "ROTATION": -128,
  "LONG_DIST": 196.9,
  "LAT_DIST": 204.7,
}

CAR_TYPE_CAR = 7
LONG_DIST_MAX_M = 194.0
LAT_DIST_LIM_M = 204.7

# the dash under-scales LAT_DIST ~0.3x in the ego frame; tuned on-car so the lead marker lands on the lane
LAT_SCALE = 0.35

ROT_BAND_M = 1.5
ROT_MAX = 6

REID_GAP_M = 8.0
REID_TAU = 1.5
REID_REFRACTORY = 1.5
MAX_OBJECT_ID = 31

DREL_SMOOTH_TAU = 0.6
YREL_SMOOTH_TAU = 0.5
FF_VREL_MIN = 0.5
DREL_RESID_CLAMP = 1.5

LEAD_PROB_ON = 0.5
LEAD_PROB_OFF = 0.35
LEAD_HOLD_S = 0.6

# modelV2.leadsV3 entries are one car at three time horizons, not three cars: only render the extra
# horizons when spatially distinct from everything already rendered (a genuinely different vehicle)
EXTRA_LEAD_SLOTS = (1, 2)
EXTRA_LEAD_MIN_SEP_D = 5.0
EXTRA_LEAD_MIN_SEP_Y = 1.5


@dataclass
class CameraObject:
  slot: int
  object_id: int
  d_rel: float
  y_rel: float
  is_lead_car: bool
  valid: bool
  car_type: int = -1
  rotation: int = -128


class CameraObjectTracker:
  def __init__(self):
    self._tracks: list[CameraObject] = [
      CameraObject(slot=i, object_id=0, d_rel=0.0, y_rel=0.0, is_lead_car=False, valid=False)
      for i in range(NUM_SLOTS)
    ]

  def update(self, cp_cam: CANParser) -> None:
    vla = cp_cam.vl_all["HUD_OBJECTS"]
    for mux, oid, ld, yd, lead, ct, rot in zip(vla["MUX"], vla["OBJECT_ID"], vla["LONG_DIST"], vla["LAT_DIST"],
                                               vla["IS_LEAD_CAR"], vla["CAR_TYPE"], vla["ROTATION"], strict=True):
      slot = (int(mux) - 1) % 16
      if 0 <= slot < NUM_SLOTS:
        self._tracks[slot] = CameraObject(
          slot=slot,
          object_id=int(oid),
          d_rel=float(ld),
          y_rel=float(yd),
          is_lead_car=bool(lead),
          valid=oid != 0 and ld < LONG_DIST_CAP_M,
          car_type=int(ct),
          rotation=int(rot),
        )

  def snapshot(self) -> list[CameraObject]:
    return self._tracks


@dataclass
class ModelLead:
  status: bool
  dRel: float
  yRel: float
  vRel: float
  prob: float = 0.0


def leads_from_model(model, v_ego, n=3):
  # modelV2's lateral is +right; the dash convention is +left. v is made relative for the smoother.
  # Data stays populated below LEAD_PROB_ON (status False, prob carried) so the author's hysteresis
  # can keep an already-rendered lead alive down to LEAD_PROB_OFF instead of blinking it
  out = []
  for i in range(n):
    if model is None or len(model.leadsV3) <= i or len(model.leadsV3[i].x) == 0:
      out.append(ModelLead(False, 0.0, 0.0, 0.0))
      continue
    lead = model.leadsV3[i]
    out.append(ModelLead(bool(lead.prob >= LEAD_PROB_ON), float(lead.x[0]), -float(lead.y[0]),
                         float(lead.v[0]) - v_ego, prob=float(lead.prob)))
  return out


def lead_rotation(lateral_left_m: float) -> int:
  magnitude = min(round(abs(lateral_left_m) / ROT_BAND_M), ROT_MAX)
  return -magnitude if lateral_left_m > 0 else magnitude


class LeadIdentity:
  """Mints a stable OBJECT_ID for the rendered lead, re-IDing on a fresh lead or a range discontinuity.
  dRel is noisy, so a leaky predictor (feed-forward vRel, leak toward dRel) accumulates the residual
  instead of a per-sample range-rate test."""

  def __init__(self):
    self.object_id = 0
    self._on = False
    self._pred = 0.0
    self._prev_t = 0.0
    self._reid_t = -1e9

  def update(self, status: bool, d_rel: float, v_rel: float, now: float) -> int:
    if not status:
      self.object_id = 0
      self._on = False
      return 0

    new_lead = not self._on
    if self._on:
      dt = max(now - self._prev_t, 1e-3)
      self._pred += v_rel * dt
      self._pred += min(dt / REID_TAU, 1.0) * (d_rel - self._pred)
      if abs(d_rel - self._pred) > REID_GAP_M and now - self._reid_t > REID_REFRACTORY:
        new_lead = True
    self._prev_t = now

    if new_lead:
      self.object_id = self.object_id % MAX_OBJECT_ID + 1
      self._reid_t = now
      self._pred = d_rel
    self._on = True
    return self.object_id


class MarkerSmoother:
  """Stabilizes a rendered marker without lagging real motion: vRel feed-forward on dRel with a
  clamped leak toward the measurement, plain low-pass on yRel, snapping on an identity change."""

  def __init__(self):
    self._id = 0
    self._d = 0.0
    self._y = 0.0
    self._t = 0.0

  def update(self, d_rel: float, y_rel: float, v_rel: float, object_id: int, now: float) -> tuple[float, float]:
    if object_id != self._id:
      self._id, self._d, self._y, self._t = object_id, d_rel, y_rel, now
      return d_rel, y_rel
    dt = max(now - self._t, 1e-3)
    self._t = now
    if abs(v_rel) >= FF_VREL_MIN:
      self._d += v_rel * dt
    resid = min(max(d_rel - self._d, -DREL_RESID_CLAMP), DREL_RESID_CLAMP)
    self._d += (1.0 - math.exp(-dt / DREL_SMOOTH_TAU)) * resid
    self._y += (1.0 - math.exp(-dt / YREL_SMOOTH_TAU)) * (y_rel - self._y)
    return self._d, self._y


def create_hud_object(packer, bus, mux, track):
  values = {"MUX": mux}
  if track is None:
    values.update(INACTIVE)
  else:
    values.update({
      "OBJECT_ID": int(track["object_id"]),
      "IS_LEAD_CAR": int(track["is_lead_car"]),
      "CAR_TYPE": int(track["car_type"]),
      "ROTATION": int(track["rotation"]),
      "LONG_DIST": min(max(track["d_rel"], 0.0), LONG_DIST_MAX_M),
      "LAT_DIST": min(max(track["y_rel"], -LAT_DIST_LIM_M), LAT_DIST_LIM_M),
    })
  return packer.make_can_msg("HUD_OBJECTS", bus, values)


def forward_hud_object(packer, bus, mux, tracks):
  slot = (mux - 1) % 16
  st = tracks[slot] if (tracks and slot < len(tracks)) else None
  track = ({"d_rel": st.d_rel, "y_rel": st.y_rel, "object_id": st.object_id, "is_lead_car": st.is_lead_car,
            "car_type": st.car_type, "rotation": st.rotation} if (st is not None and st.valid) else None)
  return create_hud_object(packer, bus, mux, track)


class DashObjectAuthor:
  """Authors HUD_OBJECTS: openpilot's lead in slot 0 with a stable identity and smoothed marker, the
  camera's non-lead cars forwarded in slots 1-9 (or distinct extra model leads where there is no
  camera to forward), one frame per mux tick."""

  def __init__(self):
    self._identity = LeadIdentity()
    self._smoother = MarkerSmoother()
    self._lead_id = 0
    self._prev_op_id = 0
    self._lead_on = False
    self._lead_hold: ModelLead | None = None
    self._lead_seen_t = -1e9
    self._extra_ids = {slot: LeadIdentity() for slot in EXTRA_LEAD_SLOTS}
    self._extra_smooth = {slot: MarkerSmoother() for slot in EXTRA_LEAD_SLOTS}
    self._extra_emit = dict.fromkeys(EXTRA_LEAD_SLOTS, 0)

  def _gate_lead(self, lead: ModelLead, now: float) -> ModelLead:
    # leadsV3[0].prob hovers around 0.5 in traffic; hysteresis plus a short dead-reckoned hold keeps
    # the marker from blinking at a cadence the stock radar never produces
    if lead.prob >= (LEAD_PROB_OFF if self._lead_on else LEAD_PROB_ON):
      self._lead_on = True
      self._lead_hold = lead
      self._lead_seen_t = now
      return lead if lead.status else ModelLead(True, lead.dRel, lead.yRel, lead.vRel, lead.prob)
    if self._lead_on and self._lead_hold is not None and now - self._lead_seen_t < LEAD_HOLD_S:
      h = self._lead_hold
      return ModelLead(True, h.dRel + h.vRel * (now - self._lead_seen_t), h.yRel, h.vRel, h.prob)
    self._lead_on = False
    self._lead_hold = None
    return ModelLead(False, 0.0, 0.0, 0.0)

  def _lead_object_id(self, status: bool, op_id: int, stock_lead_id: int | None, in_use: set[int]) -> int:
    if not status:
      self._lead_id = 0
    elif stock_lead_id is not None:
      self._lead_id = stock_lead_id
    elif self._lead_id == 0 or op_id != self._prev_op_id or self._lead_id in in_use:
      # advance from the current id rather than picking the lowest free one: with no camera ids in
      # use a handoff would keep the same id and the id-keyed smoother would slide between two cars
      # instead of snapping
      nxt = self._lead_id % MAX_OBJECT_ID + 1
      while nxt in in_use:
        nxt = nxt % MAX_OBJECT_ID + 1
      self._lead_id = nxt
    self._prev_op_id = op_id
    return self._lead_id

  def _update_extras(self, extra_leads, lead, in_use, now):
    rendered = [(lead.dRel, lead.yRel)] if lead.status else []
    out = {}
    for slot, ex in zip(EXTRA_LEAD_SLOTS, extra_leads or (), strict=False):
      distinct = ex.status and all(abs(ex.dRel - d) >= EXTRA_LEAD_MIN_SEP_D or
                                   abs(ex.yRel - y) >= EXTRA_LEAD_MIN_SEP_Y
                                   for d, y in rendered)
      op_id = self._extra_ids[slot].update(distinct, ex.dRel, ex.vRel, now)
      if not distinct:
        self._extra_emit[slot] = 0
        out[slot] = None
        continue
      emit = self._extra_emit[slot]
      if emit == 0 or emit in in_use:
        emit = op_id
        while emit in in_use:
          emit = emit % MAX_OBJECT_ID + 1
        self._extra_emit[slot] = emit
      in_use.add(emit)
      d_rel, y_rel = self._extra_smooth[slot].update(ex.dRel, LAT_SCALE * ex.yRel, ex.vRel, emit, now)
      rendered.append((ex.dRel, ex.yRel))
      out[slot] = {"d_rel": d_rel, "y_rel": y_rel, "object_id": emit, "is_lead_car": 0,
                   "car_type": CAR_TYPE_CAR, "rotation": lead_rotation(y_rel / LAT_SCALE)}
    return out

  def create(self, packer, bus, lead, tracks, mux: int, now: float, extra_leads=None):
    lead = self._gate_lead(lead, now)
    op_id = self._identity.update(lead.status, lead.dRel, lead.vRel, now)
    stock_lead, in_use = None, set()
    for t in (tracks or ()):
      if not t.valid:
        continue
      if t.is_lead_car:
        stock_lead = t
      elif t.slot != 0:
        in_use.add(t.object_id)
    stock_lead_id = stock_lead.object_id if stock_lead is not None else None
    lead_id = self._lead_object_id(lead.status, op_id, stock_lead_id, in_use)
    if lead.status:
      in_use.add(lead_id)

    # ride the lane gain-law correction at the lead's distance so the marker tracks the lane rendering
    lat_scale = LAT_SCALE * dash_lane.gain_correction(lead.dRel)
    d_rel, y_rel = self._smoother.update(lead.dRel, lat_scale * lead.yRel, lead.vRel, lead_id, now)

    extras = self._update_extras(extra_leads, lead, in_use, now) if tracks is None else {}

    slot = (mux - 1) % 16
    if slot == 0 and lead.status:
      track = {"d_rel": d_rel, "y_rel": y_rel, "object_id": lead_id, "is_lead_car": 1,
               "car_type": stock_lead.car_type if stock_lead is not None else CAR_TYPE_CAR,
               "rotation": stock_lead.rotation if stock_lead is not None else lead_rotation(y_rel / lat_scale)}
    elif slot in extras:
      track = extras[slot]
    else:
      st = tracks[slot] if (tracks and slot < len(tracks)) else None
      # never forward the camera's lead: if OP has no lead, the HUD must not flag one OP isn't acting on
      track = ({"d_rel": st.d_rel, "y_rel": st.y_rel, "object_id": st.object_id, "is_lead_car": 0,
                "car_type": st.car_type, "rotation": st.rotation}
               if (st is not None and st.valid and not st.is_lead_car) else None)
    return create_hud_object(packer, bus, mux, track)
