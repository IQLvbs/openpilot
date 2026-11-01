import queue
import threading
import time
import pyray as rl
from collections.abc import Callable
from iqpilot.common.params import Params


from iqpilot.selfdrive.ui.widgets.screen_header import ScreenHeader, HEADER_HEIGHT
from iqpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from iqpilot.system.ui.lib.multilang import tr
from iqpilot.system.ui.lib.text_measure import measure_text_cached
from iqpilot.system.ui.widgets import Widget
from iqpilot.system.ui.widgets.keyboard import Keyboard
from iqpilot.selfdrive.ui.lib import nav_search
from iqpilot.selfdrive.ui.lib.nav_search import NavSearch, SearchResult
from iqpilot.selfdrive.ui.lib.nav_helpers import active_navigation_route
from iqpilot.selfdrive.ui.widgets.interactive_map import InteractiveNavMap

MARGIN = 40
SPACING = 25
SEARCH_HEIGHT = 120

PANEL_BG = rl.Color(38, 40, 46, 255)
PANEL_BORDER = rl.Color(255, 255, 255, 38)
MUTED = rl.Color(165, 165, 170, 255)


ROW_HEIGHT = 116
ROW_GAP = 16
RESULT_NAME = rl.Color(240, 240, 244, 255)


class NavLayout(Widget):
  """Offroad destination search, saved places and offline map selection."""

  def __init__(self):
    super().__init__()
    self._params = Params()
    self._header = self._child(ScreenHeader(lambda: tr("Navigation")))
    self._search_icon = gui_app.texture("icons/iq/search.png", 52, 52, keep_aspect_ratio=True)
    self._pin_icon = gui_app.texture("icons/iq/pin.png", 90, 90, keep_aspect_ratio=True)
    self._home_icon = gui_app.texture("icons/iq/home.png", 52, 52, keep_aspect_ratio=True)
    self._work_icon = gui_app.texture("icons/iq/work.png", 52, 52, keep_aspect_ratio=True)
    self._recent_icon = gui_app.texture("icons/iq/recent.png", 52, 52, keep_aspect_ratio=True)

    self._keyboard = Keyboard(max_text_size=128, min_text_size=0)
    self._imap = self._child(InteractiveNavMap(on_destination=self._navigate_coordinates))
    self._search = NavSearch()

    self._on_back_cb: Callable[[], None] | None = None
    self._mode = "browse"          # "browse" | "results"
    self._purpose = "navigate"     # "navigate" | "set_home" | "set_work"
    self._query = ""
    self._selecting = False
    self._status_ts = 0.0
    self._selection_generation = 0
    self._selection_results: queue.SimpleQueue[tuple[int, SearchResult | None, str]] = queue.SimpleQueue()
    self._status_msg = ""

    self._home: SearchResult | None = None
    self._work: SearchResult | None = None
    self._recents: list[SearchResult] = []

    self._tap_targets: list[tuple[rl.Rectangle, Callable[[], None]]] = []
    self._reload_favorites()

  def _reload_favorites(self):
    self._home = nav_search.get_home()
    self._work = nav_search.get_work()
    self._recents = nav_search.get_recents()

  def set_on_back(self, cb: Callable[[], None]) -> None:
    self._on_back_cb = cb
    self._header.set_on_back(self._handle_back)

  def show_event(self):
    super().show_event()
    self._mode = "browse"
    self._status_msg = ""
    self._reload_favorites()

  def hide_event(self):
    super().hide_event()
    self._search.new_session()
    self._selection_generation += 1
    self._selecting = False

  def _handle_back(self):
    if self._mode == "results":
      self._exit_search()
    elif self._on_back_cb is not None:
      self._on_back_cb()

  def _open_search(self, purpose: str = "navigate"):
    self._purpose = purpose
    self._search.new_session()
    self._keyboard.reset(min_text_size=1)
    title = {"set_home": tr("Set Home"), "set_work": tr("Set Work")}.get(purpose, tr("Search"))
    self._keyboard.set_title(title, tr("Enter an address or place"))
    self._keyboard.set_text(self._query if purpose == "navigate" else "")
    gui_app.set_modal_overlay(self._keyboard, callback=self._on_search_done)

  def _on_search_done(self, result: int):
    if result != 1:
      return
    self._query = self._keyboard.text.strip()
    if not self._query:
      return
    self._search.search(self._query)
    self._mode = "results"
    self._status_msg = ""
    self._selecting = False

  def _exit_search(self):
    self._selection_generation += 1
    self._mode = "browse"
    self._selecting = False
    self._status_msg = ""
    self._reload_favorites()

  def _select_result(self, r: SearchResult):
    if self._selecting:
      return
    self._selecting = True
    self._status_msg = tr("Locating...")
    generation = self._selection_generation + 1
    self._selection_generation = generation
    threading.Thread(target=self._finish_select, args=(r, self._purpose, generation), daemon=True).start()

  def _finish_select(self, r: SearchResult, purpose: str, generation: int):
    full = self._search.retrieve(r)
    if generation != self._selection_generation:
      return
    self._selection_results.put((generation, full, purpose))

  def _consume_selection(self):
    try:
      result = self._selection_results.get_nowait()
    except queue.Empty:
      return
    generation, full, purpose = result
    if generation != self._selection_generation:
      return
    self._selecting = False
    if full is None or not full.has_coords:
      self._set_status(tr("Couldn't locate that place"))
      return
    if purpose == "set_home":
      nav_search.save_home(full)
    elif purpose == "set_work":
      nav_search.save_work(full)
    else:
      nav_search.set_destination(full.lat, full.lon, full.name)
      nav_search.add_recent(full)
    self._exit_search()
    self._dest_check_time = 0.0
    self._set_status(tr("Address saved") if purpose != "navigate" else "")

  def _navigate_coordinates(self, lat: float, lon: float):
    self._navigate_place(SearchResult(tr("Map destination"), "", lat=lat, lon=lon))

  def _navigate_place(self, place: SearchResult):
    if place is not None and place.has_coords:
      nav_search.set_destination(place.lat, place.lon, place.name)
      nav_search.add_recent(place)
      self._reload_favorites()
      self._dest_check_time = 0.0
      self._set_status("")

  def _set_status(self, msg: str):
    self._status_msg = msg
    self._status_ts = time.monotonic()

  def _cancel_nav(self):
    nav_search.cancel_navigation()
    self._dest_check_time = 0.0
    self._set_status(tr("Navigation canceled"))

  def _remove_home(self):
    nav_search.remove_home()
    self._reload_favorites()

  def _remove_work(self):
    nav_search.remove_work()
    self._reload_favorites()

  def _remove_recent(self, r: SearchResult):
    nav_search.remove_recent(r)
    self._reload_favorites()

  def _on_home(self):
    self._navigate_place(self._home) if self._home is not None else self._open_search("set_home")

  def _on_work(self):
    self._navigate_place(self._work) if self._work is not None else self._open_search("set_work")

  def _render(self, rect: rl.Rectangle):
    self._tap_targets = []
    self._consume_selection()
    header_rect = rl.Rectangle(rect.x + MARGIN, rect.y + MARGIN, rect.width - 2 * MARGIN, HEADER_HEIGHT)
    self._header.render(header_rect)
    body = rl.Rectangle(rect.x + MARGIN, header_rect.y + HEADER_HEIGHT + SPACING,
                        rect.width - 2 * MARGIN, rect.y + rect.height - (header_rect.y + HEADER_HEIGHT + SPACING) - MARGIN)
    if self._mode == "results":
      self._render_results(body)
    else:
      self._render_browse(body)

  def _row(self, rect: rl.Rectangle, icon, title: str, subtitle: str, on_tap, on_delete=None, pressed_hint=True):
    hit = rl.check_collision_point_rec(rl.get_mouse_position(), rect)
    bg = rl.Color(54, 57, 65, 255) if (hit and pressed_hint and rl.is_mouse_button_down(0)) else PANEL_BG
    rl.draw_rectangle_rounded(rect, 0.35, 20, bg)
    rl.draw_rectangle_rounded_lines_ex(rect, 0.35, 20, 2, PANEL_BORDER)
    x = rect.x + 32
    if icon is not None:
      rl.draw_texture(icon, int(x), int(rect.y + rect.height / 2 - icon.height / 2), rl.WHITE)
      x += icon.width + 24
    text_w = rect.width - (x - rect.x) - (110 if on_delete is not None else 32)
    font = gui_app.font(FontWeight.MEDIUM)
    if subtitle:
      rl.draw_text_ex(font, title, rl.Vector2(int(x), int(rect.y + 22)), 40, 0, RESULT_NAME)
      sub = self._ellipsize(font, subtitle, 30, text_w)
      rl.draw_text_ex(font, sub, rl.Vector2(int(x), int(rect.y + 66)), 30, 0, MUTED)
    else:
      ts = measure_text_cached(font, title, 42)
      rl.draw_text_ex(font, title, rl.Vector2(int(x), int(rect.y + rect.height / 2 - ts.y / 2)), 42, 0, RESULT_NAME)
    # Delete (×) button — appended first so a tap on it wins over the row's navigate tap.
    if on_delete is not None:
      cx, cy = rect.x + rect.width - 60, rect.y + rect.height / 2
      del_r = rl.Rectangle(cx - 34, cy - 34, 68, 68)
      dhit = rl.check_collision_point_rec(rl.get_mouse_position(), del_r)
      rl.draw_circle(int(cx), int(cy), 30, rl.Color(90, 62, 66, 255) if dhit else rl.Color(60, 62, 70, 255))
      rl.draw_line_ex(rl.Vector2(cx - 13, cy - 13), rl.Vector2(cx + 13, cy + 13), 4, rl.Color(230, 120, 120, 255))
      rl.draw_line_ex(rl.Vector2(cx - 13, cy + 13), rl.Vector2(cx + 13, cy - 13), 4, rl.Color(230, 120, 120, 255))
      self._tap_targets.append((del_r, on_delete))
    if on_tap is not None:
      self._tap_targets.append((rect, on_tap))

  @staticmethod
  def _ellipsize(font, text: str, size: int, max_w: float) -> str:
    if measure_text_cached(font, text, size).x <= max_w:
      return text
    while text and measure_text_cached(font, text + "...", size).x > max_w:
      text = text[:-1]
    return text + "..."

  def _render_browse(self, rect: rl.Rectangle):
    font = gui_app.font(FontWeight.MEDIUM)
    x, w = rect.x, rect.width
    y = rect.y
    bar = rl.Rectangle(x, y, w, SEARCH_HEIGHT)
    rl.draw_rectangle_rounded(bar, 0.4, 20, PANEL_BG)
    rl.draw_rectangle_rounded_lines_ex(bar, 0.4, 20, 2, PANEL_BORDER)
    cy = bar.y + bar.height / 2
    rl.draw_texture(self._search_icon, int(bar.x + 36), int(cy - self._search_icon.height / 2), MUTED)
    ph = tr("Search address or place")
    rl.draw_text_ex(font, ph, rl.Vector2(int(bar.x + 36 + self._search_icon.width + 24),
                    int(cy - 22)), 44, 0, MUTED)
    self._tap_targets.append((bar, lambda: self._open_search("navigate")))
    y += SEARCH_HEIGHT + SPACING

    # Cancel active route (param read throttled)
    now = time.monotonic()
    if now - getattr(self, "_dest_check_time", 0.0) > 1.0:
      params = self._params
      destination = params.get("NavigationDestination")
      self._has_dest = bool(destination)
      self._route_status = ""
      route = active_navigation_route(params)
      if route:
        minutes = max(1, round(route.get("durationSeconds", 0) / 60))
        self._route_status = tr("Route ready") + f" · {minutes} min"
      elif destination:
        payload = params.get("NavigationRenderRoute") or {}
        status = payload.get("status") if payload.get("requestedDestination") == destination else "calculating"
        self._route_status = {
          "failed": tr("Couldn't calculate route. Check routing settings or connection."),
          "waitingForPosition": tr("Waiting for a GPS position to plan the route..."),
        }.get(status, tr("Calculating route..."))
      self._dest_check_time = now
    if getattr(self, "_has_dest", False):
      cr = rl.Rectangle(x, y, w, ROW_HEIGHT)
      chit = rl.check_collision_point_rec(rl.get_mouse_position(), cr)
      rl.draw_rectangle_rounded(cr, 0.35, 20, rl.Color(96, 46, 48, 255) if chit else rl.Color(74, 40, 42, 255))
      rl.draw_rectangle_rounded_lines_ex(cr, 0.35, 20, 2, rl.Color(210, 90, 90, 120))
      label = self._ellipsize(font, self._route_status, 38, w - 450)
      rl.draw_text_ex(font, label, rl.Vector2(x + 32, cr.y + 35), 38, 0, rl.WHITE)
      cancel = tr("Cancel navigation")
      cancel_width = measure_text_cached(font, cancel, 38).x
      rl.draw_text_ex(font, cancel, rl.Vector2(x + w - cancel_width - 32, cr.y + 35), 38, 0,
                     rl.Color(240, 180, 180, 255))
      self._tap_targets.append((rl.Rectangle(x + w - cancel_width - 64, cr.y, cancel_width + 64, cr.height), self._cancel_nav))
      y += ROW_HEIGHT + SPACING

    hw_gap = ROW_GAP
    hw_w = (w - hw_gap) / 2
    self._row(rl.Rectangle(x, y, hw_w, ROW_HEIGHT), self._home_icon, tr("Home"),
              self._home.address if self._home else tr("Set home address"), self._on_home,
              on_delete=(self._remove_home if self._home else None))
    self._row(rl.Rectangle(x + hw_w + hw_gap, y, hw_w, ROW_HEIGHT), self._work_icon, tr("Work"),
              self._work.address if self._work else tr("Set work address"), self._on_work,
              on_delete=(self._remove_work if self._work else None))
    y += ROW_HEIGHT + SPACING

    # Recents (fit as many as room allows, leaving space for the map)
    map_min = 300
    for r in self._recents:
      if y + ROW_HEIGHT > rect.y + rect.height - map_min - SPACING:
        break
      self._row(rl.Rectangle(x, y, w, ROW_HEIGHT), self._recent_icon, r.name, r.address,
                (lambda r=r: self._navigate_place(r)), on_delete=(lambda r=r: self._remove_recent(r)))
      y += ROW_HEIGHT + ROW_GAP

    map_rect = rl.Rectangle(x, y, w, rect.y + rect.height - y)
    if map_rect.height > 120:
      self._imap.render(map_rect)
    if self._status_msg and time.monotonic() - self._status_ts < 2.5:
      self._draw_toast(map_rect, self._status_msg)

  def _draw_toast(self, area: rl.Rectangle, text: str):
    font = gui_app.font(FontWeight.MEDIUM)
    ts = measure_text_cached(font, text, 36)
    pad = 32
    pw = ts.x + pad * 2
    pill = rl.Rectangle(area.x + (area.width - pw) / 2, area.y + 24, pw, 66)
    rl.draw_rectangle_rounded(pill, 0.5, 20, rl.Color(20, 22, 26, 235))
    rl.draw_rectangle_rounded_lines_ex(pill, 0.5, 20, 2, PANEL_BORDER)
    rl.draw_text_ex(font, text, rl.Vector2(int(pill.x + pad), int(pill.y + 33 - ts.y / 2)), 36, 0, rl.WHITE)


  def _draw_center_note(self, rect: rl.Rectangle, text: str, dy: float = 0):
    font = gui_app.font(FontWeight.MEDIUM)
    size = min(40, max(20, int((rect.width - 40) / max(1, measure_text_cached(font, text, 40).x) * 40)))
    ns = measure_text_cached(font, text, size)
    rl.draw_text_ex(font, text, rl.Vector2(int(rect.x + (rect.width - ns.x) / 2),
                    int(rect.y + rect.height / 2 + dy)), size, 0, MUTED)

  def _render_results(self, rect: rl.Rectangle):
    font = gui_app.font(FontWeight.MEDIUM)
    x, w = rect.x, rect.width
    y = rect.y
    # Query bar — tap to reopen the keyboard and refine the search.
    bar = rl.Rectangle(x, y, w, SEARCH_HEIGHT)
    rl.draw_rectangle_rounded(bar, 0.4, 20, PANEL_BG)
    rl.draw_rectangle_rounded_lines_ex(bar, 0.4, 20, 2, PANEL_BORDER)
    cy = bar.y + bar.height / 2
    rl.draw_texture(self._search_icon, int(bar.x + 36), int(cy - self._search_icon.height / 2), MUTED)
    rl.draw_text_ex(font, self._query or tr("Search"),
                    rl.Vector2(int(bar.x + 36 + self._search_icon.width + 24), int(cy - 22)), 44, 0, rl.WHITE)
    self._tap_targets.append((bar, lambda: self._open_search(self._purpose)))
    y += SEARCH_HEIGHT + SPACING

    list_rect = rl.Rectangle(x, y, w, rect.y + rect.height - y)
    results = self._search.results()
    if self._selecting:
      self._draw_center_note(list_rect, self._status_msg or tr("Locating..."))
      return
    if not results:
      note = tr("Searching...") if self._search.searching else (self._status_msg or tr(self._search.error) or tr("No results"))
      self._draw_center_note(list_rect, note)
      return
    for r in results:
      if y + ROW_HEIGHT > list_rect.y + list_rect.height:
        break
      dist = f"{r.distance_m / 1609.34:.1f} mi" if r.distance_m else ""
      sub = f"{r.address}   ·   {dist}" if dist else r.address
      self._row(rl.Rectangle(x, y, w, ROW_HEIGHT), self._pin_icon, r.name, sub,
                (lambda r=r: self._select_result(r)))
      y += ROW_HEIGHT + ROW_GAP

  def _handle_mouse_release(self, mouse_pos: MousePos):
    for rect, cb in self._tap_targets:
      if rl.check_collision_point_rec(mouse_pos, rect):
        cb()
        return
