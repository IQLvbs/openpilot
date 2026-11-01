from enum import IntEnum

import pyray as rl
from iqpilot.system.ui.lib.application import FontWeight, gui_app
from iqpilot.system.ui.lib.multilang import tr
from iqpilot.system.ui.widgets import Widget
from iqpilot.system.ui.widgets.button import Button, ButtonStyle
from iqpilot.system.ui.widgets.label import Label
from iqpilot.selfdrive.ui.ui_state import ui_state
from iqpilot.system.version import terms_version

DEBUG = False


class OnboardingState(IntEnum):
  TERMS = 0
  DECLINE = 1


class TermsPage(Widget):
  def __init__(self, on_accept=None, on_decline=None):
    super().__init__()
    self._on_accept = on_accept
    self._on_decline = on_decline

    self._title = Label(tr("Welcome to IQ.Pilot"), font_size=90, font_weight=FontWeight.BOLD, text_alignment=rl.GuiTextAlignment.TEXT_ALIGN_LEFT)
    self._desc = Label(tr("You must accept the Terms of Service to use IQ.Pilot. Read the latest terms before continuing at https://iqlvbs.com/tos."),
                       font_size=90, font_weight=FontWeight.MEDIUM, text_alignment=rl.GuiTextAlignment.TEXT_ALIGN_LEFT)

    self._decline_btn = Button(tr("Decline"), click_callback=on_decline)
    self._accept_btn = Button(tr("Agree"), button_style=ButtonStyle.PRIMARY, click_callback=on_accept)

  def _render(self, _):
    welcome_x = self._rect.x + 95
    welcome_y = self._rect.y + 165
    welcome_rect = rl.Rectangle(welcome_x, welcome_y, self._rect.width - welcome_x, 90)
    self._title.render(welcome_rect)

    desc_x = welcome_x
    # TODO: Label doesn't top align when wrapping
    desc_y = welcome_y - 100
    desc_rect = rl.Rectangle(desc_x, desc_y, self._rect.width - desc_x, self._rect.height - desc_y - 250)
    self._desc.render(desc_rect)

    btn_y = self._rect.y + self._rect.height - 160 - 45
    btn_width = (self._rect.width - 45 * 3) / 2
    self._decline_btn.render(rl.Rectangle(self._rect.x + 45, btn_y, btn_width, 160))
    self._accept_btn.render(rl.Rectangle(self._rect.x + 45 * 2 + btn_width, btn_y, btn_width, 160))

    if DEBUG:
      rl.draw_rectangle_lines_ex(welcome_rect, 3, rl.RED)
      rl.draw_rectangle_lines_ex(desc_rect, 3, rl.RED)

    return -1


class DeclinePage(Widget):
  def __init__(self, back_callback=None):
    super().__init__()
    self._text = Label(tr("You must accept the Terms of Service in order to use IQ.Pilot."),
                       font_size=90, font_weight=FontWeight.MEDIUM, text_alignment=rl.GuiTextAlignment.TEXT_ALIGN_LEFT)
    self._back_btn = Button(tr("Back"), click_callback=back_callback)
    self._uninstall_btn = Button(tr("Decline, uninstall IQ.Pilot"), button_style=ButtonStyle.DANGER,
                                 click_callback=self._on_uninstall_clicked)

  def _on_uninstall_clicked(self):
    ui_state.params.put_bool("DoUninstall", True)
    gui_app.request_close()

  def _render(self, _):
    btn_y = self._rect.y + self._rect.height - 160 - 45
    btn_width = (self._rect.width - 45 * 3) / 2
    self._back_btn.render(rl.Rectangle(self._rect.x + 45, btn_y, btn_width, 160))
    self._uninstall_btn.render(rl.Rectangle(self._rect.x + 45 * 2 + btn_width, btn_y, btn_width, 160))

    # text rect in middle of top and button
    text_height = btn_y - (200 + 45)
    text_rect = rl.Rectangle(self._rect.x + 165, self._rect.y + (btn_y - text_height) / 2 + 10, self._rect.width - (165 * 2), text_height)
    if DEBUG:
      rl.draw_rectangle_lines_ex(text_rect, 3, rl.RED)
    self._text.render(text_rect)


class OnboardingWindow(Widget):
  def __init__(self):
    super().__init__()
    self._accepted_terms: bool = ui_state.params.get("HasAcceptedTerms") == terms_version
    self._state = OnboardingState.TERMS

    self._terms = TermsPage(on_accept=self._on_terms_accepted, on_decline=self._on_terms_declined)
    self._decline_page = DeclinePage(back_callback=self._on_decline_back)

  @property
  def completed(self) -> bool:
    return self._accepted_terms

  def _on_terms_declined(self):
    self._state = OnboardingState.DECLINE

  def _on_decline_back(self):
    self._state = OnboardingState.TERMS

  def _on_terms_accepted(self):
    ui_state.params.put("HasAcceptedTerms", terms_version)
    self._accepted_terms = True
    gui_app.set_modal_overlay(None)

  def _render(self, _):
    if self._state == OnboardingState.TERMS:
      self._terms.render(self._rect)
    elif self._state == OnboardingState.DECLINE:
      self._decline_page.render(self._rect)
    return -1
