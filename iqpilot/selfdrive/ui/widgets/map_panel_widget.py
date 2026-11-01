from iqpilot.selfdrive.ui.widgets.interactive_map import InteractiveNavMap


class MapPanelWidget(InteractiveNavMap):
  """Home location map using the same offline tiles and GPS puck as Navigation."""

  def __init__(self):
    super().__init__(interactive=False)
