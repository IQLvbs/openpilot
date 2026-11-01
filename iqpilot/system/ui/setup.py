#!/usr/bin/env python3
from iqpilot.system.ui.lib.application import gui_app


def main():
  if gui_app.big_ui():
    import iqpilot.system.ui.tici_setup as tici_setup
    tici_setup.main()
  else:
    import iqpilot.system.ui.mici_setup as mici_setup
    mici_setup.main()


if __name__ == "__main__":
  main()
