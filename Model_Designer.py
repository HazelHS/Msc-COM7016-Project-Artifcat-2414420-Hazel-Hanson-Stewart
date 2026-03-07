# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The Model_Designer.py script is the main entry point for the Model Designer application.
All the GUI logic lives in Interface_Modules/. and this script should be run to start the application.
"""

import tkinter as tk
from Interface_Modules.main_window import MainWindow

def main() -> None: # (Anthropic, 2026)
    """Initialise the Tk root window and start the Model Designer application.

    Creates the root Tk window, passes it to MainWindow which builds
    all widgets, then enters the Tk event loop.  The loop blocks until
    the window is closed.
    """
    root = tk.Tk()
    app = MainWindow(root)  # noqa: F841  – MainWindow registers itself on root
    root.mainloop()

if __name__ == "__main__":
    main()
