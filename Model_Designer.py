"""
Model_Designer.py
-----------------
Main entry point for the Model Designer application.
All GUI logic lives in Interface_Modules/.
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
