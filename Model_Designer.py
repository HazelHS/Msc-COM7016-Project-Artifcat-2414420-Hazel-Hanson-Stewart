"""
Model_Designer.py
-----------------
Main entry point for the Model Designer application.
All GUI logic lives in Interface_Modules/.
"""

import tkinter as tk
from Interface_Modules.main_window import MainWindow


def main() -> None:
    root = tk.Tk()
    app = MainWindow(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
