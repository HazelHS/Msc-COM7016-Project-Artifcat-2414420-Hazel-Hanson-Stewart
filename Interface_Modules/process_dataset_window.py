# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
process_dataset_window.py is a Toplevel window for the Dataset Processing Method pipeline stage. 
It allows users to select a dataset CSV and choose which processing scripts to run on it. 
The window manages subprocess execution of the selected scripts, streaming their output to an 
embedded console widget in real time. Users can stop the active script and clear the queue of pending scripts. 
Selections are persisted back to the main stage dict when the window is closed.
"""

import collections
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext

from .constants import DATASET_OUTPUT_DIR, ROOT_DIR
from .utils import discover_scripts, discover_csvs

class ProcessDatasetWindow: # (Anthropic, 2026)
    """Toplevel window for the Dataset Processing Method pipeline stage."""

    CONSOLE_BG = "#1e1e1e"
    CONSOLE_FG = "#d4d4d4"

    def __init__(self, parent: tk.Widget, stage: dict) -> None: # (Anthropic, 2026)
        """Open the dataset-processing runner window.

        Args:
            parent: Parent widget that owns this Toplevel.
            stage: Mutable stage dict from MainWindow._stages. Script
                selections and the chosen CSV are written back into this
                dict when the window is closed.
        """
        self._stage = stage
        self._dir   = stage["dir"]
        self._process: subprocess.Popen | None = None
        self._output_queue: queue.Queue[str | None] = queue.Queue()
        self._run_queue: collections.deque[str] = collections.deque()
        self._check_vars: dict[str, tk.BooleanVar] = {}
        self._csv_var = tk.StringVar(value=stage.get("dataset_csv", ""))

        self._win = tk.Toplevel(parent)
        self._win.title("Process Dataset")
        self._win.minsize(660, 580)
        self._win.resizable(True, True)
        self._win.grab_set()
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()
        self._poll_output()

    # Build 
    def _build(self) -> None: # (Anthropic, 2026)
        """Assemble all child widgets: CSV picker, script checklist, action buttons, and console."""
        outer = ttk.Frame(self._win, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)  # console row expands

        # Dataset CSV picker
        csv_frame = ttk.LabelFrame(
            outer,
            text="Dataset CSV (passed to scripts via --dataset)",
            padding=(8, 4),
        )
        csv_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        csv_frame.columnconfigure(1, weight=1)

        ttk.Label(csv_frame, text="Select dataset:").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        self._csv_combo = ttk.Combobox(
            csv_frame, textvariable=self._csv_var, state="readonly", width=40
        )
        self._csv_combo.grid(row=0, column=1, sticky="ew")
        ttk.Button(
            csv_frame, text="\u21ba", width=3, command=self._refresh_csvs
        ).grid(row=0, column=2, padx=(4, 0))
        self._refresh_csvs()

        # Scrollable checklist
        list_frame = ttk.LabelFrame(
            outer, text="Available Processing Scripts", padding=(6, 4)
        )
        list_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        canvas = tk.Canvas(list_frame, highlightthickness=0, height=160)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._inner = ttk.Frame(canvas)
        self._canvas_win_id = canvas.create_window(
            (0, 0), window=self._inner, anchor="nw"
        )
        self._inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        self._canvas = canvas
        self._populate_checks()

        # Buttons row
        btn_row = ttk.Frame(outer)
        btn_row.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        ttk.Button(btn_row, text="Select All",   command=self._select_all,      width=12).pack(side="left")
        ttk.Button(btn_row, text="Deselect All", command=self._deselect_all,    width=12).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="\u21ba Refresh", command=self._populate_checks, width=10).pack(side="left", padx=(4, 0))

        self._stop_btn = ttk.Button(
            btn_row, text="\u25a0  Stop", width=9, command=self._on_stop, state="disabled"
        )
        self._stop_btn.pack(side="right")

        self._run_btn = ttk.Button(
            btn_row, text="\u25b6 Run", width=8, command=self._on_run
        )
        self._run_btn.pack(side="right", padx=(0, 4))

        # Console output
        console_frame = ttk.LabelFrame(outer, text="Console Output", padding=(6, 4))
        console_frame.grid(row=3, column=0, sticky="nsew")
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self._console = scrolledtext.ScrolledText(
            console_frame,
            state="disabled",
            bg=self.CONSOLE_BG,
            fg=self.CONSOLE_FG,
            insertbackground=self.CONSOLE_FG,
            font=("Consolas", 10),
            wrap="word",
            relief="flat",
        )
        self._console.grid(row=0, column=0, sticky="nsew")
        self._console.tag_config("info",  foreground="#9cdcfe")
        self._console.tag_config("error", foreground="#f48771")
        self._console.tag_config("dim",   foreground="#6a9955")
        self._console.tag_config("head",  foreground="#dcdcaa")

    # Script checklist
    def _populate_checks(self) -> None: # (Anthropic, 2026)
        """Rebuild the script checklist from the processing stage's directory.

        Clears any existing checkboxes, scans the directory for available
        Python scripts, and renders a new checkbox for each one. Previously
        selected scripts are re-checked automatically.
        """
        for widget in self._inner.winfo_children():
            widget.destroy()
        self._check_vars.clear()

        scripts = discover_scripts(self._dir)
        if not scripts:
            ttk.Label(
                self._inner,
                text="(no .py scripts found in this folder)",
                foreground="grey",
            ).pack(anchor="w", pady=4, padx=4)
            return

        currently_selected: set[str] = self._stage.get("selected", set())
        for script in scripts:
            var = tk.BooleanVar(value=(script in currently_selected))
            self._check_vars[script] = var
            ttk.Checkbutton(
                self._inner, text=script, variable=var,
                onvalue=True, offvalue=False,
            ).pack(anchor="w", padx=6, pady=1)

        self._canvas.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _refresh_csvs(self) -> None: # (Anthropic, 2026)
        """Repopulate the CSV combobox by scanning the dataset_output folder."""
        csvs = discover_csvs(DATASET_OUTPUT_DIR)
        self._csv_combo["values"] = csvs
        if self._csv_var.get() not in csvs:
            self._csv_var.set(csvs[0] if csvs else "")

    def _select_all(self) -> None: # (Anthropic, 2026)
        """Tick every script checkbox in the checklist."""
        for var in self._check_vars.values():
            var.set(True)

    def _deselect_all(self) -> None: # (Anthropic, 2026)
        """Untick every script checkbox in the checklist."""
        for var in self._check_vars.values():
            var.set(False)

    # Run logic 
    def _on_run(self) -> None: # (Anthropic, 2026)
        """Validate selections and enqueue all checked scripts for sequential execution.

        Logs an error and aborts if no scripts are checked or no dataset CSV
        is selected. Otherwise builds the run queue and launches the first script.
        The chosen CSV path is forwarded to each script as ``--dataset``.
        """
        selected = sorted(name for name, var in self._check_vars.items() if var.get())
        if not selected:
            self._log("No scripts selected.\n", tag="error")
            return
        csv_name = self._csv_var.get()
        if not csv_name:
            self._log(
                "No dataset CSV selected. Please choose a CSV from the dropdown.\n",
                tag="error",
            )
            return
        self._selected_csv_path = os.path.join(DATASET_OUTPUT_DIR, csv_name)
        self._run_queue.clear()
        for name in selected:
            self._run_queue.append(os.path.join(self._dir, name))
        self._log(
            f"\n=== Running {len(selected)} processing script(s) "
            f"(dataset: {csv_name}) ===\n",
            tag="head",
        )
        self._start_next()

    def _start_next(self) -> None: # (Anthropic, 2026)
        """Pop and launch the next script from the run queue.

        Transitions to idle and logs a completion message when the queue
        is empty.
        """
        if not self._run_queue:
            self._set_running(False)
            self._log("\n=== All processing scripts finished ===\n", tag="head")
            return
        script_path = self._run_queue.popleft()
        self._log(f"\n--- Running: {script_path} ---\n", tag="dim")
        self._set_running(True)
        threading.Thread(
            target=self._run_script, args=(script_path,), daemon=True
        ).start()

    def _run_script(self, script_path: str) -> None: # (Anthropic, 2026)
        """Execute a Python script as a subprocess and stream its output.

        Launches script_path with the current interpreter, combining stdout
        and stderr into a single stream forwarded line-by-line to the output
        queue. Passes the selected dataset CSV path via ``--dataset``. Pushes
        a None sentinel on completion so the poll loop can advance the queue.

        Args:
            script_path: Absolute path to the Python script to execute.
        """
        dataset_path = getattr(self, "_selected_csv_path", "")
        extra_args = ["--dataset", dataset_path] if dataset_path else []
        try:
            self._process = subprocess.Popen(
                [sys.executable, script_path, *extra_args],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=ROOT_DIR,
            )
            for line in self._process.stdout:
                self._output_queue.put(line)
            self._process.wait()
            self._output_queue.put(
                f"\n--- Process finished with exit code {self._process.returncode} ---\n"
            )
        except Exception as exc:
            self._output_queue.put(f"[ERROR] {exc}\n")
        finally:
            self._process = None
            self._output_queue.put(None)  # sentinel

    def _on_stop(self) -> None: # (Anthropic, 2026)
        """Terminate the active subprocess and discard all queued scripts."""
        self._run_queue.clear()
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._log("\n[Process terminated by user - queue cleared]\n", tag="error")

    def _set_running(self, is_running: bool) -> None: # (Anthropic, 2026)
        """Toggle the Run and Stop buttons between running and idle states.

        Args:
            is_running: If True, disable the Run button and enable Stop.
                If False, re-enable Run and disable Stop.
        """
        self._run_btn.config(state="disabled" if is_running else "normal")
        self._stop_btn.config(state="normal"   if is_running else "disabled")

    # Output polling 
    def _poll_output(self) -> None: # (Anthropic, 2026)
        """Drain pending lines from the output queue and write them to the console.

        Reschedules itself every 50 ms via tk.after for the lifetime of the
        window. A None sentinel emitted by a finished script triggers
        _start_next to launch the next queued script or return to idle.
        """
        try:
            while True:
                item = self._output_queue.get_nowait()
                if item is None:
                    self._start_next()
                else:
                    self._log(item)
        except queue.Empty:
            pass
        finally:
            if self._win.winfo_exists():
                self._win.after(50, self._poll_output)

    # Console helpers
    def _log(self, text: str, tag: str = "") -> None:
        """Append text to the console widget, optionally applying a colour tag.

        Args:
            text: The string to append to the console.
            tag: Named colour tag to apply (``'info'``, ``'error'``, ``'dim'``,
                or ``'head'``). Pass an empty string or omit for the default
                console colour.
        """
        self._console.config(state="normal")
        if tag:
            self._console.insert("end", text, tag)
        else:
            self._console.insert("end", text)
        self._console.see("end")
        self._console.config(state="disabled")

    def _on_close(self) -> None: # (Anthropic, 2026)
        """Terminate any running subprocess and persist selections back to the stage dict.

        Writes the current checkbox state and chosen CSV filename into the
        stage dict so MainWindow reflects the updated configuration. Also
        updates the stage's status label with the new selection count.
        """
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._stage["selected"] = {
            name for name, var in self._check_vars.items() if var.get()
        }
        self._stage["dataset_csv"] = self._csv_var.get()
        n = len(self._stage["selected"])
        self._stage["status_var"].set(f"{n} script{'s' if n != 1 else ''} selected")
        self._win.destroy()
