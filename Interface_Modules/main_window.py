"""
main_window.py
--------------
Main GUI window for the Model Designer application.

Pipeline stages are displayed top-to-bottom.  Stages that support
multi-script selection ("Dataset Collection Method",
"Dataset Processing Method", "Model Evaluation Method") show a
"Configure" button that opens a checklist window; the remaining
stages use a combobox for single-script selection.

A "Run" button per stage executes that stage's selected scripts sequentially.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import subprocess
import threading
import queue
import collections
import os
import sys
import pandas as pd


# ── Project root (two levels up from this file) ─────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Pipeline stage definitions ───────────────────────────────────────
# Each tuple: (display_label, path_relative_to_ROOT_DIR, multi_select,
#              diagram_script_relative_to_ROOT_DIR | None)
# multi_select=True  → Configure button with checkboxes
# multi_select=False → single Combobox
# diagram_script     → if set, renders a "Create Diagram" button that runs that script
PIPELINE_STAGES: list[tuple[str, str, bool, str | None]] = [
    ("Dataset Collection Method",   os.path.join("Dataset_Modules", "Dataset_Collection"),         True,  None),
    ("Dataset Processing Method",   os.path.join("Dataset_Modules", "Dataset_Processing_Methods"), True,  None),
    ("AI Model Designs",            os.path.join("AI_Modules",      "Model_Designs"),              False,
     os.path.join("AI_Modules", "Model_Map_Diagram", "model_node_display.py")),
    ("AI Training Method",          os.path.join("AI_Modules",      "Training_Methods"),           False, None),
    ("Model Evaluation Method",     os.path.join("Evaluation_Modules", "Evaluation_Metrics"),      True,  None),
]


# Folder that holds the collected CSV datasets
DATASET_OUTPUT_DIR = os.path.join(ROOT_DIR, "Dataset_Modules", "dataset_output")


def discover_scripts(directory: str) -> list[str]:
    """Return a sorted list of .py filenames found in *directory*."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".py") and not f.startswith("__")
    )


def discover_csvs(directory: str) -> list[str]:
    """Return a sorted list of .csv filenames found in *directory*."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".csv")
    )


# ── Configure-window for multi-select stages ─────────────────────────

FREQ_OPTIONS: list[str] = [
    # ── Intraday ────────────────
    "1m", "2m", "5m", "15m", "30m", "90m", "1h",
    # ── Daily / coarser ─────────
    "1d", "5d", "1wk", "1mo", "3mo",
]

# Labels for which the Configure window shows the date/frequency panel.
_DATE_CONFIG_STAGES: set[str] = {"Dataset Collection Method"}

# Labels for which the Configure window shows the CSV dataset picker.
_DATASET_SELECT_STAGES: set[str] = {"Dataset Processing Method"}


class ConfigureWindow:
    """
    Toplevel window that lists all .py scripts in a directory with
    a checkbox next to each one.  The caller reads back `stage["selected"]`
    (a set of filenames) after the window is closed.

    For Dataset Collection stages an extra panel is shown at the bottom
    with start/end date inputs and a frequency drop-down.
    """

    def __init__(self, parent: tk.Widget, stage: dict) -> None:
        self._stage = stage
        self._show_date_config: bool    = stage.get("label_text", "") in _DATE_CONFIG_STAGES
        self._show_dataset_select: bool = stage.get("label_text", "") in _DATASET_SELECT_STAGES

        # Seed defaults into the stage dict on first open
        if self._show_date_config:
            stage.setdefault("start_date", "2015-01-01")
            stage.setdefault("end_date",   "2025-02-01")
            stage.setdefault("freq",       "1d")
        if self._show_dataset_select:
            stage.setdefault("dataset_csv", "")

        self._win = tk.Toplevel(parent)
        self._win.title(f"Configure: {stage['label_text']}")
        self._win.resizable(False, True)
        self._win.minsize(500, 240)
        self._win.grab_set()           # modal

        self._check_vars: dict[str, tk.BooleanVar] = {}

        # Date/freq tk variables (only created when _show_date_config is True)
        self._start_var: tk.StringVar | None = None
        self._end_var:   tk.StringVar | None = None
        self._freq_var:  tk.StringVar | None = None

        # CSV picker variable (only created when _show_dataset_select is True)
        self._csv_var: tk.StringVar | None = None

        self._build()

    # ── Build ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = ttk.Frame(self._win, padding=10)
        outer.pack(fill="both", expand=True)

        # Scrollable checklist
        list_frame = ttk.LabelFrame(outer, text="Available Scripts", padding=(6, 4))
        list_frame.pack(fill="both", expand=True, pady=(0, 8))

        canvas = tk.Canvas(list_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._inner = ttk.Frame(canvas)
        self._canvas_window = canvas.create_window((0, 0), window=self._inner, anchor="nw")

        self._inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        ))

        self._canvas = canvas
        self._populate_checks()

        # ── Date / frequency panel (Dataset Collection only) ───────────
        if self._show_date_config:
            self._build_date_panel(outer)
        # ── Dataset CSV picker (Dataset Processing only) ──────────────────
        if self._show_dataset_select:
            self._build_csv_panel(outer)
        # Buttons row
        btn_row = ttk.Frame(outer)
        btn_row.pack(fill="x", pady=(8, 0))

        ttk.Button(btn_row, text="Select All",   command=self._select_all,   width=12).pack(side="left")
        ttk.Button(btn_row, text="Deselect All", command=self._deselect_all, width=12).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="↺ Refresh",    command=self._refresh,      width=10).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="OK",           command=self._ok,           width=8).pack(side="right")

    def _build_csv_panel(self, parent: ttk.Frame) -> None:
        """Add a CSV dataset picker for Dataset Processing stages."""
        csv_frame = ttk.LabelFrame(parent, text="Dataset CSV (passed to scripts via --dataset)", padding=(8, 6))
        csv_frame.pack(fill="x", pady=(0, 6))
        csv_frame.columnconfigure(1, weight=1)

        ttk.Label(csv_frame, text="Select dataset:").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self._csv_var = tk.StringVar(value=self._stage.get("dataset_csv", ""))
        self._csv_combo = ttk.Combobox(
            csv_frame, textvariable=self._csv_var, state="readonly", width=40
        )
        self._csv_combo.grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Button(
            csv_frame, text="↺", width=3, command=self._refresh_csvs
        ).grid(row=0, column=2, padx=(4, 0), pady=3)
        self._refresh_csvs()

    def _refresh_csvs(self) -> None:
        """Repopulate the CSV combobox from the dataset_output folder."""
        csvs = discover_csvs(DATASET_OUTPUT_DIR)
        if hasattr(self, "_csv_combo") and self._csv_combo is not None:
            self._csv_combo["values"] = csvs
            current = self._csv_var.get() if self._csv_var else ""
            if current not in csvs:
                self._csv_var.set(csvs[0] if csvs else "")

    def _build_date_panel(self, parent: ttk.Frame) -> None:
        """Add start/end date entries and a frequency combobox."""
        date_frame = ttk.LabelFrame(parent, text="Date Range & Frequency", padding=(8, 6))
        date_frame.pack(fill="x", pady=(0, 6))

        date_frame.columnconfigure(1, weight=1)
        date_frame.columnconfigure(3, weight=1)

        # ── Start date ─────────────────────────────────────────────────
        ttk.Label(date_frame, text="Start Date (YYYY-MM-DD):").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self._start_var = tk.StringVar(value=self._stage.get("start_date", "2015-01-01"))
        start_entry = ttk.Entry(date_frame, textvariable=self._start_var, width=14)
        start_entry.grid(row=0, column=1, sticky="ew", pady=3)

        # ── End date ───────────────────────────────────────────────────
        ttk.Label(date_frame, text="End Date (YYYY-MM-DD):").grid(
            row=0, column=2, sticky="w", padx=(12, 6), pady=3
        )
        self._end_var = tk.StringVar(value=self._stage.get("end_date", "2025-02-01"))
        end_entry = ttk.Entry(date_frame, textvariable=self._end_var, width=14)
        end_entry.grid(row=0, column=3, sticky="ew", pady=3)

        # ── Frequency ──────────────────────────────────────────────────
        ttk.Label(date_frame, text="Data Frequency:").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self._freq_var = tk.StringVar(value=self._stage.get("freq", "1d"))
        freq_combo = ttk.Combobox(
            date_frame,
            textvariable=self._freq_var,
            values=FREQ_OPTIONS,
            state="readonly",
            width=14,
        )
        freq_combo.grid(row=1, column=1, sticky="w", pady=3)

    def _populate_checks(self) -> None:
        """Destroy existing checkboxes and rebuild from directory listing."""
        for widget in self._inner.winfo_children():
            widget.destroy()
        self._check_vars.clear()

        scripts = discover_scripts(self._stage["dir"])
        if not scripts:
            ttk.Label(self._inner, text="(no .py scripts found in this folder)",
                      foreground="grey").pack(anchor="w", pady=4, padx=4)
            return

        currently_selected: set[str] = self._stage.get("selected", set())
        for script in scripts:
            var = tk.BooleanVar(value=(script in currently_selected))
            self._check_vars[script] = var
            ttk.Checkbutton(
                self._inner,
                text=script,
                variable=var,
                onvalue=True,
                offvalue=False,
            ).pack(anchor="w", padx=6, pady=1)

    # ── Actions ────────────────────────────────────────────────────────

    def _select_all(self) -> None:
        for var in self._check_vars.values():
            var.set(True)

    def _deselect_all(self) -> None:
        for var in self._check_vars.values():
            var.set(False)

    def _refresh(self) -> None:
        self._populate_checks()

    def _ok(self) -> None:
        """Commit checked scripts and (optionally) date/freq/csv back to the stage."""
        self._stage["selected"] = {
            name for name, var in self._check_vars.items() if var.get()
        }
        if self._show_date_config:
            self._stage["start_date"] = self._start_var.get().strip()
            self._stage["end_date"]   = self._end_var.get().strip()
            self._stage["freq"]       = self._freq_var.get()
        if self._show_dataset_select and self._csv_var is not None:
            self._stage["dataset_csv"] = self._csv_var.get()
        self._stage["status_var"].set(self._status_text())
        self._win.destroy()

    def _status_text(self) -> str:
        n = len(self._stage["selected"])
        return f"{n} script{'s' if n != 1 else ''} selected"


# ── Analysis window for Dataset_Analysis_Methods ─────────────────────

class AnalysisWindow:
    """
    Toplevel window that lists all .py scripts in the Dataset_Analysis_Methods
    folder with checkboxes, a Run button, a Stop button, and an embedded
    console output area.  Runs selected scripts sequentially as subprocesses.
    """

    CONSOLE_BG = "#1e1e1e"
    CONSOLE_FG = "#d4d4d4"

    def __init__(self, parent: tk.Widget, analysis_dir: str) -> None:
        self._dir = analysis_dir
        self._process: subprocess.Popen | None = None
        self._output_queue: queue.Queue[str | None] = queue.Queue()
        self._run_queue: collections.deque[str] = collections.deque()
        self._check_vars: dict[str, tk.BooleanVar] = {}
        self._csv_var = tk.StringVar()

        self._win = tk.Toplevel(parent)
        self._win.title("Analyse Data Collection")
        self._win.minsize(660, 580)
        self._win.grab_set()  # modal
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()
        self._poll_output()

    # ── Build ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = ttk.Frame(self._win, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)  # console row expands

        # ── Dataset CSV picker ───────────────────────────────────────
        csv_frame = ttk.LabelFrame(outer, text="Dataset CSV (passed to scripts via --dataset)", padding=(8, 4))
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
            csv_frame, text="↺", width=3, command=self._refresh_csvs
        ).grid(row=0, column=2, padx=(4, 0))
        self._refresh_csvs()

        # ── Scrollable checklist ───────────────────────────────────────
        list_frame = ttk.LabelFrame(outer, text="Available Analysis Scripts", padding=(6, 4))
        list_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        canvas = tk.Canvas(list_frame, highlightthickness=0, height=160)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._inner = ttk.Frame(canvas)
        self._canvas_win_id = canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        ))
        self._canvas = canvas

        self._populate_checks()

        # ── Buttons row ─────────────────────────────────────────────
        btn_row = ttk.Frame(outer)
        btn_row.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        ttk.Button(btn_row, text="Select All",   command=self._select_all,   width=12).pack(side="left")
        ttk.Button(btn_row, text="Deselect All", command=self._deselect_all, width=12).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="↺ Refresh",    command=self._populate_checks, width=10).pack(side="left", padx=(4, 0))

        self._stop_btn = ttk.Button(btn_row, text="■  Stop", width=9,
                                    command=self._on_stop, state="disabled")
        self._stop_btn.pack(side="right")

        self._run_btn = ttk.Button(btn_row, text="▶ Run", width=8, command=self._on_run)
        self._run_btn.pack(side="right", padx=(0, 4))

        # ── Console output ───────────────────────────────────────────
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

    # ── Script checklist ───────────────────────────────────────────────

    def _populate_checks(self) -> None:
        """Rebuild the checkbox list from the analysis directory."""
        for widget in self._inner.winfo_children():
            widget.destroy()
        self._check_vars.clear()

        scripts = discover_scripts(self._dir)
        if not scripts:
            ttk.Label(
                self._inner,
                text="(no .py scripts found in Dataset_Analysis_Methods)",
                foreground="grey",
            ).pack(anchor="w", pady=4, padx=4)
            return

        for script in scripts:
            var = tk.BooleanVar(value=False)
            self._check_vars[script] = var
            ttk.Checkbutton(
                self._inner,
                text=script,
                variable=var,
                onvalue=True,
                offvalue=False,
            ).pack(anchor="w", padx=6, pady=1)

        self._canvas.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _refresh_csvs(self) -> None:
        """Repopulate the CSV combobox from the dataset_output folder."""
        csvs = discover_csvs(DATASET_OUTPUT_DIR)
        self._csv_combo["values"] = csvs
        current = self._csv_var.get()
        if current not in csvs:
            self._csv_var.set(csvs[0] if csvs else "")

    def _select_all(self) -> None:
        for var in self._check_vars.values():
            var.set(True)

    def _deselect_all(self) -> None:
        for var in self._check_vars.values():
            var.set(False)

    # ── Run logic ──────────────────────────────────────────────────────

    def _on_run(self) -> None:
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
            f"\n=== Running {len(selected)} analysis script(s) "
            f"(dataset: {csv_name}) ===\n",
            tag="head",
        )
        self._start_next()

    def _start_next(self) -> None:
        if not self._run_queue:
            self._set_running(False)
            self._log("\n=== All analysis scripts finished ===\n", tag="head")
            return
        script_path = self._run_queue.popleft()
        self._log(f"\n--- Running: {script_path} ---\n", tag="dim")
        self._set_running(True)
        threading.Thread(
            target=self._run_script, args=(script_path,), daemon=True
        ).start()

    def _run_script(self, script_path: str) -> None:
        """Execute *script_path* as a subprocess; stream stdout to the queue."""
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
            exit_code = self._process.returncode
            self._output_queue.put(
                f"\n--- Process finished with exit code {exit_code} ---\n"
            )
        except Exception as exc:
            self._output_queue.put(f"[ERROR] {exc}\n")
        finally:
            self._process = None
            self._output_queue.put(None)  # sentinel

    def _on_stop(self) -> None:
        self._run_queue.clear()
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._log("\n[Process terminated by user — queue cleared]\n", tag="error")

    def _set_running(self, is_running: bool) -> None:
        self._run_btn.config(state="disabled" if is_running else "normal")
        self._stop_btn.config(state="normal" if is_running else "disabled")

    # ── Output polling ─────────────────────────────────────────────────

    def _poll_output(self) -> None:
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

    # ── Console helpers ────────────────────────────────────────────────

    def _log(self, text: str, tag: str = "") -> None:
        self._console.config(state="normal")
        if tag:
            self._console.insert("end", text, tag)
        else:
            self._console.insert("end", text)
        self._console.see("end")
        self._console.config(state="disabled")

    def _on_close(self) -> None:
        """Terminate any running subprocess before closing."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._win.destroy()


# ── Process Dataset window ────────────────────────────────────────────────

class ProcessDatasetWindow:
    """
    Toplevel window for the 'Dataset Processing Method' pipeline stage.
    Combines a CSV dataset picker, a scrollable script checklist, Run/Stop
    buttons and an embedded console in a single self-contained window.
    Selections and the chosen CSV are persisted back into the stage dict
    when the window is closed.
    """

    CONSOLE_BG = "#1e1e1e"
    CONSOLE_FG = "#d4d4d4"

    def __init__(self, parent: tk.Widget, stage: dict) -> None:
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

    # ── Build ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = ttk.Frame(self._win, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)  # console row expands

        # ── Dataset CSV picker ───────────────────────────────────────
        csv_frame = ttk.LabelFrame(
            outer, text="Dataset CSV (passed to scripts via --dataset)", padding=(8, 4)
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

        # ── Scrollable checklist ──────────────────────────────────────
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
        self._canvas_win_id = canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        ))
        self._canvas = canvas
        self._populate_checks()

        # ── Buttons row ──────────────────────────────────────────────
        btn_row = ttk.Frame(outer)
        btn_row.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        ttk.Button(btn_row, text="Select All",   command=self._select_all,      width=12).pack(side="left")
        ttk.Button(btn_row, text="Deselect All", command=self._deselect_all,    width=12).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="\u21ba Refresh",  command=self._populate_checks, width=10).pack(side="left", padx=(4, 0))

        self._stop_btn = ttk.Button(
            btn_row, text="\u25a0  Stop", width=9, command=self._on_stop, state="disabled"
        )
        self._stop_btn.pack(side="right")

        self._run_btn = ttk.Button(
            btn_row, text="\u25b6 Run", width=8, command=self._on_run
        )
        self._run_btn.pack(side="right", padx=(0, 4))

        # ── Console output ───────────────────────────────────────────
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

    # ── Script checklist ───────────────────────────────────────────────

    def _populate_checks(self) -> None:
        """Rebuild the checkbox list from the processing directory."""
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
                self._inner, text=script, variable=var, onvalue=True, offvalue=False
            ).pack(anchor="w", padx=6, pady=1)

        self._canvas.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _refresh_csvs(self) -> None:
        """Repopulate the CSV combobox from the dataset_output folder."""
        csvs = discover_csvs(DATASET_OUTPUT_DIR)
        self._csv_combo["values"] = csvs
        current = self._csv_var.get()
        if current not in csvs:
            self._csv_var.set(csvs[0] if csvs else "")

    def _select_all(self) -> None:
        for var in self._check_vars.values():
            var.set(True)

    def _deselect_all(self) -> None:
        for var in self._check_vars.values():
            var.set(False)

    # ── Run logic ──────────────────────────────────────────────────────

    def _on_run(self) -> None:
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

    def _start_next(self) -> None:
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

    def _run_script(self, script_path: str) -> None:
        """Execute *script_path* as a subprocess; stream stdout to the queue."""
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
            exit_code = self._process.returncode
            self._output_queue.put(
                f"\n--- Process finished with exit code {exit_code} ---\n"
            )
        except Exception as exc:
            self._output_queue.put(f"[ERROR] {exc}\n")
        finally:
            self._process = None
            self._output_queue.put(None)  # sentinel

    def _on_stop(self) -> None:
        self._run_queue.clear()
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._log("\n[Process terminated by user - queue cleared]\n", tag="error")

    def _set_running(self, is_running: bool) -> None:
        self._run_btn.config(state="disabled" if is_running else "normal")
        self._stop_btn.config(state="normal" if is_running else "disabled")

    # ── Output polling ───────────────────────────────────────────────

    def _poll_output(self) -> None:
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

    # ── Console helpers ──────────────────────────────────────────────

    def _log(self, text: str, tag: str = "") -> None:
        self._console.config(state="normal")
        if tag:
            self._console.insert("end", text, tag)
        else:
            self._console.insert("end", text)
        self._console.see("end")
        self._console.config(state="disabled")

    def _on_close(self) -> None:
        """Terminate any running subprocess; persist selections back to the stage dict."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
        # Save current selections and CSV choice so the status label stays accurate
        self._stage["selected"] = {
            name for name, var in self._check_vars.items() if var.get()
        }
        self._stage["dataset_csv"] = self._csv_var.get()
        n = len(self._stage["selected"])
        self._stage["status_var"].set(f"{n} script{'s' if n != 1 else ''} selected")
        self._win.destroy()


# ── Feature Selection window ──────────────────────────────────────────────

class FeatureSelectionWindow:
    """
    Toplevel window for running feature-selection scripts located in
    Dataset_Processing_Methods/Dataset_Feature_Selection/.
    Provides a CSV dataset picker, scrollable script checklist,
    Run/Stop buttons, and an embedded console – matching the style of
    ProcessDatasetWindow and AnalysisWindow.
    """

    CONSOLE_BG = "#1e1e1e"
    CONSOLE_FG = "#d4d4d4"

    def __init__(self, parent: tk.Widget, feature_dir: str) -> None:
        self._dir = feature_dir
        self._process: subprocess.Popen | None = None
        self._output_queue: queue.Queue[str | None] = queue.Queue()
        self._run_queue: collections.deque[str] = collections.deque()
        self._check_vars: dict[str, tk.BooleanVar] = {}
        self._csv_var = tk.StringVar()
        self._target_var = tk.StringVar()

        self._win = tk.Toplevel(parent)
        self._win.title("Feature Selection")
        self._win.minsize(660, 580)
        self._win.resizable(True, True)
        self._win.grab_set()
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()
        self._poll_output()

    # ── Build ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = ttk.Frame(self._win, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)  # console row expands

        # ── Dataset CSV picker ───────────────────────────────────────
        csv_frame = ttk.LabelFrame(
            outer, text="Dataset CSV (passed to scripts via --dataset)", padding=(8, 4)
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

        # ── Target column picker ─────────────────────────────────────
        ttk.Label(csv_frame, text="Target column:").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=(4, 0)
        )
        self._target_combo = ttk.Combobox(
            csv_frame, textvariable=self._target_var, state="readonly", width=40
        )
        self._target_combo.grid(row=1, column=1, sticky="ew", pady=(4, 0))
        ttk.Button(
            csv_frame, text="\u21ba", width=3, command=self._refresh_target_columns
        ).grid(row=1, column=2, padx=(4, 0), pady=(4, 0))

        self._csv_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_target_columns())
        self._refresh_csvs()

        # ── Scrollable checklist ──────────────────────────────────────
        list_frame = ttk.LabelFrame(
            outer, text="Available Feature Selection Scripts", padding=(6, 4)
        )
        list_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        canvas = tk.Canvas(list_frame, highlightthickness=0, height=160)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._inner = ttk.Frame(canvas)
        self._canvas_win_id = canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        ))
        self._canvas = canvas
        self._populate_checks()

        # ── Buttons row ──────────────────────────────────────────────
        btn_row = ttk.Frame(outer)
        btn_row.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        ttk.Button(btn_row, text="Select All",   command=self._select_all,       width=12).pack(side="left")
        ttk.Button(btn_row, text="Deselect All", command=self._deselect_all,     width=12).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="\u21ba Refresh", command=self._populate_checks, width=10).pack(side="left", padx=(4, 0))

        self._stop_btn = ttk.Button(
            btn_row, text="\u25a0  Stop", width=9, command=self._on_stop, state="disabled"
        )
        self._stop_btn.pack(side="right")

        self._run_btn = ttk.Button(
            btn_row, text="\u25b6 Run", width=8, command=self._on_run
        )
        self._run_btn.pack(side="right", padx=(0, 4))

        # ── Console output ───────────────────────────────────────────
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

    # ── Script checklist ───────────────────────────────────────────────

    def _populate_checks(self) -> None:
        """Rebuild the checkbox list from the feature-selection directory."""
        for widget in self._inner.winfo_children():
            widget.destroy()
        self._check_vars.clear()

        scripts = discover_scripts(self._dir)
        if not scripts:
            ttk.Label(
                self._inner,
                text="(no .py scripts found in Dataset_Feature_Selection)",
                foreground="grey",
            ).pack(anchor="w", pady=4, padx=4)
            return

        for script in scripts:
            var = tk.BooleanVar(value=False)
            self._check_vars[script] = var
            ttk.Checkbutton(
                self._inner, text=script, variable=var, onvalue=True, offvalue=False
            ).pack(anchor="w", padx=6, pady=1)

        self._canvas.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _refresh_csvs(self) -> None:
        """Repopulate the CSV combobox from the dataset_output folder."""
        csvs = discover_csvs(DATASET_OUTPUT_DIR)
        self._csv_combo["values"] = csvs
        current = self._csv_var.get()
        if current not in csvs:
            self._csv_var.set(csvs[0] if csvs else "")
        self._refresh_target_columns()

    def _refresh_target_columns(self) -> None:
        """Populate the target-column combobox from the currently selected CSV headers."""
        csv_name = self._csv_var.get()
        if not csv_name:
            self._target_combo["values"] = []
            self._target_var.set("")
            return
        csv_path = os.path.join(DATASET_OUTPUT_DIR, csv_name)
        try:
            cols = list(pd.read_csv(csv_path, nrows=0, index_col=0).columns)
        except Exception:
            cols = []
        self._target_combo["values"] = cols
        current = self._target_var.get()
        if current not in cols:
            self._target_var.set(cols[0] if cols else "")

    def _select_all(self) -> None:
        for var in self._check_vars.values():
            var.set(True)

    def _deselect_all(self) -> None:
        for var in self._check_vars.values():
            var.set(False)

    # ── Run logic ──────────────────────────────────────────────────────

    def _on_run(self) -> None:
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
            f"\n=== Running {len(selected)} feature-selection script(s) "
            f"(dataset: {csv_name}) ===\n",
            tag="head",
        )
        self._start_next()

    def _start_next(self) -> None:
        if not self._run_queue:
            self._set_running(False)
            self._log("\n=== All feature-selection scripts finished ===\n", tag="head")
            return
        script_path = self._run_queue.popleft()
        self._log(f"\n--- Running: {script_path} ---\n", tag="dim")
        self._set_running(True)
        threading.Thread(
            target=self._run_script, args=(script_path,), daemon=True
        ).start()

    def _run_script(self, script_path: str) -> None:
        """Execute *script_path* as a subprocess; stream stdout to the queue."""
        dataset_path = getattr(self, "_selected_csv_path", "")
        extra_args = ["--dataset", dataset_path] if dataset_path else []
        target_col = self._target_var.get()
        if target_col:
            extra_args += ["--target", target_col]
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
            exit_code = self._process.returncode
            self._output_queue.put(
                f"\n--- Process finished with exit code {exit_code} ---\n"
            )
        except Exception as exc:
            self._output_queue.put(f"[ERROR] {exc}\n")
        finally:
            self._process = None
            self._output_queue.put(None)  # sentinel

    def _on_stop(self) -> None:
        self._run_queue.clear()
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._log("\n[Process terminated by user - queue cleared]\n", tag="error")

    def _set_running(self, is_running: bool) -> None:
        self._run_btn.config(state="disabled" if is_running else "normal")
        self._stop_btn.config(state="normal" if is_running else "disabled")

    # ── Output polling ───────────────────────────────────────────────

    def _poll_output(self) -> None:
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

    # ── Console helpers ──────────────────────────────────────────────

    def _log(self, text: str, tag: str = "") -> None:
        self._console.config(state="normal")
        if tag:
            self._console.insert("end", text, tag)
        else:
            self._console.insert("end", text)
        self._console.see("end")
        self._console.config(state="disabled")

    def _on_close(self) -> None:
        """Terminate any running subprocess before closing."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._win.destroy()


# ── Training Configure window ─────────────────────────────────────────────

class TrainingConfigureWindow:
    """
    Toplevel configuration and execution window for the AI Training Method stage.

    Allows the user to:
      - Pick a training script from the Training_Methods directory
      - Pick an input dataset CSV and target column
      - Configure model-specific hyperparameters (xLSTM-TS or MEMD-TCN)
      - Launch / stop the training process
      - Monitor console output

    Parameter groups are shown/hidden based on which training script is selected:
      - Scripts whose filename contains 'memd' or 'tcn' → MEMD-TCN params
      - All other scripts                               → xLSTM-TS params

    All configured values are persisted back into the stage dict when the
    window is closed, so they survive between re-opens.
    """

    CONSOLE_BG = "#1e1e1e"
    CONSOLE_FG = "#d4d4d4"

    # ── Default hyperparameter values ─────────────────────────────────

    _XLSTM_DEFAULTS: dict[str, str] = {
        "sequence_length":          "60",
        "epochs":                   "200",
        "batch_size":               "16",
        "learning_rate":            "0.0001",
        "embedding_dim":            "64",
        "output_size":              "7",
        "early_stopping_patience":  "30",
        "train_split":              "0.70",
        "val_split":                "0.15",
    }

    _MEMD_DEFAULTS: dict[str, str] = {
        "sequence_length":  "30",
        "epochs_per_tcn":   "100",
        "batch_size":       "16",
        "learning_rate":    "0.0001",
        "kernel_size":      "2",
        "dilations":        "1,2,4",
        "dropout":          "0.2",
        "K":                "512",
        "max_imfs":         "12",
        "max_sift":         "50",
        "sd_threshold":     "0.2",
    }

    def __init__(self, parent: tk.Widget, stage: dict) -> None:
        self._stage  = stage
        self._dir    = stage["dir"]
        self._process: subprocess.Popen | None = None
        self._output_queue: queue.Queue[str | None] = queue.Queue()

        # The script to run is whatever is selected in the main-window combobox.
        # We read it from the stage's StringVar so changes in the main window
        # are always reflected when Run is clicked.
        _default_save = os.path.join(
            ROOT_DIR, "AI_Modules", "Training_Methods", "Trained_Model_Files"
        )
        self._csv_var     = tk.StringVar(value=stage.get("dataset_csv", ""))
        self._target_var  = tk.StringVar(value=stage.get("target_col",  ""))
        self._save_dir_var = tk.StringVar(value=stage.get("save_dir", _default_save))

        # xLSTM-TS tk vars
        self._xlstm_vars: dict[str, tk.StringVar] = {
            k: tk.StringVar(value=stage.get(f"xlstm_{k}", v))
            for k, v in self._XLSTM_DEFAULTS.items()
        }
        # MEMD-TCN tk vars
        self._memd_vars: dict[str, tk.StringVar] = {
            k: tk.StringVar(value=stage.get(f"memd_{k}", v))
            for k, v in self._MEMD_DEFAULTS.items()
        }

        self._win = tk.Toplevel(parent)
        self._win.title("Configure: AI Training Method")
        self._win.minsize(700, 620)
        self._win.resizable(True, True)
        self._win.grab_set()
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()
        self._poll_output()

    # ── Build ─────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = ttk.Frame(self._win, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        # Row 0: dataset frame (fixed)
        # Row 1: param frame   (fixed)
        # Row 2: buttons row   (fixed)
        # Row 3: console       (expands)
        outer.rowconfigure(3, weight=1)

        # ── Dataset ───────────────────────────────────────────────────
        dataset_frame = ttk.LabelFrame(outer, text="Dataset", padding=(8, 6))
        dataset_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        dataset_frame.columnconfigure(1, weight=1)

        # Training script label (read-only, driven by main-window combobox)
        ttk.Label(dataset_frame, text="Training Script:").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self._script_label = ttk.Label(
            dataset_frame,
            text=self._current_script_name(),
            foreground="#9cdcfe",
            anchor="w",
        )
        self._script_label.grid(row=0, column=1, sticky="ew", pady=3, columnspan=2)

        ttk.Label(dataset_frame, text="Dataset:").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self._csv_combo = ttk.Combobox(
            dataset_frame, textvariable=self._csv_var, state="readonly", width=44
        )
        self._csv_combo.grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Button(
            dataset_frame, text="↺", width=3, command=self._refresh_csvs
        ).grid(row=1, column=2, padx=(4, 0), pady=3)

        ttk.Label(dataset_frame, text="Target Column:").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self._target_combo = ttk.Combobox(
            dataset_frame, textvariable=self._target_var, state="readonly", width=44
        )
        self._target_combo.grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Button(
            dataset_frame, text="↺", width=3, command=self._refresh_target_columns
        ).grid(row=2, column=2, padx=(4, 0), pady=3)

        # Populate CSV list and derive target columns — _target_combo must exist first
        self._csv_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_target_columns())
        self._refresh_csvs()

        ttk.Label(dataset_frame, text="Save Model To:").grid(
            row=3, column=0, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Entry(
            dataset_frame, textvariable=self._save_dir_var, width=38
        ).grid(row=3, column=1, sticky="ew", pady=3)
        ttk.Button(
            dataset_frame, text="…", width=3, command=self._browse_save_dir
        ).grid(row=3, column=2, padx=(4, 0), pady=3)

        # ── xLSTM-TS parameter panel ──────────────────────────────────
        self._xlstm_frame = ttk.LabelFrame(
            outer,
            text="xLSTM-TS Hyperparameters  (Lopez et al. 2024)",
            padding=(8, 6),
        )
        self._xlstm_frame.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        self._build_xlstm_params(self._xlstm_frame)

        # ── MEMD-TCN parameter panel ──────────────────────────────────
        self._memd_frame = ttk.LabelFrame(
            outer,
            text="MEMD-TCN Hyperparameters  (Yao et al. 2023 / Rehman & Mandic 2010)",
            padding=(8, 6),
        )
        self._memd_frame.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        self._build_memd_params(self._memd_frame)

        # Show/hide the correct panel based on the currently selected script
        self._on_script_changed()

        # ── Buttons row ───────────────────────────────────────────────
        btn_row = ttk.Frame(outer)
        btn_row.grid(row=2, column=0, sticky="ew", pady=(4, 6))

        self._stop_btn = ttk.Button(
            btn_row, text="■  Stop", width=9, command=self._on_stop, state="disabled"
        )
        self._stop_btn.pack(side="right")

        self._run_btn = ttk.Button(
            btn_row, text="▶ Run", width=8, command=self._on_run
        )
        self._run_btn.pack(side="right", padx=(0, 4))

        # ── Console output ────────────────────────────────────────────
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

    # ── Parameter panels ──────────────────────────────────────────────

    def _build_xlstm_params(self, parent: ttk.Frame) -> None:
        """Grid of entry widgets for xLSTM-TS training hyperparameters."""
        v = self._xlstm_vars
        # (label, key, grid-row, grid-col)
        params = [
            ("Sequence Length:",            "sequence_length",         0, 0),
            ("Epochs:",                     "epochs",                  0, 2),
            ("Batch Size:",                 "batch_size",              1, 0),
            ("Learning Rate:",              "learning_rate",           1, 2),
            ("Embedding Dim:",              "embedding_dim",           2, 0),
            ("Forecast Horizon (days):",    "output_size",             2, 2),
            ("Early Stop Patience:",        "early_stopping_patience", 3, 0),
            ("Train Split (0–1):",          "train_split",             3, 2),
            ("Val Split (0–1):",            "val_split",               4, 0),
        ]
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)
        for lbl, key, row, col in params:
            ttk.Label(parent, text=lbl).grid(
                row=row, column=col, sticky="w",
                padx=(0 if col == 0 else 12, 6), pady=2
            )
            ttk.Entry(parent, textvariable=v[key], width=14).grid(
                row=row, column=col + 1, sticky="ew", pady=2, padx=(0, 4)
            )

    def _build_memd_params(self, parent: ttk.Frame) -> None:
        """Grid of entry widgets for MEMD-TCN training hyperparameters."""
        v = self._memd_vars
        params = [
            ("Sequence Length (lag):",      "sequence_length",  0, 0),
            ("Epochs per TCN:",             "epochs_per_tcn",   0, 2),
            ("Batch Size:",                 "batch_size",       1, 0),
            ("Learning Rate:",              "learning_rate",    1, 2),
            ("TCN Kernel Size:",            "kernel_size",      2, 0),
            ("Dilations (comma-sep):",      "dilations",        2, 2),
            ("Dropout:",                    "dropout",          3, 0),
            ("MEMD Directions (K):",        "K",                3, 2),
            ("Max IMFs:",                   "max_imfs",         4, 0),
            ("Max Sift Iterations:",        "max_sift",         4, 2),
            ("SD Threshold:",               "sd_threshold",     5, 0),
        ]
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)
        for lbl, key, row, col in params:
            ttk.Label(parent, text=lbl).grid(
                row=row, column=col, sticky="w",
                padx=(0 if col == 0 else 12, 6), pady=2
            )
            ttk.Entry(parent, textvariable=v[key], width=14).grid(
                row=row, column=col + 1, sticky="ew", pady=2, padx=(0, 4)
            )
        ttk.Label(
            parent,
            text="Select  Train_MEMD_TCN.py  in the 'AI Training Method' dropdown "
                 "to train this model.  All parameters above are passed directly as "
                 "CLI arguments to that script.",
            foreground="#6a9955",
            wraplength=600,
            justify="left",
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(8, 2))

    # ── Script / CSV helpers ──────────────────────────────────────────

    def _current_script_name(self) -> str:
        """Return the script name currently selected in the main-window combobox."""
        try:
            return self._stage["var"].get() or "(none selected)"
        except Exception:
            return "(none selected)"

    def _refresh_csvs(self) -> None:
        csvs = discover_csvs(DATASET_OUTPUT_DIR)
        self._csv_combo["values"] = csvs
        current = self._csv_var.get()
        if current not in csvs:
            self._csv_var.set(csvs[0] if csvs else "")
        self._refresh_target_columns()

    def _refresh_target_columns(self) -> None:
        """Populate the Target Column combobox from the headers of the selected CSV."""
        csv_name = self._csv_var.get()
        if not csv_name:
            self._target_combo["values"] = []
            self._target_var.set("")
            return
        csv_path = os.path.join(DATASET_OUTPUT_DIR, csv_name)
        try:
            cols = list(pd.read_csv(csv_path, nrows=0, index_col=0).columns)
        except Exception:
            cols = []
        self._target_combo["values"] = cols
        current = self._target_var.get()
        if current not in cols:
            self._target_var.set(cols[0] if cols else "")

    def _browse_save_dir(self) -> None:
        """Open a folder-picker and update the save directory entry."""
        chosen = filedialog.askdirectory(
            title="Select folder to save trained model",
            initialdir=self._save_dir_var.get() or ROOT_DIR,
        )
        if chosen:
            self._save_dir_var.set(chosen)

    def _is_memd_script(self) -> bool:
        name = self._current_script_name().lower()
        return "memd" in name or ("tcn" in name and "xlstm" not in name)

    def _on_script_changed(self) -> None:
        """Show/hide hyperparameter panels based on the selected script name."""
        if self._is_memd_script():
            self._xlstm_frame.grid_remove()
            self._memd_frame.grid()
        else:
            self._memd_frame.grid_remove()
            self._xlstm_frame.grid()

    # ── CLI argument builder ──────────────────────────────────────────

    def _build_cli_args(self) -> list[str]:
        """Assemble the CLI argument list to pass to the training script."""
        csv_path   = os.path.join(DATASET_OUTPUT_DIR, self._csv_var.get())
        target_col = self._target_var.get().strip() or "BTC/USD"
        save_dir   = self._save_dir_var.get().strip()
        args = ["--dataset", csv_path, "--target_col", target_col,
                "--save_dir", save_dir]

        if self._is_memd_script():
            v = self._memd_vars
            args += [
                "--sequence_length", v["sequence_length"].get(),
                "--epochs",          v["epochs_per_tcn"].get(),
                "--batch_size",      v["batch_size"].get(),
                "--learning_rate",   v["learning_rate"].get(),
                "--kernel_size",     v["kernel_size"].get(),
                "--dilations",       v["dilations"].get(),
                "--dropout",         v["dropout"].get(),
                "--K",               v["K"].get(),
                "--max_imfs",        v["max_imfs"].get(),
                "--max_sift",        v["max_sift"].get(),
                "--sd_threshold",    v["sd_threshold"].get(),
            ]
        else:
            v = self._xlstm_vars
            args += [
                "--sequence_length",          v["sequence_length"].get(),
                "--epochs",                   v["epochs"].get(),
                "--batch_size",               v["batch_size"].get(),
                "--learning_rate",            v["learning_rate"].get(),
                "--embedding_dim",            v["embedding_dim"].get(),
                "--output_size",              v["output_size"].get(),
                "--early_stopping_patience",  v["early_stopping_patience"].get(),
                "--train_split",              v["train_split"].get(),
                "--val_split",                v["val_split"].get(),
            ]
        return args

    # ── Run logic ─────────────────────────────────────────────────────

    def _on_run(self) -> None:
        script_name = self._current_script_name()
        if not script_name or script_name == "(none selected)":
            self._log(
                "No training script selected. Choose one from the 'AI Training Method' "
                "dropdown in the main window first.\n",
                tag="error",
            )
            return
        csv_name = self._csv_var.get()
        if not csv_name:
            self._log("No dataset CSV selected. Please choose one above.\n", tag="error")
            return

        # Keep the script label up to date in case main-window selection changed
        self._script_label.config(text=script_name)
        script_path = os.path.join(self._dir, script_name)
        extra_args  = self._build_cli_args()
        self._log(
            f"\n=== Training: {script_name} ===\n"
            f"    {' '.join(extra_args)}\n",
            tag="head",
        )
        self._set_running(True)
        threading.Thread(
            target=self._run_script, args=(script_path, extra_args), daemon=True
        ).start()

    def _run_script(self, script_path: str, extra_args: list[str]) -> None:
        """Execute the training script as a subprocess; stream output to console."""
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
            self._output_queue.put(None)  # sentinel → _poll_output sets idle state

    def _on_stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._log("\n[Training process terminated by user]\n", tag="error")
        self._set_running(False)

    def _set_running(self, is_running: bool) -> None:
        self._run_btn.config(state="disabled" if is_running else "normal")
        self._stop_btn.config(state="normal"   if is_running else "disabled")

    # ── Output polling ────────────────────────────────────────────────

    def _poll_output(self) -> None:
        try:
            while True:
                item = self._output_queue.get_nowait()
                if item is None:
                    self._set_running(False)
                else:
                    self._log(item)
        except queue.Empty:
            pass
        finally:
            if self._win.winfo_exists():
                self._win.after(50, self._poll_output)

    # ── Console helpers ───────────────────────────────────────────────

    def _log(self, text: str, tag: str = "") -> None:
        self._console.config(state="normal")
        if tag:
            self._console.insert("end", text, tag)
        else:
            self._console.insert("end", text)
        self._console.see("end")
        self._console.config(state="disabled")

    # ── Persist + close ───────────────────────────────────────────────

    def _save_to_stage(self) -> None:
        """Write all current values back to the stage dict for persistence."""
        self._stage["dataset_csv"] = self._csv_var.get()
        self._stage["target_col"]  = self._target_var.get()
        self._stage["save_dir"]    = self._save_dir_var.get()
        for k, v in self._xlstm_vars.items():
            self._stage[f"xlstm_{k}"] = v.get()
        for k, v in self._memd_vars.items():
            self._stage[f"memd_{k}"] = v.get()

    def _on_close(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._save_to_stage()
        self._win.destroy()


class MainWindow:
    """Primary application window."""

    CONSOLE_BG = "#1e1e1e"
    CONSOLE_FG = "#d4d4d4"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Model Designer")
        self.root.minsize(980, 680)

        # Active subprocess
        self._process: subprocess.Popen | None = None
        # Live output lines → console
        self._output_queue: queue.Queue[str | None] = queue.Queue()
        # Sequential run queue
        # Each entry: (script_path: str, extra_args: list[str])
        self._run_queue: collections.deque[tuple[str, list[str]]] = collections.deque()

        # Per-stage state populated by _build_ui.
        # Single-select stages: {"multi": False, "var": StringVar,
        #                         "combo": Combobox, "run_btn": Button, "dir": str}
        # Multi-select stages:  {"multi": True, "label_text": str, "dir": str,
        #                         "selected": set[str], "status_var": StringVar,
        #                         "configure_btn": Button, "run_btn": Button}
        self._stages: list[dict] = []

        self._build_ui()
        self._poll_output()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Assemble all widgets."""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # ── Pipeline selector panel ────────────────────────────────────
        selector_panel = ttk.LabelFrame(
            self.root, text="Pipeline Stages", padding=(10, 6)
        )
        selector_panel.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        selector_panel.columnconfigure(1, weight=1)

        for row_idx, (label_text, rel_dir, multi, diagram_rel) in enumerate(PIPELINE_STAGES):
            abs_dir = os.path.join(ROOT_DIR, rel_dir)

            # Label
            ttk.Label(selector_panel, text=label_text + ":", anchor="w").grid(
                row=row_idx, column=0, sticky="w", padx=(0, 10), pady=3
            )

            if multi:
                # ── Multi-select: status label + Configure button ──────
                stage: dict = {
                    "multi": True,
                    "label_text": label_text,
                    "dir": abs_dir,
                    "selected": set(),
                }

                status_var = tk.StringVar(value="0 scripts selected")
                stage["status_var"] = status_var

                ttk.Label(
                    selector_panel,
                    textvariable=status_var,
                    foreground="grey",
                    anchor="w",
                ).grid(row=row_idx, column=1, sticky="ew", pady=3, padx=(0, 4))

                # "Dataset Processing Method" opens ProcessDatasetWindow (run + console inside).
                # All other multi-select stages get a Configure + Run button pair.
                if label_text == "Dataset Processing Method":
                    analysis_dir = os.path.join(abs_dir, "Dataset_Analysis_Methods")
                    analyse_btn = ttk.Button(
                        selector_panel,
                        text="Analyse Data Collection",
                        width=22,
                        command=lambda d=analysis_dir: AnalysisWindow(self.root, d),
                    )
                    analyse_btn.grid(row=row_idx, column=2, padx=(4, 0), pady=3)
                    stage["analyse_btn"] = analyse_btn

                    process_btn = ttk.Button(
                        selector_panel,
                        text="Process Dataset",
                        width=15,
                        command=lambda s=stage: ProcessDatasetWindow(self.root, s),
                    )
                    process_btn.grid(row=row_idx, column=3, padx=(4, 0), pady=3)
                    stage["process_btn"] = process_btn

                    feature_dir = os.path.join(abs_dir, "Dataset_Feature_Selection")
                    feature_btn = ttk.Button(
                        selector_panel,
                        text="Feature Selection",
                        width=16,
                        command=lambda d=feature_dir: FeatureSelectionWindow(self.root, d),
                    )
                    feature_btn.grid(row=row_idx, column=4, padx=(4, 0), pady=3)
                    stage["feature_btn"] = feature_btn
                    stage["run_btn"]     = None  # run lives inside dedicated windows
                else:
                    configure_btn = ttk.Button(
                        selector_panel,
                        text="Configure",
                        width=10,
                        command=lambda s=stage: ConfigureWindow(self.root, s),
                    )
                    configure_btn.grid(row=row_idx, column=2, padx=(4, 0), pady=3)
                    stage["configure_btn"] = configure_btn

                    run_btn = ttk.Button(
                        selector_panel,
                        text="▶ Run",
                        width=8,
                        command=lambda s=stage: self._on_run_multi(s),
                    )
                    run_btn.grid(row=row_idx, column=3, padx=(4, 0), pady=3)
                    stage["run_btn"] = run_btn

            else:
                # ── Single-select: Combobox ────────────────────────────
                var = tk.StringVar()
                combo = ttk.Combobox(
                    selector_panel, textvariable=var, state="readonly", width=40
                )
                combo.grid(row=row_idx, column=1, sticky="ew", pady=3)

                stage = {"multi": False, "var": var, "combo": combo,
                         "dir": abs_dir, "diagram_btn": None}

                if diagram_rel:
                    # ── Optional "Create Diagram" button ───────────────
                    diagram_abs = os.path.join(ROOT_DIR, diagram_rel)
                    diag_btn = ttk.Button(
                        selector_panel,
                        text="Create Diagram",
                        width=14,
                        command=lambda p=diagram_abs, v=var, d=abs_dir: self._on_run_diagram(p, v, d),
                    )
                    diag_btn.grid(row=row_idx, column=2, padx=(8, 0), pady=3)
                    stage["diagram_btn"] = diag_btn
                    run_col, run_span = 3, 1
                else:
                    run_col, run_span = 2, 2

                # ── AI Training Method: Configure button opens dedicated window ─
                if label_text == "AI Training Method":
                    cfg_btn = ttk.Button(
                        selector_panel,
                        text="Configure",
                        width=10,
                        command=lambda s=stage: TrainingConfigureWindow(self.root, s),
                    )
                    cfg_btn.grid(row=row_idx, column=run_col, padx=(4, 0), pady=3,
                                 columnspan=run_span, sticky="w")
                    stage["training_configure_btn"] = cfg_btn
                    stage["run_btn"] = None   # Run lives inside the configure window
                else:
                    run_btn = ttk.Button(
                        selector_panel,
                        text="▶ Run",
                        width=8,
                        command=lambda d=abs_dir, v=var: self._on_run_single(d, v),
                    )
                    run_btn.grid(row=row_idx, column=run_col, padx=(4, 0), pady=3,
                                 columnspan=run_span, sticky="w")
                    stage["run_btn"] = run_btn

                # Populate combobox now
                self._populate_combo(stage)

            self._stages.append(stage)

        # Refresh all button (bottom-right of the panel)
        ttk.Button(
            selector_panel,
            text="↺ Refresh All",
            command=self._refresh_all,
            width=12,
        ).grid(
            row=len(PIPELINE_STAGES), column=3, padx=(8, 0), pady=(6, 0), sticky="e"
        )

        # ── Console output ─────────────────────────────────────────────
        console_frame = ttk.LabelFrame(
            self.root, text="Console Output", padding=(6, 4)
        )
        console_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self.console = scrolledtext.ScrolledText(
            console_frame,
            state="disabled",
            bg=self.CONSOLE_BG,
            fg=self.CONSOLE_FG,
            insertbackground=self.CONSOLE_FG,
            font=("Consolas", 10),
            wrap="word",
            relief="flat",
        )
        self.console.grid(row=0, column=0, sticky="nsew")

        self.console.tag_config("info",  foreground="#9cdcfe")
        self.console.tag_config("error", foreground="#f48771")
        self.console.tag_config("dim",   foreground="#6a9955")
        self.console.tag_config("head",  foreground="#dcdcaa")

        # ── Bottom action bar ──────────────────────────────────────────
        action_bar = ttk.Frame(self.root, padding=(8, 6))
        action_bar.grid(row=2, column=0, sticky="ew")

        self.stop_btn = ttk.Button(
            action_bar,
            text="■  Stop",
            command=self._on_stop,
            width=10,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=(6, 0))

        ttk.Button(
            action_bar,
            text="Clear",
            command=self._clear_console,
            width=8,
        ).pack(side="left", padx=(6, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(action_bar, textvariable=self.status_var).pack(
            side="right", padx=(0, 4)
        )

    # ------------------------------------------------------------------
    # Script discovery / combo population
    # ------------------------------------------------------------------

    def _populate_combo(self, stage: dict) -> None:
        """Fill a single-select stage's combobox from its source directory."""
        scripts = discover_scripts(stage["dir"])
        stage["combo"]["values"] = scripts
        if scripts:
            if stage["var"].get() not in scripts:
                stage["var"].set(scripts[0])
        else:
            stage["var"].set("")

    def _refresh_all(self) -> None:
        """Re-scan all stage directories; update comboboxes / prune selections."""
        for stage in self._stages:
            if stage["multi"]:
                # Drop any selected scripts that no longer exist on disk
                available = set(discover_scripts(stage["dir"]))
                stage["selected"] = stage["selected"] & available
                n = len(stage["selected"])
                stage["status_var"].set(f"{n} script{'s' if n != 1 else ''} selected")
            else:
                self._populate_combo(stage)
        self._log("All script lists refreshed.\n", tag="dim")

    # ------------------------------------------------------------------
    # Run helpers
    # ------------------------------------------------------------------

    def _set_running(self, is_running: bool) -> None:
        """Toggle button states and update status label."""
        state = "disabled" if is_running else "normal"
        self.stop_btn.config(state="normal" if is_running else "disabled")
        for stage in self._stages:
            if stage["run_btn"] is not None:
                stage["run_btn"].config(state=state)
            if stage["multi"]:
                if stage.get("configure_btn"):
                    stage["configure_btn"].config(state=state)
                if stage.get("process_btn"):
                    stage["process_btn"].config(state=state)
                if stage.get("analyse_btn"):
                    stage["analyse_btn"].config(state=state)
                if stage.get("feature_btn"):
                    stage["feature_btn"].config(state=state)
            elif stage.get("diagram_btn"):
                stage["diagram_btn"].config(state=state)
            if stage.get("training_configure_btn"):
                stage["training_configure_btn"].config(state=state)
        if not is_running:
            self.status_var.set("Ready")

    def _on_run_single(self, abs_dir: str, var: tk.StringVar) -> None:
        """Run the script selected in a single-select combobox stage."""
        selected = var.get()
        if not selected:
            self._log("No script selected.\n", tag="error")
            return
        self._run_queue.clear()
        self._run_queue.append((os.path.join(abs_dir, selected), []))
        self._start_next()

    def _on_run_diagram(self, diagram_script: str, model_var: tk.StringVar,
                        model_dir: str) -> None:
        """Run the diagram generator, passing the currently-selected model script."""
        selected = model_var.get()
        if not selected:
            self._log(
                "No model selected in 'AI Model Designs'.  "
                "Please choose a model from the combobox first.\n",
                tag="error",
            )
            return
        model_path = os.path.join(model_dir, selected)
        self._run_queue.clear()
        self._run_queue.append((diagram_script, [model_path]))
        self._log(
            f"\n=== Create Diagram: {selected} ===\n",
            tag="head",
        )
        self._start_next()

    def _on_run_multi(self, stage: dict) -> None:
        """Run all checked scripts in a multi-select stage."""
        selected = sorted(stage["selected"])   # deterministic order
        if not selected:
            self._log(
                f"No scripts selected for '{stage['label_text']}'.  "
                "Use 'Configure' to choose scripts.\n",
                tag="error",
            )
            return

        # Build extra CLI args for stages that carry date/freq config
        extra: list[str] = []
        if stage.get("label_text", "") in _DATE_CONFIG_STAGES:
            start = stage.get("start_date", "2015-01-01")
            end   = stage.get("end_date",   "2025-02-01")
            freq  = stage.get("freq",       "1d")
            extra = ["--start", start, "--end", end, "--freq", freq]

        # Pass the selected dataset CSV path for processing stages
        if stage.get("label_text", "") in _DATASET_SELECT_STAGES:
            csv_name = stage.get("dataset_csv", "")
            if not csv_name:
                self._log(
                    "No dataset CSV selected for 'Dataset Processing Method'.  "
                    "Open Configure and choose a CSV file.\n",
                    tag="error",
                )
                return
            csv_path = os.path.join(DATASET_OUTPUT_DIR, csv_name)
            extra = ["--dataset", csv_path]

        self._run_queue.clear()
        for name in selected:
            self._run_queue.append((os.path.join(stage["dir"], name), list(extra)))

        # For Dataset Collection stages, queue a post-run merge step that
        # combines all individual CSVs into one consolidated file.
        if stage.get("label_text", "") in _DATE_CONFIG_STAGES:
            self._run_queue.append((None, lambda s=stage: self._merge_collection_csvs(s)))

        self._log(
            f"\n=== Running {len(selected)} script(s) from '{stage['label_text']}' ===\n",
            tag="head",
        )
        self._start_next()

    def _start_next(self) -> None:
        """Pop and launch the next script (or callback) in the queue, if any."""
        if not self._run_queue:
            self._set_running(False)
            self._log("\n=== All queued scripts finished ===\n", tag="head")
            return

        entry = self._run_queue.popleft()

        # Callback entry: (None, callable) – e.g. post-run CSV merge
        if entry[0] is None:
            _, callback = entry
            self._log("\n--- Running post-process step ---\n", tag="dim")
            self.status_var.set("Post-processing …")
            thread = threading.Thread(
                target=self._run_callback, args=(callback,), daemon=True
            )
            thread.start()
            return

        script_path, extra_args = entry
        script_name = os.path.basename(script_path)
        self._log(f"\n--- Running: {script_path} ---\n", tag="dim")
        self.status_var.set(f"Running {script_name} …")
        self._set_running(True)

        thread = threading.Thread(
            target=self._run_script, args=(script_path, extra_args), daemon=True
        )
        thread.start()

    def _run_callback(self, callback) -> None:
        """Execute a post-run *callback* on a background thread, then advance the queue."""
        try:
            callback()
        except Exception as exc:
            self._output_queue.put(f"[ERROR] Post-run step failed: {exc}\n")
        finally:
            self._output_queue.put(None)  # sentinel → _poll_output → _start_next

    def _merge_collection_csvs(self, stage: dict) -> None:
        """
        After all Dataset Collection scripts finish, merge every individual
        single-column CSV they produced into one combined CSV, then delete
        the individual files.

        Expected file names are derived directly from the selected script
        names (e.g. ``currency_btc_price.py`` → ``currency_btc_price.csv``).
        The combined file is written to the same ``dataset_output/`` folder
        and named ``{start_year}-{end_year}_dataset_collection.csv``.
        """
        output_dir = os.path.join(os.path.dirname(stage["dir"]), "dataset_output")
        selected   = sorted(stage["selected"])

        # Map each selected script to its expected CSV output path
        csv_paths: list[str] = []
        missing:   list[str] = []
        for script_name in selected:
            csv_name = os.path.splitext(script_name)[0] + ".csv"
            csv_path = os.path.join(output_dir, csv_name)
            if os.path.isfile(csv_path):
                csv_paths.append(csv_path)
            else:
                missing.append(csv_name)

        if missing:
            self._output_queue.put(
                f"[WARN] {len(missing)} expected output file(s) not found:\n"
                + "".join(f"  {m}\n" for m in missing)
            )
        if not csv_paths:
            self._output_queue.put("[WARN] No CSV files to merge – skipping.\n")
            return

        self._output_queue.put(
            f"\n=== Merging {len(csv_paths)} CSV file(s) into combined dataset ===\n"
        )
        try:
            frames: list[pd.DataFrame] = []
            for csv_path in csv_paths:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                frames.append(df)
                self._output_queue.put(
                    f"  Read : {os.path.basename(csv_path)}"
                    f" – {df.shape[0]} rows, columns: {list(df.columns)}\n"
                )

            # Outer-join on the shared datetime index so every row and column
            # from every source is preserved in the correct sequential order.
            combined: pd.DataFrame = pd.concat(frames, axis=1, join="outer")
            combined.sort_index(inplace=True)

            # Build the combined filename from the configured date range
            start = stage.get("start_date", "start")
            end   = stage.get("end_date",   "end")
            try:
                combined_name = f"{start[:4]}-{end[:4]}_dataset_collection.csv"
            except Exception:
                combined_name = "dataset_collection.csv"

            combined_path = os.path.join(output_dir, combined_name)
            combined.to_csv(combined_path)
            self._output_queue.put(
                f"  Combined dataset saved → {combined_path}\n"
                f"  Shape : {combined.shape[0]} rows × {combined.shape[1]} column(s)\n"
                f"  Columns: {list(combined.columns)}\n"
            )

            # Delete the individual single-column files now that the combined
            # file is safely on disk.
            for csv_path in csv_paths:
                try:
                    os.remove(csv_path)
                    self._output_queue.put(
                        f"  Deleted: {os.path.basename(csv_path)}\n"
                    )
                except Exception as exc:
                    self._output_queue.put(
                        f"  [WARN] Could not delete {os.path.basename(csv_path)}: {exc}\n"
                    )

            self._output_queue.put(
                f"\n=== Dataset Collection merge complete → {combined_name} ===\n",
            )
        except Exception as exc:
            self._output_queue.put(f"[ERROR] CSV merge failed: {exc}\n")

    def _run_script(self, script_path: str, extra_args: list[str]) -> None:
        """Execute *script_path* as a subprocess; stream stdout to the queue."""
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
            exit_code = self._process.returncode
            self._output_queue.put(
                f"\n--- Process finished with exit code {exit_code} ---\n"
            )
        except Exception as exc:
            self._output_queue.put(f"[ERROR] {exc}\n")
        finally:
            self._process = None
            # None sentinel → poll loop will call _start_next
            self._output_queue.put(None)

    def _on_stop(self) -> None:
        """Terminate the running subprocess and clear any queued scripts."""
        self._run_queue.clear()
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._log("\n[Process terminated by user — queue cleared]\n", tag="error")

    # ------------------------------------------------------------------
    # Output polling
    # ------------------------------------------------------------------

    def _poll_output(self) -> None:
        """Drain the output queue; on sentinel, advance to the next script."""
        try:
            while True:
                item = self._output_queue.get_nowait()
                if item is None:
                    # Current script done — start the next one (or go idle)
                    self._start_next()
                else:
                    self._log(item)
        except queue.Empty:
            pass
        finally:
            self.root.after(50, self._poll_output)

    # ------------------------------------------------------------------
    # Console helpers
    # ------------------------------------------------------------------

    def _log(self, text: str, tag: str = "") -> None:
        self.console.config(state="normal")
        if tag:
            self.console.insert("end", text, tag)
        else:
            self.console.insert("end", text)
        self.console.see("end")
        self.console.config(state="disabled")

    def _clear_console(self) -> None:
        self.console.config(state="normal")
        self.console.delete("1.0", "end")
        self.console.config(state="disabled")
