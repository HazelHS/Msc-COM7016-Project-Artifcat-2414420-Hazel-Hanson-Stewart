"""
configure_window.py
-------------------
ConfigureWindow – modal checklist for multi-select pipeline stages.

Depending on the stage label the window also shows:
  - Date / frequency panel   (Dataset Collection Method)
  - CSV dataset picker        (Dataset Processing Method)
  - Trained-model + CSV picker (Model Evaluation Method)
"""

import tkinter as tk
from tkinter import ttk

from .constants import (
    DATASET_OUTPUT_DIR,
    TRAINED_MODEL_DIR,
    FREQ_OPTIONS,
    DATE_CONFIG_STAGES,
    DATASET_SELECT_STAGES,
    EVAL_STAGES,
)
from .utils import discover_scripts, discover_csvs, discover_models


class ConfigureWindow:
    """
    Toplevel window that lists all .py scripts in a directory with a
    checkbox next to each one.  The caller reads back ``stage["selected"]``
    (a set of filenames) after the window is closed.

    For Dataset Collection stages an extra panel is shown at the bottom
    with start/end date inputs and a frequency drop-down.
    """

    def __init__(self, parent: tk.Widget, stage: dict) -> None:
        self._stage = stage
        self._show_date_config:    bool = stage.get("label_text", "") in DATE_CONFIG_STAGES
        self._show_dataset_select: bool = stage.get("label_text", "") in DATASET_SELECT_STAGES
        self._show_eval_select:    bool = stage.get("label_text", "") in EVAL_STAGES

        # Seed defaults into the stage dict on first open
        if self._show_date_config:
            stage.setdefault("start_date", "2015-01-01")
            stage.setdefault("end_date",   "2025-02-01")
            stage.setdefault("freq",       "1d")
        if self._show_dataset_select:
            stage.setdefault("dataset_csv", "")
        if self._show_eval_select:
            stage.setdefault("model_file",  "")
            stage.setdefault("dataset_csv", "")

        self._win = tk.Toplevel(parent)
        self._win.title(f"Configure: {stage['label_text']}")
        self._win.resizable(False, True)
        self._win.minsize(540, 280)
        self._win.grab_set()  # modal

        self._check_vars: dict[str, tk.BooleanVar] = {}

        # Date/freq tk variables (created only when _show_date_config is True)
        self._start_var: tk.StringVar | None = None
        self._end_var:   tk.StringVar | None = None
        self._freq_var:  tk.StringVar | None = None

        # CSV picker variable (created only when _show_dataset_select is True)
        self._csv_var: tk.StringVar | None = None

        # Eval-stage variables (created only when _show_eval_select is True)
        self._eval_model_var: tk.StringVar | None = None
        self._eval_csv_var:   tk.StringVar | None = None

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
        canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        self._canvas = canvas
        self._populate_checks()

        # ── Stage-specific extra panels ────────────────────────────────
        if self._show_date_config:
            self._build_date_panel(outer)
        if self._show_dataset_select:
            self._build_csv_panel(outer)
        if self._show_eval_select:
            self._build_eval_panel(outer)

        # Buttons row
        btn_row = ttk.Frame(outer)
        btn_row.pack(fill="x", pady=(8, 0))

        ttk.Button(btn_row, text="Select All",   command=self._select_all,   width=12).pack(side="left")
        ttk.Button(btn_row, text="Deselect All", command=self._deselect_all, width=12).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="↺ Refresh",    command=self._refresh,      width=10).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="OK",           command=self._ok,           width=8).pack(side="right")

    # ── Extra panels ───────────────────────────────────────────────────

    def _build_date_panel(self, parent: ttk.Frame) -> None:
        """Add start/end date entries and a frequency combobox."""
        date_frame = ttk.LabelFrame(parent, text="Date Range & Frequency", padding=(8, 6))
        date_frame.pack(fill="x", pady=(0, 6))
        date_frame.columnconfigure(1, weight=1)
        date_frame.columnconfigure(3, weight=1)

        ttk.Label(date_frame, text="Start Date (YYYY-MM-DD):").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self._start_var = tk.StringVar(value=self._stage.get("start_date", "2015-01-01"))
        ttk.Entry(date_frame, textvariable=self._start_var, width=14).grid(
            row=0, column=1, sticky="ew", pady=3
        )

        ttk.Label(date_frame, text="End Date (YYYY-MM-DD):").grid(
            row=0, column=2, sticky="w", padx=(12, 6), pady=3
        )
        self._end_var = tk.StringVar(value=self._stage.get("end_date", "2025-02-01"))
        ttk.Entry(date_frame, textvariable=self._end_var, width=14).grid(
            row=0, column=3, sticky="ew", pady=3
        )

        ttk.Label(date_frame, text="Data Frequency:").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self._freq_var = tk.StringVar(value=self._stage.get("freq", "1d"))
        ttk.Combobox(
            date_frame,
            textvariable=self._freq_var,
            values=FREQ_OPTIONS,
            state="readonly",
            width=14,
        ).grid(row=1, column=1, sticky="w", pady=3)

    def _build_csv_panel(self, parent: ttk.Frame) -> None:
        """Add a CSV dataset picker for Dataset Processing stages."""
        csv_frame = ttk.LabelFrame(
            parent,
            text="Dataset CSV (passed to scripts via --dataset)",
            padding=(8, 6),
        )
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

    def _build_eval_panel(self, parent: ttk.Frame) -> None:
        """Add a trained-model picker and a dataset CSV picker for the
        Model Evaluation Method stage."""
        eval_frame = ttk.LabelFrame(
            parent,
            text="Evaluation Inputs  (passed to scripts via --model and --dataset)",
            padding=(8, 6),
        )
        eval_frame.pack(fill="x", pady=(0, 6))
        eval_frame.columnconfigure(1, weight=1)

        # Trained model file
        ttk.Label(eval_frame, text="Trained Model:").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self._eval_model_var = tk.StringVar(value=self._stage.get("model_file", ""))
        self._eval_model_combo = ttk.Combobox(
            eval_frame, textvariable=self._eval_model_var, state="readonly", width=46
        )
        self._eval_model_combo.grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Button(
            eval_frame, text="\u21ba", width=3, command=self._refresh_models
        ).grid(row=0, column=2, padx=(4, 0), pady=3)

        # Dataset CSV
        ttk.Label(eval_frame, text="Dataset CSV:").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self._eval_csv_var = tk.StringVar(value=self._stage.get("dataset_csv", ""))
        self._eval_csv_combo = ttk.Combobox(
            eval_frame, textvariable=self._eval_csv_var, state="readonly", width=46
        )
        self._eval_csv_combo.grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Button(
            eval_frame, text="\u21ba", width=3, command=self._refresh_eval_csvs
        ).grid(row=1, column=2, padx=(4, 0), pady=3)

        self._refresh_models()
        self._refresh_eval_csvs()

    # ── Refresh helpers ────────────────────────────────────────────────

    def _refresh_csvs(self) -> None:
        """Repopulate the CSV combobox from the dataset_output folder."""
        csvs = discover_csvs(DATASET_OUTPUT_DIR)
        if hasattr(self, "_csv_combo") and self._csv_combo is not None:
            self._csv_combo["values"] = csvs
            current = self._csv_var.get() if self._csv_var else ""
            if current not in csvs:
                self._csv_var.set(csvs[0] if csvs else "")

    def _refresh_models(self) -> None:
        """Repopulate the trained-model combobox."""
        models = discover_models(TRAINED_MODEL_DIR)
        if hasattr(self, "_eval_model_combo") and self._eval_model_combo is not None:
            self._eval_model_combo["values"] = models
            current = self._eval_model_var.get() if self._eval_model_var else ""
            if current not in models:
                self._eval_model_var.set(models[0] if models else "")

    def _refresh_eval_csvs(self) -> None:
        """Repopulate the eval dataset CSV combobox."""
        csvs = discover_csvs(DATASET_OUTPUT_DIR)
        if hasattr(self, "_eval_csv_combo") and self._eval_csv_combo is not None:
            self._eval_csv_combo["values"] = csvs
            current = self._eval_csv_var.get() if self._eval_csv_var else ""
            if current not in csvs:
                self._eval_csv_var.set(csvs[0] if csvs else "")

    # ── Checklist ──────────────────────────────────────────────────────

    def _populate_checks(self) -> None:
        """Destroy existing checkboxes and rebuild from directory listing."""
        for widget in self._inner.winfo_children():
            widget.destroy()
        self._check_vars.clear()

        scripts = discover_scripts(self._stage["dir"])
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
        """Commit checked scripts and optional date/freq/csv back to the stage."""
        self._stage["selected"] = {
            name for name, var in self._check_vars.items() if var.get()
        }
        if self._show_date_config:
            self._stage["start_date"] = self._start_var.get().strip()
            self._stage["end_date"]   = self._end_var.get().strip()
            self._stage["freq"]       = self._freq_var.get()
        if self._show_dataset_select and self._csv_var is not None:
            self._stage["dataset_csv"] = self._csv_var.get()
        if self._show_eval_select:
            if self._eval_model_var is not None:
                self._stage["model_file"]  = self._eval_model_var.get()
            if self._eval_csv_var is not None:
                self._stage["dataset_csv"] = self._eval_csv_var.get()
        self._stage["status_var"].set(self._status_text())
        self._win.destroy()

    def _status_text(self) -> str:
        n = len(self._stage["selected"])
        return f"{n} script{'s' if n != 1 else ''} selected"
