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
from tkinter import ttk, scrolledtext
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


def discover_scripts(directory: str) -> list[str]:
    """Return a sorted list of .py filenames found in *directory*."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".py") and not f.startswith("__")
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
        self._show_date_config: bool = stage.get("label_text", "") in _DATE_CONFIG_STAGES

        # Seed defaults into the stage dict on first open
        if self._show_date_config:
            stage.setdefault("start_date", "2015-01-01")
            stage.setdefault("end_date",   "2025-02-01")
            stage.setdefault("freq",       "1d")

        self._win = tk.Toplevel(parent)
        self._win.title(f"Configure: {stage['label_text']}")
        self._win.resizable(False, True)
        self._win.minsize(460, 220)
        self._win.grab_set()           # modal

        self._check_vars: dict[str, tk.BooleanVar] = {}

        # Date/freq tk variables (only created when _show_date_config is True)
        self._start_var: tk.StringVar | None = None
        self._end_var:   tk.StringVar | None = None
        self._freq_var:  tk.StringVar | None = None

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

        # Buttons row
        btn_row = ttk.Frame(outer)
        btn_row.pack(fill="x", pady=(8, 0))

        ttk.Button(btn_row, text="Select All",   command=self._select_all,   width=12).pack(side="left")
        ttk.Button(btn_row, text="Deselect All", command=self._deselect_all, width=12).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="↺ Refresh",    command=self._refresh,      width=10).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="OK",           command=self._ok,           width=8).pack(side="right")

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
        """Commit checked scripts and (optionally) date/freq back to the stage."""
        self._stage["selected"] = {
            name for name, var in self._check_vars.items() if var.get()
        }
        if self._show_date_config:
            self._stage["start_date"] = self._start_var.get().strip()
            self._stage["end_date"]   = self._end_var.get().strip()
            self._stage["freq"]       = self._freq_var.get()
        self._stage["status_var"].set(self._status_text())
        self._win.destroy()

    def _status_text(self) -> str:
        n = len(self._stage["selected"])
        return f"{n} script{'s' if n != 1 else ''} selected"


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
            stage["run_btn"].config(state=state)
            if stage["multi"]:
                stage["configure_btn"].config(state=state)
            elif stage.get("diagram_btn"):
                stage["diagram_btn"].config(state=state)
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
