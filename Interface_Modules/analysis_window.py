"""
analysis_window.py
------------------
AnalysisWindow – modal window for running Dataset Analysis scripts.

Lists all .py scripts in the ``Dataset_Analysis_Methods`` folder with
checkboxes, a CSV dataset picker, Run/Stop buttons and an embedded
console.  Scripts are executed sequentially as subprocesses.
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


class AnalysisWindow:
    """
    Toplevel window that lists all .py scripts in the
    Dataset_Analysis_Methods folder with checkboxes, a Run button,
    a Stop button, and an embedded console output area.
    Runs selected scripts sequentially as subprocesses.
    """

    CONSOLE_BG = "#1e1e1e"
    CONSOLE_FG = "#d4d4d4"

    def __init__(self, parent: tk.Widget, analysis_dir: str) -> None:
        """Open the analysis-script runner window.

        Args:
            parent: Parent widget that owns this ``Toplevel``.
            analysis_dir: Absolute path to the folder containing analysis scripts.
        """
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
        """Assemble all child widgets: CSV picker, checklist, buttons, and console."""
        outer = ttk.Frame(self._win, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)  # console row expands

        # ── Dataset CSV picker ───────────────────────────────────────
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
            csv_frame, text="↺", width=3, command=self._refresh_csvs
        ).grid(row=0, column=2, padx=(4, 0))
        self._refresh_csvs()

        # ── Scrollable checklist ───────────────────────────────────────
        list_frame = ttk.LabelFrame(
            outer, text="Available Analysis Scripts", padding=(6, 4)
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

        # ── Buttons row ─────────────────────────────────────────────
        btn_row = ttk.Frame(outer)
        btn_row.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        ttk.Button(btn_row, text="Select All",   command=self._select_all,        width=12).pack(side="left")
        ttk.Button(btn_row, text="Deselect All", command=self._deselect_all,      width=12).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="↺ Refresh",    command=self._populate_checks,   width=10).pack(side="left", padx=(4, 0))

        self._stop_btn = ttk.Button(
            btn_row, text="■  Stop", width=9, command=self._on_stop, state="disabled"
        )
        self._stop_btn.pack(side="right")

        self._run_btn = ttk.Button(
            btn_row, text="▶ Run", width=8, command=self._on_run
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
        if self._csv_var.get() not in csvs:
            self._csv_var.set(csvs[0] if csvs else "")

    def _select_all(self) -> None:
        """Tick all script checkboxes."""
        for var in self._check_vars.values():
            var.set(True)

    def _deselect_all(self) -> None:
        """Clear all script checkboxes."""
        for var in self._check_vars.values():
            var.set(False)

    # ── Run logic ──────────────────────────────────────────────────────

    def _on_run(self) -> None:
        """Validate selections and enqueue checked scripts for sequential execution."""
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
        """Pop and launch the next script in the run queue."""
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
            self._output_queue.put(
                f"\n--- Process finished with exit code {self._process.returncode} ---\n"
            )
        except Exception as exc:
            self._output_queue.put(f"[ERROR] {exc}\n")
        finally:
            self._process = None
            self._output_queue.put(None)  # sentinel

    def _on_stop(self) -> None:
        """Terminate the running subprocess and clear the pending queue."""
        self._run_queue.clear()
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._log("\n[Process terminated by user \u2014 queue cleared]\n", tag="error")

    def _set_running(self, is_running: bool) -> None:
        """Toggle Run/Stop button states.

        Args:
            is_running: When ``True`` the Run button is disabled and Stop enabled.
        """
        self._run_btn.config(state="disabled" if is_running else "normal")
        self._stop_btn.config(state="normal"   if is_running else "disabled")

    # ── Output polling ─────────────────────────────────────────────────

    def _poll_output(self) -> None:
        """Drain the output queue and flush lines to the console; reschedule every 50 ms."""
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
        """Append *text* to the embedded console, optionally coloured by *tag*.

        Args:
            text: The string to append.
            tag: Optional named tag for foreground colour.
        """
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
