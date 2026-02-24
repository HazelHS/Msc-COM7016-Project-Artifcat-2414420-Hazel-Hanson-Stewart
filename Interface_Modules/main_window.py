"""
main_window.py
--------------
Main GUI window for the Model Designer application.

Pipeline stages are displayed top-to-bottom, each with a labelled
combobox (auto-populated from its source folder) and an individual
"Run" button.  A "Run All" button in the action bar executes every
selected script sequentially in order.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import queue
import collections
import os
import sys


# ── Project root (two levels up from this file) ─────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Pipeline stage definitions ───────────────────────────────────────
# Each tuple: (display_label, path_relative_to_ROOT_DIR)
PIPELINE_STAGES: list[tuple[str, str]] = [
    ("Dataset Collection Method",   os.path.join("Dataset_Modules", "Dataset_Collection")),
    ("Dataset Processing Method",   os.path.join("Dataset_Modules", "Dataset_Processing_Methods")),
    ("AI Model Designs",            os.path.join("AI_Modules",      "Model_Designs")),
    ("AI Training Method",          os.path.join("AI_Modules",      "Training_Methods")),
    ("Model Evaluation Method",     os.path.join("Evaluation_Modules", "Evaluation_Metrics")),
]


def discover_scripts(directory: str) -> list[str]:
    """Return a sorted list of .py filenames found in *directory*."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".py") and not f.startswith("__")
    )


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
        # Sequential run queue (for Run All)
        self._run_queue: collections.deque[str] = collections.deque()

        # Per-stage state populated by _build_ui
        # Each entry: {"var": StringVar, "combo": Combobox,
        #               "run_btn": Button, "dir": str}
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

        for row_idx, (label_text, rel_dir) in enumerate(PIPELINE_STAGES):
            abs_dir = os.path.join(ROOT_DIR, rel_dir)

            # Label
            ttk.Label(selector_panel, text=label_text + ":", anchor="w").grid(
                row=row_idx, column=0, sticky="w", padx=(0, 10), pady=3
            )

            # Combobox
            var = tk.StringVar()
            combo = ttk.Combobox(
                selector_panel, textvariable=var, state="readonly", width=46
            )
            combo.grid(row=row_idx, column=1, sticky="ew", pady=3)

            # Individual Run button
            run_btn = ttk.Button(
                selector_panel,
                text="▶ Run",
                width=8,
                command=lambda d=abs_dir, v=var, b=None: self._on_run_single(d, v),
            )
            run_btn.grid(row=row_idx, column=2, padx=(8, 0), pady=3)

            stage = {"var": var, "combo": combo, "run_btn": run_btn, "dir": abs_dir}
            self._stages.append(stage)

            # Populate scripts now
            self._populate_combo(stage)

        # Refresh all button (top-right of the panel header area)
        ttk.Button(
            selector_panel,
            text="↺ Refresh All",
            command=self._refresh_all,
            width=12,
        ).grid(
            row=len(PIPELINE_STAGES), column=2, padx=(8, 0), pady=(6, 0), sticky="e"
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

        self.run_all_btn = ttk.Button(
            action_bar,
            text="▶▶  Run All",
            command=self._on_run_all,
            width=14,
        )
        self.run_all_btn.pack(side="left")

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
        """Fill a stage's combobox from its source directory."""
        scripts = discover_scripts(stage["dir"])
        stage["combo"]["values"] = scripts
        if scripts:
            if stage["var"].get() not in scripts:
                stage["var"].set(scripts[0])
        else:
            stage["var"].set("")

    def _refresh_all(self) -> None:
        """Re-scan all stage directories and update every combobox."""
        for stage in self._stages:
            self._populate_combo(stage)
        self._log("All script lists refreshed.\n", tag="dim")

    # ------------------------------------------------------------------
    # Run helpers
    # ------------------------------------------------------------------

    def _set_running(self, is_running: bool) -> None:
        """Toggle button states and update status label."""
        state = "disabled" if is_running else "normal"
        self.run_all_btn.config(state=state)
        self.stop_btn.config(state="normal" if is_running else "disabled")
        for stage in self._stages:
            stage["run_btn"].config(state=state)
        if not is_running:
            self.status_var.set("Ready")

    def _on_run_single(self, abs_dir: str, var: tk.StringVar) -> None:
        """Run the script selected in one specific combobox."""
        selected = var.get()
        if not selected:
            self._log("No script selected.\n", tag="error")
            return
        self._run_queue.clear()
        self._run_queue.append(os.path.join(abs_dir, selected))
        self._start_next()

    def _on_run_all(self) -> None:
        """Queue every stage's selected script and run them in order."""
        self._run_queue.clear()
        for stage in self._stages:
            selected = stage["var"].get()
            if selected:
                self._run_queue.append(os.path.join(stage["dir"], selected))

        if not self._run_queue:
            self._log("No scripts selected in any stage.\n", tag="error")
            return

        self._log(
            f"\n=== Run All: {len(self._run_queue)} script(s) queued ===\n",
            tag="head",
        )
        self._start_next()

    def _start_next(self) -> None:
        """Pop and launch the next script in the queue, if any."""
        if not self._run_queue:
            self._set_running(False)
            self._log("\n=== All queued scripts finished ===\n", tag="head")
            return

        script_path = self._run_queue.popleft()
        script_name = os.path.basename(script_path)
        self._log(f"\n--- Running: {script_path} ---\n", tag="dim")
        self.status_var.set(f"Running {script_name} …")
        self._set_running(True)

        thread = threading.Thread(
            target=self._run_script, args=(script_path,), daemon=True
        )
        thread.start()

    def _run_script(self, script_path: str) -> None:
        """Execute *script_path* as a subprocess; stream stdout to the queue."""
        try:
            self._process = subprocess.Popen(
                [sys.executable, script_path],
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
