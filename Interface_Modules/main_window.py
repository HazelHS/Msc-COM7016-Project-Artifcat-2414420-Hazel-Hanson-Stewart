"""
main_window.py
--------------
Main GUI window for the Model Designer application.

Pipeline stages are displayed top-to-bottom.  Stages that support
multi-script selection ("Dataset Collection Method",
"Dataset Processing Method", "Model Evaluation Method") show a
"Configure" button that opens a checklist window; the remaining
stages use a combobox for single-script selection.

A "Run" button per stage and a "Run All" button in the action bar
execute every selected script sequentially in order.
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

class ConfigureWindow:
    """
    Toplevel window that lists all .py scripts in a directory with
    a checkbox next to each one.  The caller reads back `stage["selected"]`
    (a set of filenames) after the window is closed.
    """

    def __init__(self, parent: tk.Widget, stage: dict) -> None:
        self._stage = stage

        self._win = tk.Toplevel(parent)
        self._win.title(f"Configure: {stage['label_text']}")
        self._win.resizable(False, True)
        self._win.minsize(420, 200)
        self._win.grab_set()           # modal

        self._check_vars: dict[str, tk.BooleanVar] = {}
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

        # Buttons row
        btn_row = ttk.Frame(outer)
        btn_row.pack(fill="x")

        ttk.Button(btn_row, text="Select All",   command=self._select_all,   width=12).pack(side="left")
        ttk.Button(btn_row, text="Deselect All", command=self._deselect_all, width=12).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="↺ Refresh",    command=self._refresh,      width=10).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="OK",           command=self._ok,           width=8).pack(side="right")

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
        """Commit checked scripts back to the stage and close."""
        self._stage["selected"] = {
            name for name, var in self._check_vars.items() if var.get()
        }
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
        # Sequential run queue (for Run All)
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
        self.run_all_btn.config(state=state)
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
        self._run_queue.clear()
        for name in selected:
            self._run_queue.append((os.path.join(stage["dir"], name), []))
        self._log(
            f"\n=== Running {len(selected)} script(s) from '{stage['label_text']}' ===\n",
            tag="head",
        )
        self._start_next()

    def _on_run_all(self) -> None:
        """Queue every stage's selected script(s) and run them in order."""
        self._run_queue.clear()
        for stage in self._stages:
            if stage["multi"]:
                for name in sorted(stage["selected"]):
                    self._run_queue.append((os.path.join(stage["dir"], name), []))
            else:
                selected = stage["var"].get()
                if selected:
                    self._run_queue.append((os.path.join(stage["dir"], selected), []))

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

        script_path, extra_args = self._run_queue.popleft()
        script_name = os.path.basename(script_path)
        self._log(f"\n--- Running: {script_path} ---\n", tag="dim")
        self.status_var.set(f"Running {script_name} …")
        self._set_running(True)

        thread = threading.Thread(
            target=self._run_script, args=(script_path, extra_args), daemon=True
        )
        thread.start()

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
