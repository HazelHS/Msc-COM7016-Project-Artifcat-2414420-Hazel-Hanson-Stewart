"""
training_configure_window.py
----------------------------
TrainingConfigureWindow – configuration and execution window for the
AI Training Method pipeline stage.

Allows the user to:
  - Pick a training script from the Training_Methods directory
  - Pick an input dataset CSV and target column
  - Configure model-specific hyperparameters (xLSTM-TS or MEMD-TCN)
  - Launch / stop the training process
  - Monitor console output

Parameter groups are shown/hidden based on which training script is
selected:
  - Scripts whose filename contains 'memd' or 'tcn' → MEMD-TCN params
  - All other scripts                               → xLSTM-TS params

All configured values are persisted back into the stage dict when the
window is closed, so they survive between re-opens.
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog

import pandas as pd

from .constants import DATASET_OUTPUT_DIR, TRAINED_MODEL_DIR, ROOT_DIR
from .utils import discover_csvs


class TrainingConfigureWindow:
    """
    Toplevel configuration and execution window for the AI Training
    Method stage.
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
        "sequence_length": "30",
        "epochs_per_tcn":  "100",
        "batch_size":      "16",
        "learning_rate":   "0.0001",
        "kernel_size":     "2",
        "dilations":       "1,2,4",
        "dropout":         "0.2",
        "K":               "512",
        "max_imfs":        "12",
        "max_sift":        "50",
        "sd_threshold":    "0.2",
    }

    def __init__(self, parent: tk.Widget, stage: dict) -> None:
        self._stage = stage
        self._dir   = stage["dir"]
        self._process: subprocess.Popen | None = None
        self._output_queue: queue.Queue[str | None] = queue.Queue()

        _default_save = os.path.join(
            ROOT_DIR, "AI_Modules", "Training_Methods", "Trained_Model_Files"
        )
        self._csv_var      = tk.StringVar(value=stage.get("dataset_csv", ""))
        self._target_var   = tk.StringVar(value=stage.get("target_col",  ""))
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
        outer.rowconfigure(3, weight=1)

        # ── Dataset section ───────────────────────────────────────────
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

        self._csv_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._refresh_target_columns(),
        )
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
                padx=(0 if col == 0 else 12, 6), pady=2,
            )
            ttk.Entry(parent, textvariable=v[key], width=14).grid(
                row=row, column=col + 1, sticky="ew", pady=2, padx=(0, 4),
            )

    def _build_memd_params(self, parent: ttk.Frame) -> None:
        """Grid of entry widgets for MEMD-TCN training hyperparameters."""
        v = self._memd_vars
        params = [
            ("Sequence Length (lag):", "sequence_length", 0, 0),
            ("Epochs per TCN:",        "epochs_per_tcn",  0, 2),
            ("Batch Size:",            "batch_size",       1, 0),
            ("Learning Rate:",         "learning_rate",    1, 2),
            ("TCN Kernel Size:",       "kernel_size",      2, 0),
            ("Dilations (comma-sep):", "dilations",        2, 2),
            ("Dropout:",               "dropout",          3, 0),
            ("MEMD Directions (K):",   "K",                3, 2),
            ("Max IMFs:",              "max_imfs",         4, 0),
            ("Max Sift Iterations:",   "max_sift",         4, 2),
            ("SD Threshold:",          "sd_threshold",     5, 0),
        ]
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)
        for lbl, key, row, col in params:
            ttk.Label(parent, text=lbl).grid(
                row=row, column=col, sticky="w",
                padx=(0 if col == 0 else 12, 6), pady=2,
            )
            ttk.Entry(parent, textvariable=v[key], width=14).grid(
                row=row, column=col + 1, sticky="ew", pady=2, padx=(0, 4),
            )
        ttk.Label(
            parent,
            text=(
                "Select  Train_MEMD_TCN.py  in the 'AI Training Method' dropdown "
                "to train this model.  All parameters above are passed directly as "
                "CLI arguments to that script."
            ),
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
        if self._csv_var.get() not in csvs:
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
        if self._target_var.get() not in cols:
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
                "--sequence_length",         v["sequence_length"].get(),
                "--epochs",                  v["epochs"].get(),
                "--batch_size",              v["batch_size"].get(),
                "--learning_rate",           v["learning_rate"].get(),
                "--embedding_dim",           v["embedding_dim"].get(),
                "--output_size",             v["output_size"].get(),
                "--early_stopping_patience", v["early_stopping_patience"].get(),
                "--train_split",             v["train_split"].get(),
                "--val_split",               v["val_split"].get(),
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
            self._log(
                "No dataset CSV selected. Please choose one above.\n", tag="error"
            )
            return

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
            self._output_queue.put(None)  # sentinel

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
