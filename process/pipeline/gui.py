"""
GUI layer (Tkinter): file-picker dialog for the .mp4 + a window showing progress logs.
This module contains NO pipeline logic - it receives a `run_pipeline` callback
from the outside (dependency injection), keeping the UI fully decoupled from processing.
"""
import os
import sys
import queue
import threading
import subprocess
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class App:
    def __init__(self, root, run_pipeline, initial_path=None):
        self.root = root
        self.run_pipeline = run_pipeline
        self.root.title("Video -> Text (FFmpeg + Whisper)")
        self.root.geometry("640x420")
        self.log_queue = queue.Queue()

        top = tk.Frame(root, padx=10, pady=10)
        top.pack(fill=tk.X)
        self.choose_btn = tk.Button(top, text="Choose MP4 video...", command=self.choose_file)
        self.choose_btn.pack(side=tk.LEFT)
        self.path_label = tk.Label(top, text="(no file selected)", anchor="w")
        self.path_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.log_text = tk.Text(root, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.root.after(100, self._poll_log_queue)

        if initial_path:
            self.root.after(200, lambda: self.start_processing(initial_path))

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Choose an MP4 video file",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
        )
        if path:
            self.start_processing(path)

    def start_processing(self, path):
        self.path_label.config(text=path)
        self.choose_btn.config(state=tk.DISABLED)
        self._append_log(f"Selected: {path}")
        self.progress.start(12)
        thread = threading.Thread(target=self._worker, args=(path,), daemon=True)
        thread.start()

    def _worker(self, path):
        try:
            output_path = self.run_pipeline(path, self._log_from_thread)
            self.log_queue.put(("done", output_path))
        except Exception as exc:
            self.log_queue.put(("error", f"{exc}\n\n{traceback.format_exc()}"))

    def _log_from_thread(self, message: str):
        self.log_queue.put(("log", message))

    def _append_log(self, message: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _poll_log_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "done":
                    self.progress.stop()
                    self._append_log("")
                    self._append_log("=== DONE ===")
                    self.choose_btn.config(state=tk.NORMAL)
                    self._offer_open_folder(payload)
                elif kind == "error":
                    self.progress.stop()
                    self._append_log("=== ERROR ===")
                    self._append_log(payload)
                    self.choose_btn.config(state=tk.NORMAL)
                    messagebox.showerror("Error", "Something went wrong - see the log for details.")
        except queue.Empty:
            pass
        self.root.after(150, self._poll_log_queue)

    def _offer_open_folder(self, output_path):
        if messagebox.askyesno(
            "Done",
            f"Created file:\n{output_path}\n\nOpen the folder containing this file?",
        ):
            folder = os.path.dirname(output_path)
            try:
                if sys.platform.startswith("win"):
                    os.startfile(folder)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.run(["open", folder])
                else:
                    subprocess.run(["xdg-open", folder])
            except Exception:
                pass
