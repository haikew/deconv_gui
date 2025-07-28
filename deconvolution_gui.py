#!/usr/bin/env python
"""
Deconvolution GUI — Python front-end + Julia back-end
=====================================================
* PyQt5 + matplotlib are used to display a Z-max projection and let the user draw an ROI.
* Calls julia/deconv_cli.jl to perform the actual 3-D Richardson–Lucy deconvolution.
* Parses Julia output lines of the form “PROGRESS n” to update the progress bar;  
  if the exit code is non-zero, a dialog pops up showing the entire log.

Directory layout
----------------
project_root/
├── deconvolution_gui.py   ← this file
└── julia/
    ├── Project.toml
    ├── Manifest.toml
    └── deconv_cli.jl
"""

import sys, subprocess, re, textwrap
from pathlib import Path

import numpy as np
import tifffile as tiff
from PyQt5.QtCore    import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSpinBox, QCheckBox, QProgressBar,
    QMessageBox, QMainWindow
)

# ---------------- matplotlib ROI widget -----------------  
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


class ROIWidget(QWidget):
    """Widget that shows the projection image and lets the user draw a rectangular ROI."""
    roi_changed = pyqtSignal(tuple)  # (x0, x1, y0, y1)

    def __init__(self):
        super().__init__()
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax.set_axis_off()
        # Set up the rectangle selector for ROI selection
        self.selector = RectangleSelector(
            self.ax, self._on_select,
            useblit=True, button=[1], minspanx=10, minspany=10,
            interactive=True
        )
        # lay is the layout for this widget
        lay = QVBoxLayout(self)
        lay.addWidget(self.canvas)
        self.img = None

    def set_image(self, img):
        self.img = img
        # Clear and hide the axes, then display the image
        self.ax.clear()
        self.ax.set_axis_off()
        # make the image grayscale and sharp
        self.ax.imshow(img, cmap="gray", interpolation="nearest")
        # Redraw the canvas to show the new image when it is not busy
        self.canvas.draw_idle()

    def _on_select(self, e0, e1):
        x0, x1 = sorted([int(e0.xdata), int(e1.xdata)])
        y0, y1 = sorted([int(e0.ydata), int(e1.ydata)])
        # Emit the ROI coordinates only if they are large enough
        if (x1 - x0) >= 10 and (y1 - y0) >= 10:
            # use emit to notify the main window about the ROI change
            self.roi_changed.emit((x0, x1, y0, y1))


# -------------- Julia worker thread ---------------------
class JuliaWorker(QThread):
    # Send a progress update (0–100)
    progress    = pyqtSignal(int)
    finished_ok = pyqtSignal(str)
    error       = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.p = params

    def run(self):
        # Build the Julia command
        proj_dir = self.p["proj"]
        script   = proj_dir / "deconv_cli.jl"
        roi_str  = f"{self.p['x0']}:{self.p['x1']},{self.p['y0']}:{self.p['y1']}"
        # prepare the command line arguments for the Julia script
        cmd = [
            self.p["julia"], f"--project={proj_dir}", str(script),
            "--roi", roi_str,
            "--zsize", str(self.p["zsize"]),
            "--iter",  str(self.p["iter"]),
            "--gpu",   str(self.p["gpu"]).lower(),
            "--sigma_z", str(self.p["sigmaz"]),
            self.p["input"], self.p["output"],
        ]
        # open a subprocess to run the Julia script line by line
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        log, prog_re = [], re.compile(r"^PROGRESS\s+(\d+)")
        for line in proc.stdout:
            log.append(line)
            m = prog_re.match(line)
            if m:
                self.progress.emit(int(m.group(1)))     
        # Wait for the process to finish and check the exit code
        proc.wait()
        out = "".join(log)
        if proc.returncode == 0:
            self.finished_ok.emit(self.p["output"])
        else:
            self.error.emit(out)


# ------------------------ Main GUI ----------------------
class MainWin(QMainWindow):
    # main window for the deconvolution GUI
    def __init__(self, enable_gpu: bool = True):
        super().__init__()
        self.setWindowTitle("Deconvolution with ROI Selection (Julia back-end)")
        self.resize(900, 650)
        # Set the directory where the Julia project is located
        SCRIPT_DIR = Path(__file__).resolve().parent
        self.proj_dir  = SCRIPT_DIR / "julia"
        # If the executable is not in PATH, change this to the absolute path
        self.julia_exe = "julia"

        # central widget
        cw = QWidget()
        self.setCentralWidget(cw)
        vbox = QVBoxLayout(cw)

        # ROI widget
        self.roi_widget = ROIWidget()
        vbox.addWidget(self.roi_widget, 1)
        # display the default information of the loaded image
        self.lbl_info = QLabel("No file loaded")
        vbox.addWidget(self.lbl_info)

        # controls row
        row = QHBoxLayout()
        vbox.addLayout(row)

        # load all buttons and give them default values
        self.btn_load = QPushButton("Load 3-D TIFF…")
        row.addWidget(self.btn_load)
        self.btn_load.clicked.connect(self.load_stack)

        row.addWidget(QLabel("Z:"))
        self.sp_z = QSpinBox()
        self.sp_z.setRange(1, 2048)
        self.sp_z.setValue(64)
        row.addWidget(self.sp_z)

        row.addWidget(QLabel("Iter:"))
        self.sp_iter = QSpinBox()
        self.sp_iter.setRange(1, 200)
        self.sp_iter.setValue(20)
        row.addWidget(self.sp_iter)

        row.addWidget(QLabel("σz:"))
        self.sp_sig = QSpinBox()
        self.sp_sig.setRange(1, 50)
        self.sp_sig.setValue(15)
        row.addWidget(self.sp_sig)

        self.cb_gpu = QCheckBox("GPU")
        row.addWidget(self.cb_gpu)
        self.cb_gpu.setEnabled(enable_gpu)
        if not enable_gpu:
            self.cb_gpu.hide()

        self.btn_run = QPushButton("Run")
        self.btn_run.setEnabled(False)
        row.addWidget(self.btn_run)
        self.btn_run.clicked.connect(self.run_deconv)

        self.bar = QProgressBar()
        self.bar.setValue(0)
        vbox.addWidget(self.bar)

        self.btn_warm = QPushButton("Warm-up Julia")
        vbox.addWidget(self.btn_warm)
        self.btn_warm.clicked.connect(self.warmup)

        # data
        self.stack_path = None
        self.roi_xy     = None
        self.roi_widget.roi_changed.connect(self.set_roi)

    # ---------------- callbacks ----------------
    def load_stack(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF", "", "TIFF (*.tif *.tiff)"
        )
        if not f:
            return
        self.stack_path = f
        stack = tiff.imread(f)

    # preview the stack algorithm
    # Combine all Z‑planes by summing them
        proj = stack.sum(axis=0, dtype=np.float32)

        # Normalize by the brightest pixel in the summed image
        max_val = proj.max()
        proj_norm = proj / max_val if max_val else np.zeros_like(proj, dtype=np.float32)

        self.roi_widget.set_image(proj_norm)
        self.lbl_info.setText(f"Loaded {f}  shape={stack.shape}")

    # Reset the ROI and enable the run button
    def set_roi(self, roi):
        self.roi_xy = roi
        self.lbl_info.setText(f"ROI set: x{roi[0]}–{roi[1]}, y{roi[2]}–{roi[3]}")
        self.btn_run.setEnabled(True)

    # Run the Julia warm-up script to ensure all packages are installed
    def warmup(self):
        # Prepare the Julia warm-up script
        code = (
                "using Pkg; "
                "Pkg.activate(\".\"); "
                "Pkg.instantiate(); "
                'println("Julia core warm-up is done")'
        )
        # Launch Julia warm-up script
        res = subprocess.run(
            [self.julia_exe, f"--project={self.proj_dir}", "-e", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Check exit code and output to determine success
        if res.returncode == 0 and "warm-up ok" in res.stdout:
            QMessageBox.information(self, "Warm‑up", "Julia warm‑up succeeded")
        else:
            err_msg = res.stderr.strip() or res.stdout.strip() or "Unknown error"
            QMessageBox.critical(self, "Warm‑up Failed", f"Warm‑up failed:\n{err_msg}")


    def run_deconv(self):
        # Check if the stack and ROI are set
        if not (self.stack_path and self.roi_xy):
            return
        
        # Ask the user for the output file name
        out, _ = QFileDialog.getSaveFileName(
            self, "Save result as", "deconv.tif", "TIFF (*.tif)"
        )
        if not out:
            return
        # Prepare the parameters for the Julia worker
        p = dict(
            proj   = self.proj_dir,
            julia  = self.julia_exe,
            x0     = self.roi_xy[0],
            x1     = self.roi_xy[1],
            y0     = self.roi_xy[2],
            y1     = self.roi_xy[3],
            zsize  = self.sp_z.value(),
            iter   = self.sp_iter.value(),
            sigmaz = self.sp_sig.value(),
            gpu    = self.cb_gpu.isChecked(),
            input  = self.stack_path,
            output = out,
        )
        self.bar.setValue(0)
        self.btn_run.setEnabled(False)

        # Create and start the Julia worker thread
        self.th = JuliaWorker(p)
        self.th.progress.connect(self.bar.setValue)
        self.th.finished_ok.connect(self.finished)
        self.th.error.connect(self.err)
        self.th.start()

    # send a message to the user when the deconvolution is finished
    def finished(self, path):
        QMessageBox.information(self, "Done", f"Saved: {path}")
        self.bar.setValue(100)
        self.btn_run.setEnabled(True)

    # Show an error message if the Julia script fails
    def err(self, log):
        QMessageBox.critical(self, "Julia error", textwrap.shorten(log, 8000))
        self.btn_run.setEnabled(True)


# ---------------- main -----------------------
def main(argv=None):
    """Entry point for the GUI."""
    import argparse

    # Parse command-line arguments
    # on Raspberry Pi, we hide the GPU option with --nogpu
    parser = argparse.ArgumentParser(description="3-D Deconvolution GUI")
    parser.add_argument(
        "--nogpu",
        action="store_true",
        help="Hide GPU option (use on Raspberry Pi)",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    w = MainWin(enable_gpu=not args.nogpu)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
