import sys
import serial
import traceback
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QPushButton,
    QHBoxLayout,
    QLabel,
)
from PyQt5.QtGui import QFont
import pyqtgraph as pg

# set this to your device
SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200
TIMEOUT = 5  # seconds
MAX_PLOTS = 1  # keep last N plots to avoid unbounded memory growth

class SerialReader(QThread):
    frame_received = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, port, baud, timeout=5, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._running = True
        self.ser = None

    def run(self):
        try:
            with serial.Serial(self.port, self.baud, timeout=self.timeout) as ser:
                self.ser = ser
                while self._running:
                    try:
                        raw = ser.readline()
                        if not raw:
                            continue
                        line = raw.decode(errors="ignore").strip()
                        if not line:
                            continue
                        if line.startswith("DATA,"):
                            # allow DATA,<val1>,<val2>,...
                            parts = line.split(",", 1)[1]
                            # parts may contain extra commas, so split and convert
                            try:
                                vals = [float(x) for x in parts.split(",") if x != ""]
                                if vals:
                                    self.frame_received.emit(vals)
                            except Exception:
                                # ignore malformed frames but continue
                                continue
                        # else ignore other messages
                    except Exception:
                        # If a read error occurs, emit and break
                        self.error.emit(traceback.format_exc())
                        break
        except Exception:
            self.error.emit(traceback.format_exc())

    def stop(self):
        self._running = False
        # closing serial port is handled by context manager in run
        self.quit()
        self.wait()

class PlotWindow(QWidget):
    def __init__(self, port=SERIAL_PORT, baud=BAUD, timeout=TIMEOUT):
        super().__init__()
        self.setWindowTitle("CarDetect - Incoming Frames")
        self.resize(800, 600)

        self.reader = SerialReader(port, baud, timeout)
        self.reader.frame_received.connect(self.add_plot)
        self.reader.error.connect(self.on_error)

        main_layout = QVBoxLayout(self)

        # controls
        ctrl_layout = QHBoxLayout()
        self.status_label = QLabel("Connecting to {} @ {}".format(port, baud))
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        ctrl_layout.addWidget(self.status_label)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)
        main_layout.addLayout(ctrl_layout)

        # scrollable area for plots
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.plot_container = QWidget()
        self.vbox = QVBoxLayout(self.plot_container)
        self.vbox.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.plot_container)
        main_layout.addWidget(self.scroll)

        # connections
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

        self.plot_count = 0
        # persistent plot widgets to avoid flicker
        self.plot_widgets = []  # each item: {'frame','pw','curve','label'}

    def start(self):
        if not self.reader.isRunning():
            self.reader = SerialReader(self.reader.port, self.reader.baud, self.reader.timeout)
            self.reader.frame_received.connect(self.add_plot)
            self.reader.error.connect(self.on_error)
            self.reader.start()
            self.status_label.setText("Listening on {} @ {}".format(self.reader.port, self.reader.baud))
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

    def stop(self):
        if self.reader.isRunning():
            self.reader.stop()
        self.status_label.setText("Stopped")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def add_plot(self, samples):
        samples = np.array(samples)
        samples = (samples - 32768) / 32768.0

        # constants
        FS = 10000.0           # sample rate (Hz)
        FC = 60e9              # carrier frequency (Hz)
        C = 3e8                # speed of light (m/s)

        N = len(samples)
        if N == 0:
            return

        # window to reduce spectral leakage
        win = np.hanning(N)
        x = samples * win

        # FFT and positive-frequency half
        fft = np.fft.fft(x)
        mag = np.abs(fft)[: N // 2]
        mag[0] = 0  # remove DC
        mag[1] = 0  # remove DC

        # find peak bin (use raw magnitude for peak detection)
        peak = int(np.argmax(mag))

        # quadratic interpolation around the peak for sub-bin accuracy
        if 1 <= peak < (len(mag) - 1):
            a = mag[peak - 1]
            b = mag[peak]
            c = mag[peak + 1]
            denom = (a - 2 * b + c)
            if denom != 0:
                delta = 0.5 * (a - c) / denom
            else:
                delta = 0.0
        else:
            delta = 0.0

        peak_bin = peak + delta
        freq = peak_bin * (FS / N)   # Doppler frequency in Hz
        velocity = freq * C / (2 * FC)   # v = f_d * c / (2*fc)

        print(f"Peak bin: {peak} + {delta:.3f} => freq={freq:.2f} Hz, velocity={velocity:.3f} m/s")

        # prepare data for plotting
        freqs = np.linspace(0, FS / 2, len(mag))
        y = 20 * np.log10(mag + 1e-12)

        info = f"Peak bin: {peak} + {delta:.3f}\nfreq={freq:.2f} Hz\nvelocity={velocity:.3f} m/s"

        # If we haven't yet created MAX_PLOTS widgets, create one now and store its curve/label.
        if len(self.plot_widgets) < MAX_PLOTS:
            pw = pg.PlotWidget()
            curve = pw.plot(freqs, y, pen=pg.mkPen(color=(200, 200, 0)))
            pw.setLabel('left', 'Magnitude (dB)')
            pw.setLabel('bottom', 'Frequency (Hz)')
            pw.showGrid(x=True, y=True)
            # limit y-axis to [-60, +20] dB
            pw.setYRange(-60, 20)

            label = QLabel(info)
            font = QFont()
            font.setPointSize(18)
            font.setBold(True)
            label.setFont(font)
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)

            frame = QWidget()
            h = QHBoxLayout(frame)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(pw, 3)
            h.addWidget(label, 1)

            self.vbox.addWidget(frame)
            self.plot_widgets.append({'frame': frame, 'pw': pw, 'curve': curve, 'label': label})
            # set the title once
            pw.setTitle("Frame #{} (N={})".format(self.plot_count + 1, N))
        else:
            # reuse an existing widget in-place to avoid recreating widgets (no flicker)
            idx = self.plot_count % MAX_PLOTS
            item = self.plot_widgets[idx]
            item['curve'].setData(freqs, y)
            item['label'].setText(info)
            item['pw'].setTitle("Frame #{} (N={})".format(self.plot_count + 1, N))
            # ensure y-axis limit remains applied
            item['pw'].setYRange(-60, 20)

        self.plot_count += 1
        # (no widget deletion; we reuse widgets in-place)

    def on_error(self, err):
        self.status_label.setText("Error: see console")
        print(err)

    def closeEvent(self, event):
        try:
            if self.reader.isRunning():
                self.reader.stop()
        except Exception:
            pass
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    win = PlotWindow()
    win.show()
    # auto-start
    win.start()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()