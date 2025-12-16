import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QLabel
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from attractor import Simulator, Feedback, rössler_attractor
from typing import Dict


class PlotWindow(QWidget):
  edits_desc = {
    'delay': {
      'min': 0.1,
      'max': 100.,
      'step': 0.1,
      'default': 11.7,
      'row': 2,
    },
    'fb coef': {
      'min': -100,
      'max': 100.,
      'step': 0.1,
      'default': 0.2,
      'row': 2,
    },
    'sim time': {
      'min': 1,
      'max': 1000.,
      'step': 5,
      'default': 100,
      'row': 2,
    },
    'time step log': {
      'min': 1,
      'max': 5,
      'step': 0.1,
      'default': 2,
      'row': 2,
    },
    'x0': {
      'min': -10,
      'max': 10.,
      'step': 0.1,
      'default': 5,
      'row': 1,
    },
    'y0': {
      'min': -10,
      'max': 10.,
      'step': 0.1,
      'default': 3,
      'row': 1,
    },
    'z0': {
      'min': -10,
      'max': 10.,
      'step': 0.1,
      'default': -5,
      'row': 1,
    },
  }

  def __init__(self):
    super().__init__()
    self.setWindowTitle("Simple Plot Example")

    layout = QVBoxLayout(self)

    self.fig1 = Figure(figsize=(4, 3))
    self.canvas1 = FigureCanvas(self.fig1)
    layout.addWidget(self.canvas1)

    self.fig2 = Figure(figsize=(4, 3))
    self.canvas2 = FigureCanvas(self.fig2)
    layout.addWidget(self.canvas2)

    rows = [
      QHBoxLayout(),
      QHBoxLayout()
    ]
    edits : Dict[str, QDoubleSpinBox] = {}

    for name,edit in self.edits_desc.items():
      box = QDoubleSpinBox()
      box.setMaximum(edit['max'])
      box.setMinimum(edit['min'])
      box.saved_value = float(edit['default'])
      box.setValue(box.saved_value)
      box.setDecimals(3)
      box.setSingleStep(edit['step'])
      edits[name] = box
      row = edit['row'] - 1
      rows[row].addWidget(QLabel(name))
      rows[row].addWidget(box)

    layout.addLayout(rows[0])
    layout.addLayout(rows[1])
    self._edits = edits

    for edit in self._edits.values():
      edit.valueChanged.connect(self._update_plot)

    self._update_plot()

  def _restore_values(self):
    for edit in self._edits.values():
      edit.setValue(edit.saved_value)

  def _accept_values(self):
    for edit in self._edits.values():
      edit.saved_value = edit.value()

  def _update_plot(self):
    delay = self._edits['delay'].value()
    fbcoef = self._edits['fb coef'].value()
    simtime = self._edits['sim time'].value()
    x0 = self._edits['x0'].value()
    y0 = self._edits['y0'].value()
    z0 = self._edits['z0'].value()
    step_log = self._edits['time step log'].value()
    step = np.pow(10, -step_log)

    fb = Feedback(delay, fbcoef)
    initial_state = np.array([x0, y0, z0])
    sim = Simulator(rössler_attractor, fb, initial_state, step)
    res = sim.run(simtime)
    if res is None:
      print('failed to change, restoring')
      self._restore_values()
      return

    t, x, u = res
    self._accept_values()

    self.fig1.clear()
    ax = self.fig1.add_subplot(111)
    ax.plot(t, u, alpha=0.8, lw=1)
    ax.set_title("u(t)")
    ax.grid(True)
    self.canvas1.draw()

    self.fig2.clear()
    ax = self.fig2.add_subplot(111)
    ax.plot(x[:,0], x[:,1], alpha=0.8, lw=1)
    ax.set_title("x-y")
    ax.grid(True)
    self.canvas2.draw()

if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = PlotWindow()
  window.show()
  sys.exit(app.exec_())
