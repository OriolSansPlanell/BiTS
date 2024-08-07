# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:42:56 2024

@author: Dr Oriol Sans Planell
"""
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter
from qtrangeslider import QRangeSlider

class HistogramWidget(QWidget):
    selection_changed = pyqtSignal(object, str)
    rectangle_updated = pyqtSignal(tuple)
    histogram_updated = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create the figure and canvas immediately
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Create range slider
        self.range_slider = QRangeSlider(Qt.Horizontal)
        self.range_slider.setRange(0, 100)
        self.range_slider.setValue((0, 100))
        self.range_slider.valueChanged.connect(self.update_scale)
        
        # Create scale toggle button
        self.scale_toggle = QPushButton("Log Scale")
        self.scale_toggle.setCheckable(True)
        self.scale_toggle.setChecked(True)
        self.scale_toggle.clicked.connect(self.toggle_scale)
        
        # Create labels for min and max values
        self.min_label = QLabel("Min: 0")
        self.max_label = QLabel("Max: 100")
        
        # Layout
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)
        
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.min_label)
        control_layout.addWidget(self.range_slider)
        control_layout.addWidget(self.max_label)
        control_layout.addWidget(self.scale_toggle)
        
        layout = QVBoxLayout()
        layout.addLayout(plot_layout)
        layout.addLayout(control_layout)
        self.setLayout(layout)
        
        self.selector = None
        self.im = None
        self.colorbar = None
        self.xedges = None
        self.yedges = None
        self.hist = None
        self.log_scale = True
        self.selection_type = 'polygon'
        self.polygon_selected = None
        self.rectangle_patch = None
        self.drag_start = None

    def plot_histogram(self, hist, xedges, yedges):
        self.hist = hist
        self.xedges = xedges
        self.yedges = yedges
        
        # Set range slider based on histogram data
        self.hist_min = max(1, np.min(self.hist[self.hist > 0]))  # Minimum non-zero value
        self.hist_max = np.max(self.hist)
        self.range_slider.setRange(0, 100)
        self.range_slider.setValue((0, 100))
        
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        
        if self.hist is None:
            return
        
        vmin, vmax = self.range_slider.value()
        if self.log_scale:
            vmin = np.exp(np.log(self.hist_min) + (vmin / 100) * (np.log(self.hist_max) - np.log(self.hist_min)))
            vmax = np.exp(np.log(self.hist_min) + (vmax / 100) * (np.log(self.hist_max) - np.log(self.hist_min)))
            norm = LogNorm(vmin=max(1, vmin), vmax=vmax)  # Ensure vmin is at least 1 for log scale
        else:
            vmin = self.hist_min + (vmin / 100) * (self.hist_max - self.hist_min)
            vmax = self.hist_min + (vmax / 100) * (self.hist_max - self.hist_min)
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        self.im = self.ax.imshow(self.hist.T, norm=norm, cmap='viridis', aspect='auto',
                                 extent=[self.xedges[0], self.xedges[-1],
                                         self.yedges[0], self.yedges[-1]],
                                 origin='lower')
        
        self.ax.set_xlabel('Volume 1 intensity')
        self.ax.set_ylabel('Volume 2 intensity')
        
        if self.colorbar:
            self.colorbar.remove()
        if self.log_scale:
            self.colorbar = self.figure.colorbar(self.im, ax=self.ax, label='Counts (log scale)')
        else:
            self.colorbar = self.figure.colorbar(self.im, ax=self.ax, label='Counts')
        
        self.ax.set_title('2D Histogram of Registered Tomography Volumes')
        
        self.setup_selector()
        
        self.canvas.draw()
        self.histogram_updated.emit()

    def update_scale(self):
        vmin, vmax = self.range_slider.value()
        if self.log_scale:
            vmin = np.exp(np.log(self.hist_min) + (vmin / 100) * (np.log(self.hist_max) - np.log(self.hist_min)))
            vmax = np.exp(np.log(self.hist_min) + (vmax / 100) * (np.log(self.hist_max) - np.log(self.hist_min)))
        else:
            vmin = self.hist_min + (vmin / 100) * (self.hist_max - self.hist_min)
            vmax = self.hist_min + (vmax / 100) * (self.hist_max - self.hist_min)
        self.min_label.setText(f"Min: {vmin:.2e}")
        self.max_label.setText(f"Max: {vmax:.2e}")
        self.update_plot()

    def toggle_scale(self):
        self.log_scale = self.scale_toggle.isChecked()
        if self.log_scale:
            self.scale_toggle.setText("Log Scale")
        else:
            self.scale_toggle.setText("Linear Scale")
        self.update_plot()

    def setup_selector(self):
        if self.selector:
            self.selector.set_visible(False)
        
        if self.selection_type == 'polygon':
            self.selector = PolygonSelector(self.ax, self.onselect_polygon)
        else:  # rectangle
            self.selector = None
            self.canvas.mpl_connect('button_press_event', self.onpress_rectangle)
            self.canvas.mpl_connect('button_release_event', self.onrelease_rectangle)
            self.canvas.mpl_connect('motion_notify_event', self.onmove_rectangle)

    def onselect_polygon(self, vertices):
        self.polygon_selected = Path(vertices)
        self.selection_changed.emit(vertices, 'polygon')
        
    def onpress_rectangle(self, event):
        if event.inaxes != self.ax:
            return
        if self.rectangle_patch:
            contains, _ = self.rectangle_patch.contains(event)
            if contains:
                self.drag_start = (event.xdata, event.ydata)
            else:
                self.drag_start = None
                self.press = (event.xdata, event.ydata)
        else:
            self.press = (event.xdata, event.ydata)

    def onmove_rectangle(self, event):
        if event.inaxes != self.ax:
            return
        if self.drag_start:
            # Move existing rectangle
            dx = event.xdata - self.drag_start[0]
            dy = event.ydata - self.drag_start[1]
            x0, y0 = self.rectangle_patch.get_xy()
            width = self.rectangle_patch.get_width()
            height = self.rectangle_patch.get_height()
            new_x0 = x0 + dx
            new_y0 = y0 + dy
            self.update_rectangle((new_x0, new_x0 + width, new_y0, new_y0 + height))
            self.drag_start = (event.xdata, event.ydata)
        elif self.press:
            # Draw new rectangle
            x0, y0 = self.press
            x1, y1 = event.xdata, event.ydata
            self.update_rectangle((min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)))

    def onrelease_rectangle(self, event):
        if event.inaxes != self.ax:
            return
        self.press = None
        self.drag_start = None
        if self.rectangle_patch:
            x0, y0 = self.rectangle_patch.get_xy()
            x1 = x0 + self.rectangle_patch.get_width()
            y1 = y0 + self.rectangle_patch.get_height()
            vertices = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            if self.log_scale:
                vertices = np.power(10, vertices)
            self.polygon_selected = Path(vertices)
            self.selection_changed.emit(vertices, 'rectangle')
            self.rectangle_updated.emit((x0, x1, y0, y1))

    def onselect_rectangle(self, eclick, erelease):
        extents = self.selector.extents
        vertices = [(extents[0], extents[2]), (extents[1], extents[2]),
                    (extents[1], extents[3]), (extents[0], extents[3])]
        if self.log_scale:
            vertices = np.power(10, vertices)
        self.polygon_selected = Path(vertices)
        self.selection_changed.emit(vertices, 'rectangle')
        self.rectangle_updated.emit(extents)

    def update_rectangle(self, extents):
        x0, x1, y0, y1 = extents
        if self.rectangle_patch:
            self.rectangle_patch.remove()
        self.rectangle_patch = Rectangle((x0, y0), x1-x0, y1-y0,
                                         fill=False, edgecolor='red', linewidth=2)
        self.ax.add_patch(self.rectangle_patch)
        self.canvas.draw()

    def rectangle_changed(self, event):
        if self.selection_type == 'rectangle':
            extents = self.selector.extents
            self.rectangle_updated.emit(extents)

    def set_selection_type(self, selection_type):
        self.selection_type = selection_type
        self.setup_selector()

    def reset_selection(self):
        if self.selector:
            self.selector.set_visible(False)
        self.polygon_selected = None
        self.canvas.draw()

#    def toggle_scale(self):
#        self.log_scale = not self.log_scale
#        self.plot_histogram(self.hist, self.xedges, self.yedges, self.log_scale)

