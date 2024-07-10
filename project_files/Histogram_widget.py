# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:42:56 2024

@author: Dr Oriol Sans Planell
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Rectangle
from matplotlib.path import Path
import numpy as np

class HistogramWidget(QWidget):
    selection_changed = pyqtSignal(object, str)  # Emit vertices and selection type
    rectangle_updated = pyqtSignal(tuple)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.ax = self.figure.add_subplot(111)
        self.selector = None
        self.im = None
        self.colorbar = None
        self.xedges = None
        self.yedges = None
        self.hist = None
        self.log_scale = False
        self.selection_type = 'polygon'
        self.polygon_selected = None
        self.rectangle_patch = None
        self.drag_start = None

    def plot_histogram(self, hist, xedges, yedges, log_scale=False):
        self.hist = hist
        self.xedges = xedges
        self.yedges = yedges
        self.log_scale = log_scale

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        if log_scale:
            xedges_plot = np.log10(xedges + 1)
            yedges_plot = np.log10(yedges + 1)
        else:
            xedges_plot = xedges
            yedges_plot = yedges

        self.im = self.ax.imshow(np.log(hist.T + 1), cmap='RdBu', aspect='auto',
                                 extent=[xedges_plot[0], xedges_plot[-1], yedges_plot[0], yedges_plot[-1]],
                                 origin='lower')
        
        self.colorbar = self.figure.colorbar(self.im, ax=self.ax, label='Log count')
        
        if log_scale:
            self.ax.set_xlabel('Volume 1 intensity (log scale)')
            self.ax.set_ylabel('Volume 2 intensity (log scale)')
            self.ax.xaxis.set_major_formatter(lambda x, pos: f"{10**x:.0f}")
            self.ax.yaxis.set_major_formatter(lambda x, pos: f"{10**x:.0f}")
        else:
            self.ax.set_xlabel('Volume 1 intensity')
            self.ax.set_ylabel('Volume 2 intensity')
        
        self.ax.set_title('2D Histogram of Registered Tomography Volumes')
        
        self.setup_selector()
        
        
        if self.rectangle_patch:
            self.rectangle_patch.remove()
            self.rectangle_patch = None
        
        self.canvas.draw()

    def setup_selector(self):
        if self.selector:
            self.selector.set_visible(False)
        
        if self.selection_type == 'polygon':
            self.selector = PolygonSelector(self.ax, self.onselect_polygon)
            if self.rectangle_patch:
                self.rectangle_patch.remove()
                self.rectangle_patch = None
        else:  # rectangle
            self.selector = None
            self.canvas.mpl_connect('button_press_event', self.onpress_rectangle)
            self.canvas.mpl_connect('button_release_event', self.onrelease_rectangle)
            self.canvas.mpl_connect('motion_notify_event', self.onmove_rectangle)
        
        self.canvas.draw()

    def onselect_polygon(self, vertices):
        if self.log_scale:
            vertices = np.power(10, vertices)
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

    def toggle_scale(self):
        self.log_scale = not self.log_scale
        self.plot_histogram(self.hist, self.xedges, self.yedges, self.log_scale)

