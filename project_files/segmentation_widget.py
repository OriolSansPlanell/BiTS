# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:58:20 2024

@author: Dr Oriol Sans Planell
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

class SegmentationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(15, 10))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.ax1 = self.figure.add_subplot(231)
        self.ax2 = self.figure.add_subplot(232)
        self.ax3 = self.figure.add_subplot(233)
        self.ax4 = self.figure.add_subplot(234)
        self.ax5 = self.figure.add_subplot(235)
        self.figure.subplots_adjust(right=0.9)
        
        self.im1 = None
        self.im2 = None
        self.im3 = None
        self.im4 = None
        self.im5 = None
        self.colorbar1 = None
        self.colorbar2 = None

        # Ensure that plt.show() doesn't create a new window
        plt.ion()

    def plot_images(self, image1, image2, mask, segmented1, segmented2):
        self.figure.clear()
        self.ax1 = self.figure.add_subplot(231)
        self.ax2 = self.figure.add_subplot(232)
        self.ax3 = self.figure.add_subplot(233)
        self.ax4 = self.figure.add_subplot(234)
        self.ax5 = self.figure.add_subplot(235)

        vmin1, vmax1 = np.nanmin(image1), np.nanmax(image1)
        vmin2, vmax2 = np.nanmin(image2), np.nanmax(image2)

        self.im1 = self.ax1.imshow(image1, cmap='Spectral', vmin=vmin1, vmax=vmax1)
        self.ax1.set_title('Volume 1 (Original)')
        self.ax1.axis('off')

        self.im2 = self.ax2.imshow(image2, cmap='Spectral', vmin=vmin2, vmax=vmax2)
        self.ax2.set_title('Volume 2 (Original)')
        self.ax2.axis('off')

        self.im3 = self.ax3.imshow(mask, cmap='binary')
        self.ax3.set_title('Mask')
        self.ax3.axis('off')

        self.im4 = self.ax4.imshow(segmented1, cmap='Spectral', vmin=vmin1, vmax=vmax1)
        self.ax4.set_title('Volume 1 (Segmented)')
        self.ax4.axis('off')

        self.im5 = self.ax5.imshow(segmented2, cmap='Spectral', vmin=vmin2, vmax=vmax2)
        self.ax5.set_title('Volume 2 (Segmented)')
        self.ax5.axis('off')

        self.colorbar1 = self.figure.colorbar(self.im1, ax=self.ax1, label='Intensity')
        self.colorbar2 = self.figure.colorbar(self.im2, ax=self.ax2, label='Intensity')

        self.figure.tight_layout()
        self.canvas.draw()
        self.canvas.flush_events()  # Ensure all drawing commands are processed

    def update_images(self, image1, image2, mask, segmented1, segmented2):
        vmin1, vmax1 = np.nanmin(image1), np.nanmax(image1)
        vmin2, vmax2 = np.nanmin(image2), np.nanmax(image2)

        self.im1.set_data(image1)
        self.im1.set_clim(vmin=vmin1, vmax=vmax1)
        
        self.im2.set_data(image2)
        self.im2.set_clim(vmin=vmin2, vmax=vmax2)
        
        self.im3.set_data(mask)
        
        self.im4.set_data(segmented1)
        self.im4.set_clim(vmin=vmin1, vmax=vmax1)
        
        self.im5.set_data(segmented2)
        self.im5.set_clim(vmin=vmin2, vmax=vmax2)

        self.canvas.draw()
        self.canvas.flush_events()