# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:58:20 2024

@author: Dr Oriol Sans Planell
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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

    def plot_images(self, image1, image2, mask, segmented1, segmented2):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()

        self.im1 = self.ax1.imshow(image1, cmap='Spectral')
        self.ax1.set_title('Volume 1 (Original)')
        self.ax1.axis('off')

        self.im2 = self.ax2.imshow(image2, cmap='Spectral')
        self.ax2.set_title('Volume 2 (Original)')
        self.ax2.axis('off')

        self.im3 = self.ax3.imshow(mask, cmap='binary')
        self.ax3.set_title('Mask')
        self.ax3.axis('off')

        self.im4 = self.ax4.imshow(segmented1, cmap='Spectral')
        self.ax4.set_title('Volume 1 (Segmented)')
        self.ax4.axis('off')

        self.im5 = self.ax5.imshow(segmented2, cmap='Spectral')
        self.ax5.set_title('Volume 2 (Segmented)')
        self.ax5.axis('off')

        if self.colorbar1 is None:
            self.colorbar1 = self.figure.colorbar(self.im1, ax=self.ax1, label='Intensity')
        else:
            self.colorbar1.update_normal(self.im1)

        if self.colorbar2 is None:
            self.colorbar2 = self.figure.colorbar(self.im2, ax=self.ax2, label='Intensity')
        else:
            self.colorbar2.update_normal(self.im2)

        self.figure.tight_layout()
        self.canvas.draw()