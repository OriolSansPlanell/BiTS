# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:01:58 2024

@author: Dr Oriol Sans Planell
"""

import os
import numpy as np
import tifffile
import torch
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QSlider, QLabel, QTabWidget, QComboBox,
                             QStatusBar, QProgressBar, QAction, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
import json
from matplotlib.path import Path
import matplotlib.pyplot as plt
from Histogram_widget import HistogramWidget
from segmentation_widget import SegmentationWidget
from matplotlib.widgets import RectangleSelector, PolygonSelector

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BiTS: Bivariate Tomography Segmentation")
        self.setGeometry(100, 100, 1600, 1000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.selection_type = "polygon"
        self.setup_ui()
        self.setup_connections()
        self.setup_shortcuts()
        self.setup_menus()

        self.volume1 = None
        self.volume2 = None
        self.current_plane = "XY"
        self.polygon = None
        self.histogram_widget.selection_changed.connect(self.update_segmentation)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_layout = self.create_left_layout()
        right_layout = self.create_right_layout()

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        self.progress_bar = QProgressBar()
        self.statusBar.addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()
        
        self.toggle_scale_button = QPushButton("Toggle Log Scale")
        left_layout.addWidget(self.toggle_scale_button)
        
        # Add selection type toggle button
        self.selection_toggle_button = QPushButton("Switch to Rectangle Selection")
        left_layout.addWidget(self.selection_toggle_button)

        # Add sliders for rectangular selection
        self.slider_widget = QWidget()
        slider_layout = QVBoxLayout(self.slider_widget)
        self.min_x_slider = QSlider(Qt.Horizontal)
        self.max_x_slider = QSlider(Qt.Horizontal)
        self.min_y_slider = QSlider(Qt.Horizontal)
        self.max_y_slider = QSlider(Qt.Horizontal)
        
        for slider in [self.min_x_slider, self.max_x_slider, self.min_y_slider, self.max_y_slider]:
            slider_layout.addWidget(slider)
            slider.setRange(0, 100)  # We'll update this range when we load the data
            slider.valueChanged.connect(self.update_rectangle_selection)

        left_layout.addWidget(self.slider_widget)
        self.slider_widget.hide()  # Initially hide the sliders

    def create_left_layout(self):
        layout = QVBoxLayout()

        load_buttons_layout = QHBoxLayout()
        self.load_button1 = QPushButton("Load Volume 1")
        self.load_button2 = QPushButton("Load Volume 2")
        load_buttons_layout.addWidget(self.load_button1)
        load_buttons_layout.addWidget(self.load_button2)
        layout.addLayout(load_buttons_layout)

        self.new_segmentation_button = QPushButton("New Segmentation")
        layout.addWidget(self.new_segmentation_button)

        self.histogram_widget = HistogramWidget(self)
        layout.addWidget(self.histogram_widget)

        self.reset_selection_button = QPushButton("Reset Selection")
        layout.addWidget(self.reset_selection_button)

        self.save_button = QPushButton("Save Images")
        layout.addWidget(self.save_button)

        self.reset_button = QPushButton("Reset Settings")
        layout.addWidget(self.reset_button)

        selection_instructions = QLabel("Draw a polygon on the histogram to select the region of interest.")
        layout.addWidget(selection_instructions)

        return layout

    def create_right_layout(self):
        layout = QVBoxLayout()

        self.tab_widget = QTabWidget()
        self.segmentation_widget = SegmentationWidget()
        self.tab_widget.addTab(self.segmentation_widget, "Segmentation")

        plane_selection_widget = self.create_plane_selection_widget()
        self.tab_widget.addTab(plane_selection_widget, "Plane Selection")

        layout.addWidget(self.tab_widget)

        self.slice_slider = QSlider(Qt.Horizontal)
        layout.addWidget(QLabel("Slice"))
        layout.addWidget(self.slice_slider)

        return layout

    def create_plane_selection_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["XY", "XZ", "YZ"])
        layout.addWidget(QLabel("Select Plane:"))
        layout.addWidget(self.plane_combo)
        return widget

    def setup_connections(self):
        self.load_button1.clicked.connect(lambda: self.load_volume(1))
        self.load_button2.clicked.connect(lambda: self.load_volume(2))
        self.new_segmentation_button.clicked.connect(self.start_new_segmentation)
        self.histogram_widget.selection_changed.connect(self.update_selection)  # Changed from polygon_selected to selection_changed
        self.reset_selection_button.clicked.connect(self.reset_polygon_selection)
        self.save_button.clicked.connect(self.save_images)
        self.reset_button.clicked.connect(self.reset_settings)
        self.plane_combo.currentTextChanged.connect(self.change_plane)
        self.slice_slider.valueChanged.connect(self.update_segmentation)
        self.histogram_widget.selection_changed.connect(self.update_selection)
        self.histogram_widget.rectangle_updated.connect(self.update_rectangle_sliders)
        self.selection_toggle_button.clicked.connect(self.toggle_selection_type)
        self.min_x_slider.valueChanged.connect(self.update_rectangle_from_sliders)
        self.max_x_slider.valueChanged.connect(self.update_rectangle_from_sliders)
        self.min_y_slider.valueChanged.connect(self.update_rectangle_from_sliders)
        self.max_y_slider.valueChanged.connect(self.update_rectangle_from_sliders)
    
    def toggle_histogram_scale(self):
        if hasattr(self, 'histogram_widget'):
            self.histogram_widget.toggle_scale()
            self.update_segmentation()  # Update segmentation to reflect the new scale
    
    def toggle_selection_type(self):
        self.selection_type = 'rectangle' if self.selection_type == 'polygon' else 'polygon'
        self.histogram_widget.set_selection_type(self.selection_type)
        self.slider_widget.setVisible(self.selection_type == 'rectangle')
        self.selection_toggle_button.setText(f"Switch to {'Polygon' if self.selection_type == 'rectangle' else 'Rectangle'} Selection")
        
        # Reset the selection when switching types
        self.histogram_widget.reset_selection()
        self.polygon = None
        self.update_segmentation()
        
    def update_selection(self, vertices, selection_type):
        self.polygon = Path(vertices)
        if selection_type == 'rectangle':
            self.update_rectangle_sliders(self.histogram_widget.selector.extents)
        self.update_segmentation()

    def update_rectangle_selection(self):
        if self.selection_type == 'rectangle':
            x_range = self.histogram_widget.xedges[-1] - self.histogram_widget.xedges[0]
            y_range = self.histogram_widget.yedges[-1] - self.histogram_widget.yedges[0]
            
            x_min = self.histogram_widget.xedges[0] + x_range * self.min_x_slider.value() / 100
            x_max = self.histogram_widget.xedges[0] + x_range * self.max_x_slider.value() / 100
            y_min = self.histogram_widget.yedges[0] + y_range * self.min_y_slider.value() / 100
            y_max = self.histogram_widget.yedges[0] + y_range * self.max_y_slider.value() / 100
            
            self.histogram_widget.selector.extents = (x_min, x_max, y_min, y_max)
            self.histogram_widget.canvas.draw()
            
            vertices = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            self.polygon = Path(vertices)
            self.update_segmentation()
        
    def update_polygon_selection(self, vertices):
        print("update_polygon_selection called in MainWindow")
        self.polygon = Path(vertices)
        print(f"Polygon set in MainWindow with {len(vertices)} vertices")
        self.update_segmentation()
    
    def update_rectangle_sliders(self, extents):
        x0, x1, y0, y1 = extents
        x_range = self.histogram_widget.xedges[-1] - self.histogram_widget.xedges[0]
        y_range = self.histogram_widget.yedges[-1] - self.histogram_widget.yedges[0]
        
        self.min_x_slider.setValue(int((x0 - self.histogram_widget.xedges[0]) / x_range * 100))
        self.max_x_slider.setValue(int((x1 - self.histogram_widget.xedges[0]) / x_range * 100))
        self.min_y_slider.setValue(int((y0 - self.histogram_widget.yedges[0]) / y_range * 100))
        self.max_y_slider.setValue(int((y1 - self.histogram_widget.yedges[0]) / y_range * 100))

    def update_rectangle_from_sliders(self):
        if self.histogram_widget.selection_type == 'rectangle':
            x_range = self.histogram_widget.xedges[-1] - self.histogram_widget.xedges[0]
            y_range = self.histogram_widget.yedges[-1] - self.histogram_widget.yedges[0]
            
            x0 = self.histogram_widget.xedges[0] + (self.min_x_slider.value() / 100) * x_range
            x1 = self.histogram_widget.xedges[0] + (self.max_x_slider.value() / 100) * x_range
            y0 = self.histogram_widget.yedges[0] + (self.min_y_slider.value() / 100) * y_range
            y1 = self.histogram_widget.yedges[0] + (self.max_y_slider.value() / 100) * y_range
            
            extents = (x0, x1, y0, y1)
            self.histogram_widget.update_rectangle(extents)
            self.update_segmentation()


    def reset_polygon_selection(self):
        self.histogram_widget.reset_selection()
        self.polygon = None
        self.update_segmentation()
            
    def scale_polygon_to_image(self, vertices, image_shape):
        # Get the extent of the histogram
        hist_extent = self.histogram_widget.im.get_extent()
        
        # Scale x and y coordinates
        scaled_x = np.interp(vertices[:, 0], [hist_extent[0], hist_extent[1]], [0, image_shape[1] - 1])
        scaled_y = np.interp(vertices[:, 1], [hist_extent[2], hist_extent[3]], [0, image_shape[0] - 1])
        
        return np.column_stack((scaled_x, scaled_y))
    
    def start_new_segmentation(self):
        print("Starting new segmentation")
        if self.histogram_widget.selector:
            self.histogram_widget.selector.set_visible(False)
        self.histogram_widget.setup_selector()
        self.polygon = None  # Reset the polygon in MainWindow
        print("Polygon reset in MainWindow")
        self.statusBar.showMessage("Draw a new selection on the histogram", 3000)

    def setup_tooltips(self):
        self.load_button1.setToolTip("Load the first volume (Ctrl+1)")
        self.load_button2.setToolTip("Load the second volume (Ctrl+2)")
        self.save_button.setToolTip("Save segmented images (Ctrl+S)")
        self.reset_button.setToolTip("Reset all settings to default values")
        self.plane_combo.setToolTip("Select the plane for visualization")

    def setup_help_menu(self):
        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        instructions_action = QAction("Instructions", self)
        instructions_action.triggered.connect(self.show_instructions)
        help_menu.addAction(instructions_action)

    def setup_settings_menu(self):
        menu_bar = self.menuBar()
        settings_menu = menu_bar.addMenu("Settings")

        save_settings_action = QAction("Save Settings", self)
        save_settings_action.triggered.connect(self.save_settings)
        settings_menu.addAction(save_settings_action)

        load_settings_action = QAction("Load Settings", self)
        load_settings_action.triggered.connect(self.load_settings)
        settings_menu.addAction(load_settings_action)

    def reset_settings(self):
        self.reset_polygon_selection()
        self.plane_combo.setCurrentIndex(0)
        self.slice_slider.setValue(0)
        self.statusBar.showMessage("Settings reset to default values", 3000)

    def setup_menus(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")
        
        load1_action = QAction("Load Volume 1", self)
        load1_action.triggered.connect(lambda: self.load_volume(1))
        file_menu.addAction(load1_action)

        load2_action = QAction("Load Volume 2", self)
        load2_action.triggered.connect(lambda: self.load_volume(2))
        file_menu.addAction(load2_action)

        save_action = QAction("Save Images", self)
        save_action.triggered.connect(self.save_images)
        file_menu.addAction(save_action)

        # Settings menu
        settings_menu = menu_bar.addMenu("Settings")

        save_settings_action = QAction("Save Settings", self)
        save_settings_action.triggered.connect(self.save_settings)
        settings_menu.addAction(save_settings_action)

        load_settings_action = QAction("Load Settings", self)
        load_settings_action.triggered.connect(self.load_settings)
        settings_menu.addAction(load_settings_action)

        # Help menu
        help_menu = menu_bar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        instructions_action = QAction("Instructions", self)
        instructions_action.triggered.connect(self.show_instructions)
        help_menu.addAction(instructions_action)

    def setup_shortcuts(self):
        load1_shortcut = QAction("Load Volume 1", self)
        load1_shortcut.setShortcut(QKeySequence("Ctrl+1"))
        load1_shortcut.triggered.connect(lambda: self.load_volume(1))
        self.addAction(load1_shortcut)

        load2_shortcut = QAction("Load Volume 2", self)
        load2_shortcut.setShortcut(QKeySequence("Ctrl+2"))
        load2_shortcut.triggered.connect(lambda: self.load_volume(2))
        self.addAction(load2_shortcut)

        save_shortcut = QAction("Save Images", self)
        save_shortcut.setShortcut(QKeySequence("Ctrl+S"))
        save_shortcut.triggered.connect(self.save_images)
        self.addAction(save_shortcut)

    def show_about_dialog(self):
        QMessageBox.about(self, "About", "BiTS (Bivariate Tomography Segmentation)\nVersion 1.0\nCreated by Dr Oriol Sans-Planell")

    def show_instructions(self):
        instructions = """
        1. Load Volume 1 and Volume 2 using the respective buttons or Ctrl+1 and Ctrl+2 shortcuts.
        2. Once the histogram is displayed, draw a polygon on it to select the region of interest.
        3. Use the plane selection tab to change the visualization plane.
        4. Use the slice slider to navigate through the volume.
        5. Click 'Save Images' or use Ctrl+S to save the segmented images.
        6. Use 'Reset Selection' to clear the polygon selection.
        7. Save and load your settings using the Settings menu.
        """
        QMessageBox.information(self, "Instructions", instructions)

    def save_settings(self):
        settings = {
            "plane": self.plane_combo.currentText(),
            "slice": self.slice_slider.value(),
            "polygon": self.polygon.vertices.tolist() if self.polygon else None
        }
        filename, _ = QFileDialog.getSaveFileName(self, "Save Settings", "Sample_selection.json", "JSON Files (*.json)")
        if filename:
            with open(filename, 'w') as f:
                json.dump(settings, f)
            self.statusBar.showMessage(f"Settings saved to {filename}", 3000)

    def load_settings(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Settings", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'r') as f:
                settings = json.load(f)
            
            self.plane_combo.setCurrentText(settings["plane"])
            self.slice_slider.setValue(settings["slice"])
            
            if settings["polygon"]:
                vertices = np.array(settings["polygon"])
                self.polygon = Path(vertices)
                
                # Redraw the polygon or rectangle on the histogram
                if len(vertices) == 4:  # Rectangle
                    self.histogram_widget.selection_type = 'rectangle'
                    x0, y0 = vertices.min(axis=0)
                    x1, y1 = vertices.max(axis=0)
                    self.histogram_widget.selector = RectangleSelector(
                        self.histogram_widget.ax,
                        self.histogram_widget.onselect_rectangle,
                        useblit=True,
                        button=[1],
                        minspanx=5, minspany=5,
                        spancoords='pixels',
                        interactive=True
                    )
                    self.histogram_widget.selector.extents = (x0, x1, y0, y1)
                else:  # Polygon
                    self.histogram_widget.selection_type = 'polygon'
                    self.histogram_widget.selector = PolygonSelector(
                        self.histogram_widget.ax,
                        self.histogram_widget.onselect_polygon
                    )
                    self.histogram_widget.selector.verts = vertices.tolist()
    
                self.histogram_widget.canvas.draw()
            
            self.update_segmentation()
            self.statusBar.showMessage(f"Settings loaded from {filename}", 3000)

    
    def load_volume(self, volume_number):
        folder = QFileDialog.getExistingDirectory(self, f"Select folder for Volume {volume_number}")
        if folder:
            self.statusBar.showMessage(f"Loading Volume {volume_number}...")
            self.progress_bar.show()
            self.progress_bar.setRange(0, 0)  # Indeterminate progress

            # Use QTimer to allow GUI to update
            QTimer.singleShot(100, lambda: self._load_volume_worker(folder, volume_number))

    def _load_volume_worker(self, folder, volume_number):
        tiff_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tiff') or f.endswith('.tif')])
        volume = torch.stack([torch.from_numpy(tifffile.imread(f).astype(np.float32)) for f in tiff_files]).to(self.device)
        if volume_number == 1:
            self.volume1 = volume
        else:
            self.volume2 = volume

        self.progress_bar.hide()
        self.statusBar.showMessage(f"Volume {volume_number} loaded successfully", 3000)

        if self.volume1 is not None and self.volume2 is not None:
            self.calculate_histogram()

    def save_images(self):
        if self.volume1 is None or self.volume2 is None:
            self.statusBar.showMessage("Please load both volumes before saving", 3000)
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Images")
        if not save_dir:
            return

        self.statusBar.showMessage("Saving images...")
        self.progress_bar.show()
        self.progress_bar.setRange(0, self.volume1.shape[0])

        # Use QTimer to allow GUI to update
        QTimer.singleShot(100, lambda: self._save_images_worker(save_dir))
        
    def _save_images_worker(self, save_dir):
        subdirs = ['segmented_volume1', 'segmented_volume2', 'mask_volume1', 'mask_volume2']
        for subdir in subdirs:
            os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)
        
        hist_extent = self.histogram_widget.im.get_extent()
        print(f"Histogram extent: {hist_extent}")
        
        for i in range(self.volume1.shape[0]):
            image1 = self.volume1[i].cpu().numpy()
            image2 = self.volume2[i].cpu().numpy()
            
            print(f"\nSlice {i}:")
            print(f"Image1 range: {np.min(image1)} to {np.max(image1)}")
            print(f"Image2 range: {np.min(image2)} to {np.max(image2)}")
            
            if self.polygon is not None:
                print(f"Polygon vertices: {self.polygon.vertices}")
                
                # Normalize polygon vertices
                x_min, x_max = np.min(image1), np.max(image1)
                y_min, y_max = np.min(image2), np.max(image2)
                if self.histogram_widget.log_scale:
                    x_min, x_max = np.log10(x_min), np.log10(x_max)
                    y_min, y_max = np.log10(y_min), np.log10(y_max)
                    normalized_vertices = [(np.log10(x) - x_min) / (x_max - x_min) for x in self.polygon.vertices[:, 0]], \
                                          [(np.log10(y) - y_min) / (y_max - y_min) for y in self.polygon.vertices[:, 1]]
                else:
                    normalized_vertices = [(x - x_min) / (x_max - x_min) for x in self.polygon.vertices[:, 0]], \
                                          [(y - y_min) / (y_max - y_min) for y in self.polygon.vertices[:, 1]]
                normalized_polygon = Path(list(zip(*normalized_vertices)))
                
                print("Normalized polygon vertices:")
                print(normalized_polygon.vertices)
    
                # Create a mask based on the histogram values
                mask = np.zeros(image1.shape, dtype=bool)
                points_checked = 0
                points_inside = 0
                for ii in range(image1.shape[0]):
                    for jj in range(image1.shape[1]):
                        val1 = image1[ii, jj]
                        val2 = image2[ii, jj]
                        
                        if self.histogram_widget.log_scale:
                            point = [(np.log10(val1) - x_min) / (x_max - x_min),
                                     (np.log10(val2) - y_min) / (y_max - y_min)]
                        else:
                            point = [(val1 - x_min) / (x_max - x_min),
                                     (val2 - y_min) / (y_max - y_min)]
                        
                        points_checked += 1
                        if normalized_polygon.contains_point(point):
                            mask[ii, jj] = True
                            points_inside += 1
    
                print(f"Number of points in mask: {np.sum(mask)}")
                print(f"Percentage of points in mask: {np.sum(mask) / mask.size * 100:.2f}%")
            else:
                mask = np.ones_like(image1, dtype=bool)
                print("Using full mask")
    
            segmented1 = np.where(mask, image1, 0)
            segmented2 = np.where(mask, image2, 0)
    
            print(f"Segmented1 range: {np.min(segmented1)} to {np.max(segmented1)}")
            print(f"Segmented2 range: {np.min(segmented2)} to {np.max(segmented2)}")
    
            tifffile.imwrite(os.path.join(save_dir, 'segmented_volume1', f'slice_{i:04d}.tiff'), segmented1)
            tifffile.imwrite(os.path.join(save_dir, 'segmented_volume2', f'slice_{i:04d}.tiff'), segmented2)
            tifffile.imwrite(os.path.join(save_dir, 'mask_volume1', f'slice_{i:04d}.tiff'), mask.astype(np.uint8) * 255)
            tifffile.imwrite(os.path.join(save_dir, 'mask_volume2', f'slice_{i:04d}.tiff'), mask.astype(np.uint8) * 255)
            
            self.progress_bar.setValue(i + 1)
        
        self.progress_bar.hide()
        self.statusBar.showMessage("Images saved successfully!", 3000)
    
    def calculate_histogram(self):
        # Move data to CPU for histogram calculation
        volume1_cpu = self.volume1.cpu()
        volume2_cpu = self.volume2.cpu()
    
        # Flatten and combine volumes
        combined = torch.stack([volume1_cpu.flatten(), volume2_cpu.flatten()])
        
        # Calculate histogram
        hist_2d = torch.histogramdd(combined.t(), bins=256)
        
        # Convert to numpy for plotting
        hist = hist_2d.hist.numpy()
        xedges = hist_2d.bin_edges[0].numpy()
        yedges = hist_2d.bin_edges[1].numpy()
        
        # Plot histogram
        self.histogram_widget.plot_histogram(hist, xedges, yedges)
        
        # Set slice slider maximum
        self.slice_slider.setMaximum(self.volume1.shape[0] - 1)
    
        # Update range slider range based on histogram data
        hist_max = hist.max()
        self.histogram_widget.range_slider.setRange(0, int(hist_max))
        self.histogram_widget.range_slider.setValue((0, int(hist_max)))
    
        # Update status
        self.statusBar.showMessage("Histogram calculated. You can now adjust the scale and make a selection.", 5000)
    
        # Connect histogram_updated signal to update_segmentation
        self.histogram_widget.histogram_updated.connect(self.update_segmentation)
        
    def change_plane(self, plane):
        self.current_plane = plane
        self.update_segmentation()
    
    def update_segmentation(self):
        print("update_segmentation called")
        if self.volume1 is None or self.volume2 is None:
            print("Volumes not loaded yet")
            return
        slice_index = self.slice_slider.value()
    
        if self.current_plane == "XY":
            image1 = self.volume1[slice_index]
            image2 = self.volume2[slice_index]
        elif self.current_plane == "XZ":
            image1 = self.volume1[:, slice_index, :]
            image2 = self.volume2[:, :, slice_index]
        else:  # YZ
            image1 = self.volume1[:, :, slice_index]
            image2 = self.volume2[:, :, slice_index]
    
        # Move tensors to CPU and convert to numpy for operations
        image1_np = image1.cpu().numpy()
        image2_np = image2.cpu().numpy()
    
        print(f"Checking polygon in update_segmentation. Polygon is {'not None' if self.polygon is not None else 'None'}")
    
        # Debug: Print histogram information
        print(f"Histogram extent: {self.histogram_widget.im.get_extent()}")
        print(f"Histogram x-range: {np.min(self.histogram_widget.xedges)} to {np.max(self.histogram_widget.xedges)}")
        print(f"Histogram y-range: {np.min(self.histogram_widget.yedges)} to {np.max(self.histogram_widget.yedges)}")
    
        # Debug: Print image value ranges
        print(f"Image 1 value range: {np.min(image1_np)} to {np.max(image1_np)}")
        print(f"Image 2 value range: {np.min(image2_np)} to {np.max(image2_np)}")
    
        if self.polygon is not None:
            print("Polygon is not None, proceeding with segmentation")
            print(f"Polygon vertices: {self.polygon.vertices}")
            
            # Normalize polygon vertices
            x_min, x_max = np.min(image1_np), np.max(image1_np)
            y_min, y_max = np.min(image2_np), np.max(image2_np)
            if self.histogram_widget.log_scale:
                x_min, x_max = np.log10(x_min), np.log10(x_max)
                y_min, y_max = np.log10(y_min), np.log10(y_max)
                normalized_vertices = [(np.log10(x) - x_min) / (x_max - x_min) for x in self.polygon.vertices[:, 0]], \
                                      [(np.log10(y) - y_min) / (y_max - y_min) for y in self.polygon.vertices[:, 1]]
            else:
                normalized_vertices = [(x - x_min) / (x_max - x_min) for x in self.polygon.vertices[:, 0]], \
                                      [(y - y_min) / (y_max - y_min) for y in self.polygon.vertices[:, 1]]
            normalized_polygon = Path(list(zip(*normalized_vertices)))
            
            print("Normalized polygon vertices:")
            print(normalized_polygon.vertices)
    
            # Create a mask based on the histogram values
            mask_np = np.zeros(image1_np.shape, dtype=bool)
            points_checked = 0
            points_inside = 0
            for i in range(image1_np.shape[0]):
                for j in range(image1_np.shape[1]):
                    val1 = image1_np[i, j]
                    val2 = image2_np[i, j]
                    
                    if self.histogram_widget.log_scale:
                        point = [(np.log10(val1) - x_min) / (x_max - x_min),
                                 (np.log10(val2) - y_min) / (y_max - y_min)]
                    else:
                        point = [(val1 - x_min) / (x_max - x_min),
                                 (val2 - y_min) / (y_max - y_min)]
                    
                    points_checked += 1
                    if normalized_polygon.contains_point(point):
                        mask_np[i, j] = True
                        points_inside += 1
    
                    # Debug: Print some sample points
                    if points_checked % 10000 == 0:
                        print(f"Sample point: val1={val1}, val2={val2}, normalized=({point[0]:.4f}, {point[1]:.4f}), inside={normalized_polygon.contains_point(point)}")
    
            print(f"Total points checked: {points_checked}")
            print(f"Points inside polygon: {points_inside}")
            
            # Count values inside the polygon for both volumes
            values_inside_polygon1 = image1_np[mask_np]
            values_inside_polygon2 = image2_np[mask_np]
            
            if points_inside > 0:
                print(f"Range of values inside polygon (Volume 1): {np.min(values_inside_polygon1)} to {np.max(values_inside_polygon1)}")
                print(f"Range of values inside polygon (Volume 2): {np.min(values_inside_polygon2)} to {np.max(values_inside_polygon2)}")
            else:
                print("No values found inside the polygon. The segmentation might be empty.")
            
            # Debug: Visualize the mask and segmentation
            plt.figure(figsize=(20, 5))
            plt.subplot(141)
            plt.imshow(image1_np, cmap='gray')
            plt.title('Image 1 (Original)')
            plt.subplot(142)
            plt.imshow(mask_np, cmap='binary')
            plt.title('Mask')
            plt.subplot(143)
            plt.imshow(np.where(mask_np, image1_np, np.nan), cmap='gray')
            plt.title('Segmented Image 1')
            plt.subplot(144)
            plt.imshow(np.where(mask_np, image2_np, np.nan), cmap='gray')
            plt.title('Segmented Image 2')
            plt.show()
        else:
            print("No polygon selected, using full image mask")
            mask_np = np.ones_like(image1_np, dtype=bool)
            points_inside = np.prod(image1_np.shape)  # All points are inside when no polygon is selected
    
        # Apply segmentation
        segmented_image1 = np.where(mask_np, image1_np, np.nan)
        segmented_image2 = np.where(mask_np, image2_np, np.nan)
    
        print(f"Slice index: {slice_index}")
        print(f"Current plane: {self.current_plane}")
        print(f"Image 1 shape: {image1_np.shape}")
        print(f"Image 1 range: {np.nanmin(image1_np)} to {np.nanmax(image1_np)}")
        print(f"Image 2 shape: {image2_np.shape}")
        print(f"Image 2 range: {np.nanmin(image2_np)} to {np.nanmax(image2_np)}")
        print(f"Mask shape: {mask_np.shape}")
        print(f"Number of True values in mask: {np.sum(mask_np)}")
        print(f"Percentage of True values in mask: {np.sum(mask_np) / mask_np.size * 100:.2f}%")
    
        # Update the SegmentationWidget with the segmented images
        self.segmentation_widget.plot_images(image1_np, image2_np, mask_np, segmented_image1, segmented_image2)
    
        # Debug: Print sample values from the images
        print("Sample values from Image 1:")
        print(image1_np[:5, :5])
        print("Sample values from Image 2:")
        print(image2_np[:5, :5])
        