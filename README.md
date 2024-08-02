# BiTS: Bivariate Tomography Segmentation Software

## Table of Contents
1. [Introduction](##-1.-Introduction)
2. [Installation](##-2.-Installation)
3. [Getting Started](##-3.-Getting-Started)
4. [User Interface](##-4.-User-Interface)
5. [Features](##-5.-Features)
6. [Workflow](##-6.-Workflow)
7. [Advanced Usage](##-7.-Advanced-Usage)
8. [Troubleshooting](##-8.-Troubleshooting)
9. [FAQ](##-9.-FAQ)
10. [Technical Details](##-10.-Technical-Details)
11. [Contributing](##-11.-Contributing)
12. [License](##-12.-License)

## 1. Introduction

BiTS (Bivariate Tomography Segmentation) is an advanced, open-source software tool designed for processing and interpreting large volumes of tomographic data across multiple scientific disciplines. It employs the bivariate histogram, allowing for intuitive and precise segmentation of 3D tomographic data based on two correlated volumetric datasets.

### Key Features:
- Interactive 2D histogram-based segmentation
- Support for both polygon and rectangular selection methods
- Real-time segmentation updates
- GPU acceleration (with CPU fallback)
- Compatibility with various tomographic data formats

## 2. Installation

### Prerequisites:
- Python 3.7+
- PyQt5
- NumPy
- Matplotlib
- PyTorch (for GPU acceleration)

### Installation Steps:
1. Clone the repository: `git clone https://github.com/OriolSansPlanell/BiTS.git`
2. Navigate to the BiTS directory: `cd BiTS`
3. Install required packages: `pip install -r requirements.txt`
4. Run the application: `python main.py`

## 3. Getting Started

To begin using BiTS:
1. Launch the application
2. Load your tomographic volumes (File > Load Volume 1/2)
3. The 2D histogram will automatically generate
4. Use polygon or rectangular selection tools to define regions of interest
5. Observe real-time segmentation results

## 4. User Interface

![image](https://github.com/user-attachments/assets/c5fa3383-a679-45ae-9542-772960a75358)

- A: Volume loading buttons
- B: 2D Histogram display
- C: Segmentation display
- D: Selection type toggle
- E: Slice navigation slider
- F: Save/Load settings buttons

## 5. Features

### 5.1 2D Histogram Generation
- Automatically generated from two loaded volumes
- Logarithmic scale option for better visualization of data distribution in certain datasets

### 5.2 Selection Methods
- Polygon selection: Freeform selection for complex regions
- Rectangular selection: Quick selection of rectangular regions

### 5.3 Real-time Segmentation
- Immediate visual feedback on segmentation results
- Efficient processing for large datasets

### 5.4 Multi-plane Viewing
- XY, XZ, and YZ plane options for comprehensive data exploration (under construction)

### 5.5 Save/Load Functionality
- Save segmentation settings for later use
- Load previous settings for consistency across sessions

## 6. Workflow

1. Load Volumes: Import your X-ray and neutron tomography data
2. Explore Data: Use the 2D histogram to understand data distribution
3. Select Region: Use polygon or rectangular selection to define ROI
4. Refine Segmentation: Adjust selection as needed, observing real-time updates
5. Analyze Results: Examine segmented data across different slices and planes
6. Save Results: Export segmented volumes and masks

## 7. Advanced Usage

### 7.1 GPU Acceleration
BiTS automatically utilizes GPU acceleration when available. 

### 7.2 Custom Colormap Selection
Will be implemented as soon as one afternoon is free.

### 7.3 Batch Processing
Coming soon...

## 8. Troubleshooting
Common issues and their solutions:

Issue: Volumes fail to load
Solution: Ensure files are in the correct format (TIFF stack)
Issue: GPU acceleration not working
Solution: Check CUDA installation and PyTorch GPU support

## 9. FAQ
Q: What file formats are supported?
A: Currently, BiTS supports TIFF stack formats for tomographic data.
Q: Can I use BiTS for single-volume segmentation?
A: While BiTS is optimized for bivariate data, it can be used with a single volume by duplicating the data for the second input.

## 10. Technical Details
Segmentation Algorithm
BiTS uses a histogram-based segmentation approach. The algorithm works as follows:

1. Generate 2D histogram from two input volumes
2. User defines ROI on the histogram
3. Each voxel is classified based on its position relative to the ROI in the histogram space

Performance Optimizations

PyTorch for GPU acceleration of computationally intensive tasks
Efficient numpy operations for CPU processing

## 11. Contributing
We welcome contributions to BiTS!

## 12. License
BiTS is released under the MIT License. See the LICENSE file for more details.
