## RGB-D Data Collector with Intel RealSense Camera

### About

This GUI-based tool allows you to capture synchronized RGB, depth, and segmentation mask images from an Intel RealSense D435i camera. It also automatically generates annotations in YOLO format, stores point clouds, and logs camera intrinsics and scene metadata.

### Camera Specifications 
Intel Realsense D435i

### Folder structure

```
├── main.py                  # Entry point: GUI app for capturing & labeling
├── camera_interface.py      # RealSense camera setup and frame retrieval
├── segmentation_helper.py   # Depth segmentation + plane removal
├── annotation_writer.py     # YOLO-style annotation writer
├── ply_viewer.py            # Script to view .ply images
├── requirements.txt         # Python dependencies
├── depth_info.py            # Script to convert .npy to depth image
├── README.md                # This file
```

### Setup procedure

1. pyrealsense2 supports Python 3.7–3.11, and does NOT support Python 3.12+ yet via PyPI (as of July 2025). So if you are using python versions that are 3.12 or above then
    1. Install Python via PPA if python versions are limited on your system
    ```+
    # 1. Install prerequisites
    sudo apt update
    sudo apt install software-properties-common

    # 2. Add deadsnakes PPA (trusted for Python versions)
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update

    # 3. Install Python 3.10 and venv
    sudo apt install python3.10 python3.10-venv
    ```

    2. To verify python version
    ```
    python3.10 --version
    ```

2. Create virtual environment with Python 3.10
    ```
    python3.10 -m venv ~/realsense-venv
    source ~/realsense-venv/bin/activate
    ```

3. Install the requirements

    The python libraries pyrealsense2, opencv-python, numpy, tkinter, pillow and open3d are essential.
    ```
    pip install -r requirements.txt
    ```

4. Run the main.py file
    ```
    python3 main.py
    ```

5. Controls (using kevboard or gui)
    1. Enter - Capture current RGB-D frame
    2. S - Save captured frame and label
    3. R - Retake / discard frame
    4. P - Preview 3D point cloud
    5. Q - Quit the application
    6. Dropdown to select the class (0 - Copper, 1 - Steel)

6. Captured data is stored in a dataset/ folder
    ```
    dataset/
    ├── images/         # RGB images (img0000.png, ...)
    ├── depth/          # Normalized color depth images (.png)
    ├── labels/         # YOLO-format annotations (.txt) <class_label> x1 y1 x2 y2 x3 y3 ... xn yn
    ├── pointcloud/     # 3D point clouds (.ply)
    ├── info/           # Per-frame metadata logs like intrinsics and depth info (.txt)
    ```
### Features

- Synchronized RGB + Depth capture
- Depth-based segmentation with plane removal using RANSAC
- YOLO-style annotation writer
- Tkinter GUI with interactive controls
- Live preview and contour overlay
- Saves metadata, point cloud, and intrinsics per frame