## Capturing RGBD Images Using an Intel Realsense Camera

### Camera Specifications 
Intel Realsense D435i

### Folder structure

├── annotation_writer.py # Handles saving YOLO-format annotations
├── camera_interface.py # Intel RealSense camera setup and frame capture
├── main.py # Entry point for the GUI application
├── segmentation_helper.py # Performs segmentation (depth-based or model-based)
├── utils.py # Utility functions (e.g., frame conversion)
├── requirements.txt # Python dependencies
├── README.md # This file
├── blah.txt # Temporary/unused file

### Setup procedure

python3.10 -m venv ~/realsense-venv
source ~/realsense-venv/bin/activate


### Yolo format
<class_label> x1 y1 x2 y2 x3 y3 ... xn yn

Formula to Project 3D Point to 2D Pixel

Given a 3D point (x,y,z)(x,y,z) and intrinsics (fx,fy,cx,cy)(fx,fy,cx,cy):
 u = (x . fx) / z + cx
 v = (y . fy) / z + cy