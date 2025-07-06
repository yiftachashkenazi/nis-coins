# YOLOv8 Israeli Coin Detector

## Project Overview
This project implements a coin detection and sum evaluation system for Israeli coins using YOLOv8 for object detection and OpenCV for circle detection and refinement. The goal is to accurately identify and count Israeli coins in images, and then calculate their total value.

## Attribution
Developed by Yiftach Ashlekanzi as part of the Google × Reichman Artificial‑Intelligence & Deep‑Learning Development Program.

## Project Structure
- `images/from data set/Israeli-coins-detected-3/`: Original dataset with train, valid, and test splits.
- `merged_output/`: Contains the merged COCO annotations and all images consolidated from the original dataset splits.
- `yolo_dataset/`: Formatted dataset in YOLO format, including `data.yaml` configuration, images, and labels for train, validation, and test sets.
- `yolo_runs/`: Directory where YOLOv8 training logs, weights, and results are stored.
- `merge_dataset.py`: Python script to merge the original COCO datasets into a single, unified dataset.
- `yolo_coin_detector_v2.py`: The main script for YOLOv8 model training and evaluation, including OpenCV-based circle filtering.
- `yolo_coin_detector_v2_gpu.py`: GPU-optimized version for M1 Mac with MPS acceleration (recommended for faster training).
- `requirements.txt`: Lists all Python dependencies required to run the project.
- `error_analysis_table.csv`: Output CSV file containing detailed error analysis after model evaluation.

## Setup and Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd nis-coins
    ```
2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1.  **Prepare the dataset:**
    First, run the `merge_dataset.py` script to consolidate the annotation files and images.
    ```bash
    python merge_dataset.py
    ```
2.  **Train and evaluate the YOLOv8 model:**
    After preparing the dataset, run the main YOLOv8 script to start training and evaluation.
    
    **For M1 Mac users (recommended):**
    ```bash
    conda activate yolo-gpu
    python yolo_coin_detector_v2_gpu.py
    ```
    
    **For other systems:**
    ```bash
    python yolo_coin_detector_v2.py
    ```
    
    The training progress will be logged in the `yolo_runs` directory, and an `error_analysis_table.csv` will be generated upon completion.

## Results
Upon completion of the `yolo_coin_detector_v2.py` script, an `error_analysis_table.csv` will be generated in the root directory, providing a detailed breakdown of prediction errors and accuracy metrics.

## Future Work
- Improve model accuracy with further hyperparameter tuning and data augmentation.
- Explore real-time coin detection and sum calculation from video streams.
- Implement a user interface for easier interaction and visualization of results.
