# yolo_coin_detector_v2.py
"""
YOLOv8 Coin Detector Pipeline (Modular, Documented, Maintainable)
- Converts COCO to YOLO
- Trains YOLOv8 with augmentations
- Evaluates using OpenCV HoughCircles
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

# ========== CONFIGURATION ==========
BASE_DIR = Path("/Users/yiftachashkenazi/nis-coins")
ANNOTATIONS = BASE_DIR / "merged_output/merged_annotations.coco.json"
IMAGES_DIR = BASE_DIR / "merged_output/images"
SPLIT_DIR = BASE_DIR / "yolo_dataset"
YOLO_MODEL = "yolov8s.pt"
EPOCHS = 30
LR0, LRF = 0.001, 0.20
SEED = 42
SPLITS = {"train": 0.7, "val": 0.1, "test": 0.2}
CONFIDENCE_THRESHOLD = 0.35

# Coin values
VALUE_MAP = {"One": 1, "Two": 2, "Five": 5, "Ten": 10}

# ========== UTILITIES ==========
def resize_image(img: np.ndarray, size: int = 640) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def find_coin_circles(img: np.ndarray, min_r=10, max_r=160) -> List[List[int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=60, param2=30, minRadius=min_r, maxRadius=max_r)
    boxes = []
    if circles is not None:
        for (x, y, r) in np.round(circles[0]).astype(int):
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = x + r, y + r
            boxes.append([x1, y1, x2, y2])
    return boxes

def center_inside_any(box: List[float], circles: List[List[int]]) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in circles)

# ========== DATA PREP ==========
def prepare_dataset():
    with open(ANNOTATIONS, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]
    class_names = [c["name"] for c in categories]
    num_classes = len(class_names)

    random.seed(SEED)
    random.shuffle(images)
    n = len(images)
    tr = int(n * SPLITS["train"])
    vl = tr + int(n * SPLITS["val"])
    splits = {"train": images[:tr], "val": images[tr:vl], "test": images[vl:]}

    # Make directories
    for split in splits:
        (SPLIT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (SPLIT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
        with open(SPLIT_DIR / f"{split}_split.json", 'w', encoding='utf-8') as f:
            json.dump([img['file_name'] for img in splits[split]], f, indent=2)

    # Group annotations
    ann_by_img = {}
    for a in annotations:
        ann_by_img.setdefault(a['image_id'], []).append(a)

    # Save labels and resized images
    for split, img_list in splits.items():
        for img in img_list:
            w, h = img['width'], img['height']
            label_lines = []
            for a in ann_by_img.get(img['id'], []):
                x, y, bw, bh = a['bbox']
                label_lines.append(
                    f"{a['category_id']} {(x + bw/2) / w:.6f} {(y + bh/2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                )
            (SPLIT_DIR / split / 'labels' / (Path(img['file_name']).stem + '.txt')).write_text("\n".join(label_lines))
            img_bgr = cv2.imread(str(IMAGES_DIR / img['file_name']))
            img_bgr = resize_image(img_bgr)
            cv2.imwrite(str(SPLIT_DIR / split / 'images' / img['file_name']), img_bgr)

    # Save YAML config
    yaml.dump({
        'path': str(SPLIT_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': num_classes,
        'names': class_names
    }, open(SPLIT_DIR / 'data.yaml', 'w', encoding='utf-8'), allow_unicode=True)
    
    return splits, annotations, class_names

# ========== TRAIN ==========
def train_yolo():
    model = YOLO(YOLO_MODEL)
    model.train(
        data=str(SPLIT_DIR / 'data.yaml'),
        epochs=EPOCHS,
        lr0=LR0, lrf=LRF,
        augment=True, mosaic=1.0, mixup=0.5, imgsz=640,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=15,
        translate=0.1, scale=0.5, flipud=0.5, fliplr=0.5,
        project=str(BASE_DIR / 'yolo_runs'), name='coin_detector_cv'
    )
    return model

# ========== EVALUATE ==========
def evaluate(model: YOLO, test_images: List[Dict], annotations: List[Dict], class_names: List[str]):
    gt_sums = {}
    pred_sums = {}
    ann_by_img = {img['id']: [] for img in test_images}
    for a in annotations:
        if a['image_id'] in ann_by_img:
            ann_by_img[a['image_id']].append(a)

    for img in test_images:
        fname = img['file_name']
        gt_sums[fname] = sum(VALUE_MAP[class_names[a['category_id']]] for a in ann_by_img[img['id']])

    results = model.predict(source=str(SPLIT_DIR / 'test/images'), conf=CONFIDENCE_THRESHOLD)
    for res in results:
        fname = Path(res.path).name
        img_bgr = cv2.imread(str(SPLIT_DIR / 'test/images' / fname))
        img_bgr = resize_image(img_bgr)
        circles = find_coin_circles(img_bgr)
        keep = [VALUE_MAP[class_names[int(cls)]]
                for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy())
                if not circles or center_inside_any(box, circles)]
        pred_sums[fname] = sum(keep)

    # Error Table
    rows = []
    for fname, gt in gt_sums.items():
        pr = pred_sums.get(fname, 0)
        err = pr - gt
        acc = 1 - abs(err) / gt if gt > 0 else (1 if pr == 0 else 0)
        flag = ("FP" if gt == 0 and pr > 0 else
                "FN" if gt > 0 and pr == 0 else
                "Large" if abs(err) >= max(2, 0.3 * gt) else
                "Minor" if abs(err) > 0 else "OK")
        rows.append({"File": fname, "GT": gt, "Pred": pr, "Err": err, "Acc": f"{acc:.2%}", "Flag": flag})

    df = pd.DataFrame(rows)
    df.to_csv(BASE_DIR / "error_analysis_table.csv", index=False)
    print("Top Errors:")
    print(df.sort_values('Err', ascending=False).head(10))
    print("\nAverage Absolute Error:", df['Err'].abs().mean())
    print("Average Accuracy:", df['Acc'].str.rstrip('%').astype(float).mean())

# ========== MAIN RUN ==========
if __name__ == '__main__':
    splits, annotations, class_names = prepare_dataset()
    model = train_yolo()
    evaluate(model, splits['test'], annotations, class_names)