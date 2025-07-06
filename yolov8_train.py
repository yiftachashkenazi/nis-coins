#!/usr/bin/env python
# coding: utf-8
"""
YOLOv8 + OpenCV
"""


import os, json, random, shutil, io
from pathlib import Path
import yaml, pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

#Resize image to 640x640
def resize_to_640(img_bgr):
    return cv2.resize(img_bgr, (640, 640), interpolation=cv2.INTER_AREA)

#  Configuration
BASE_DIR    = Path("/Users/yiftachashkenazi/nis-coins")
ANNOTATIONS = BASE_DIR / "merged_output" / "merged_annotations.coco.json"
IMAGES_DIR  = BASE_DIR / "merged_output" / "images"
SPLIT_DIR   = BASE_DIR / "yolo_dataset"
SEED        = 42
SPLITS      = {"train": 0.7, "val": 0.1, "test": 0.2} # Adjusted test split to make total 1.0 (0.7 + 0.1 + 0.2)
YOLO_MODEL  = "yolov8s.pt"
EPOCHS      = 30
LR0, LRF    = 0.001, 0.20

# OpenCV suppurt function
def find_coin_circles(img_bgr, min_r=10, max_r=160):
    """Return list of [x1,y1,x2,y2] boxes around circles found by Hough."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=60, param2=30, minRadius=min_r, maxRadius=max_r
    )
    boxes = []
    if circles is not None:
        for (x, y, r) in np.round(circles[0]).astype(int):
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = x + r, y + r
            boxes.append([x1, y1, x2, y2])
    return boxes

def center_in_any_circle(box, circles):
    """True if YOLO bbox-center falls in at least one circle-bbox."""
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    for c in circles:
        if c[0] <= cx <= c[2] and c[1] <= cy <= c[3]:
            return True
    return False

# Data formation
with open(ANNOTATIONS, 'r', encoding='utf8') as f:
    coco = json.load(f)
images, annotations, categories = coco["images"], coco["annotations"], coco["categories"]
class_names = [c["name"] for c in categories]
value_map   = {"One":1, "Two":2, "Five":5, "Ten":10, "Half_Shekel": 0.5, "Ten_Agora": 0.1, "Five_Shekel": 5, "Two_Shekel": 2, "One_Shekel": 1} # Added missing values
num_classes = len(class_names)

random.seed(SEED); random.shuffle(images)
n = len(images); tr = int(n*SPLITS["train"]); vl = tr+int(n*SPLITS["val"])
splits = {"train":images[:tr], "val":images[tr:vl], "test":images[vl:]}

for s in splits:
    (SPLIT_DIR/s/'images').mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR/s/'labels').mkdir(parents=True, exist_ok=True)
    with open(SPLIT_DIR/f"{s}_split.json",'w',encoding='utf8') as f:
        json.dump([i['file_name'] for i in splits[s]], f, indent=2, ensure_ascii=False)

ann_by_img = {}
for a in annotations:
    ann_by_img.setdefault(a['image_id'],[]).append(a)

def save_yolo_labels(img_list, split):
    for img in img_list:
        w,h = img['width'], img['height']
        rows=[]
        for a in ann_by_img.get(img['id'],[]):
            x,y,bw,bh=a['bbox']
            # Ensure category_id is within bounds of class_names
            if a['category_id'] < len(class_names):
                rows.append(f"{a['category_id']} {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}")
            else:
                print(f"Warning: category_id {a['category_id']} out of bounds for image {img['file_name']}")
        (SPLIT_DIR/split/'labels'/ (Path(img['file_name']).stem+'.txt')).write_text("\n".join(rows))
        img_path = IMAGES_DIR / img['file_name']
        out_path = SPLIT_DIR/split/'images'/img['file_name']
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is not None:
            img_bgr = resize_to_640(img_bgr)
            cv2.imwrite(str(out_path), img_bgr)
        else:
            print(f"Warning: Could not read image {img_path}. Skipping.")

for s,l in splits.items(): save_yolo_labels(l,s)

yaml.dump({
    'path': str(SPLIT_DIR),'train':'train/images','val':'val/images','test':'test/images',
    'nc':num_classes,'names':class_names
}, (SPLIT_DIR/'data.yaml').open('w',encoding='utf8'), allow_unicode=True)

# Train YOLOv8
model = YOLO(YOLO_MODEL)
model.train(
    data=str(SPLIT_DIR/'data.yaml'), epochs=EPOCHS, lr0=LR0, lrf=LRF,
    augment=True, mosaic=1.0, mixup=0.5, imgsz=640,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=15,
    translate=0.1, scale=0.5, flipud=0.5, fliplr=0.5,
    project=str(BASE_DIR/'yolo_runs'), name='coin_detector_cv'
)
print("Training complete :)")

# Evaluation with circle-filter
print("\nEvaluating coin sums on test set:")
gt_sums, pred_sums = {}, {}
for img in splits['test']:
    fname = img['file_name']
    gt_sums[fname] = sum(value_map[class_names[a['category_id']]]
                         for a in annotations if a['image_id']==img['id'])

test_dir = SPLIT_DIR/'test'/'images'
results  = model.predict(source=str(test_dir), conf=0.35)

for res in results:
    fname = Path(res.path).name
    img_bgr = cv2.imread(str(test_dir/fname))
    if img_bgr is not None:
        img_bgr = resize_to_640(img_bgr)
        circles = find_coin_circles(img_bgr)
        keep = []
        for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            if not circles or center_in_any_circle(box, circles):
                keep.append(value_map[class_names[int(cls)]])
        pred_sums[fname] = sum(keep)
    else:
        print(f"Warning: Could not read image {test_dir/fname} for prediction. Skipping.")

#  Error table
rows=[]
for f,gt in gt_sums.items():
    pr = pred_sums.get(f,0)
    err,abs_err = pr-gt, abs(pr-gt)
    acc = 1-(abs_err/gt) if gt>0 else (1 if pr==0 else 0)
    flag = ("FP" if gt==0 and pr>0 else "FN" if gt>0 and pr==0
            else "Large" if abs_err>=max(2,0.3*gt) else "Minor" if abs_err>0 else "OK")
    rows.append(dict(File=f,GT=gt,Pred=pr,Err=err,Acc=f"{acc:.2%}",Flag=flag))
df=pd.DataFrame(rows); print(df.sort_values('Err',ascending=False).head(20))
df.to_csv(BASE_DIR/"error_analysis_table.csv",index=False)
print("Avg Err",df['Err'].abs().mean()," | Avg Acc",df['Acc'].str.rstrip('%').astype(float).mean())
print("done") 