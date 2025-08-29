#!/usr/bin/env python3
from ultralytics import YOLO
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/detect/train3/weights/best.pt")
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--split", default="test", choices=["val", "test"])
    return p.parse_args()

def main():
    args = parse_args()
    assert Path(args.weights).exists(), f"Weights not found: {args.weights}"
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, plots=True, save_json=True)
    # Ultralytics saves confusion_matrix.png inside the validation run dir
    print("✅ Validation done.")
    print("📊 Look under runs/detect/val*/ for plots (e.g., confusion_matrix.png, PR curve).")
    print(f"ℹ️ Metrics summary: {metrics.results_dict}")

if __name__ == "__main__":
    main()
