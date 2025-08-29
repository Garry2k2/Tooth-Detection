#!/usr/bin/env python3
from ultralytics import YOLO
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8s.pt", help="Base model to finetune")
    p.add_argument("--data", default="data.yaml", help="Path to dataset yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="train3", help="Run name (creates runs/detect/<name>)")
    p.add_argument("--project", default="runs/detect", help="Project folder")
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
    )
    print("✅ Training complete.")
    print(f"📦 Best weights: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()
