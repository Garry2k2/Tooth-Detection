#!/usr/bin/env python3
import argparse
from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path

def assign_fdi_numbers(boxes, mirror=False):
    """
    boxes: Nx4 array [x1,y1,x2,y2] in image coords
    Returns list of tuples: (cx, cy, fdi_number, (x1,y1,x2,y2))
    Strategy:
      1) Split by median Y into upper vs lower jaw
      2) Split each jaw by median X into left vs right
      3) Sort within each quadrant from midline outward and assign 1..8
      4) Quadrants (patient perspective):
         Q1=upper-right, Q2=upper-left, Q3=lower-left, Q4=lower-right
         If --mirror is set, swap left/right sides.
    """
    if len(boxes) == 0:
        return []

    centers = []
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        centers.append((cx, cy, (x1, y1, x2, y2)))

    y_med = np.median([c[1] for c in centers])
    upper = [c for c in centers if c[1] < y_med]
    lower = [c for c in centers if c[1] >= y_med]

    def split_lr(items):
        if not items:
            return [], []
        x_med = np.median([c[0] for c in items])
        left = [c for c in items if c[0] < x_med]
        right = [c for c in items if c[0] >= x_med]
        return left, right, x_med

    up_left, up_right, _ = split_lr(upper)
    lo_left, lo_right, _ = split_lr(lower)

    # Orientation handling (patient vs image)
    if mirror:
        # image-left == patient right -> swap
        Q1, Q2, Q3, Q4 = up_left, up_right, lo_right, lo_left
        sides = {"Q1": "left", "Q2": "right", "Q3": "right", "Q4": "left"}
    else:
        Q1, Q2, Q3, Q4 = up_right, up_left, lo_left, lo_right
        sides = {"Q1": "right", "Q2": "left", "Q3": "left", "Q4": "right"}

    def order_quadrant(q, side):
        # Sort from midline outward
        if not q:
            return []
        x_med_local = np.median([c[0] for c in q])
        if side == "left":
            return sorted(q, key=lambda c: -c[0])  # near center (higher x) -> far left (lower x)
        else:
            return sorted(q, key=lambda c: c[0])   # near center (lower x) -> far right (higher x)

    qr1 = order_quadrant(Q1, sides["Q1"])
    qr2 = order_quadrant(Q2, sides["Q2"])
    qr3 = order_quadrant(Q3, sides["Q3"])
    qr4 = order_quadrant(Q4, sides["Q4"])

    def label(q, quad_code):
        return [(c[0], c[1], quad_code * 10 + (i + 1), c[2]) for i, c in enumerate(q)]

    f1 = label(qr1, 1)
    f2 = label(qr2, 2)
    f3 = label(qr3, 3)
    f4 = label(qr4, 4)
    return f1 + f2 + f3 + f4

def draw_fdi(image, fdi_items):
    for cx, cy, fdi, (x1, y1, x2, y2) in fdi_items:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            image, str(fdi), (int(x1), int(y1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )
    return image

def save_fdi_overlay(img_path, boxes, out_dir, mirror=False):
    img = cv2.imread(str(img_path))
    fdi_items = assign_fdi_numbers(boxes, mirror=mirror)
    vis = draw_fdi(img, fdi_items)
    out_path = Path(out_dir) / (Path(img_path).stem + "_FDI.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return out_path, fdi_items

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/detect/train3/weights/best.pt")
    p.add_argument("--source", required=True, help="Image, folder, or video")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--mirror", action="store_true",
                   help="Swap left/right to match patient perspective if your images are mirrored")
    p.add_argument("--out", default="runs/postprocess", help="Folder for FDI overlays")
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)

    # Run YOLOv8 inference and also build our FDI overlays
    results = model.predict(source=args.source, conf=args.conf, save=True, verbose=False)

    # Build FDI overlays per image result
    for r in results:
        path = r.path
        boxes = r.boxes.xyxy.cpu().numpy() if (r.boxes is not None and len(r.boxes) > 0) else np.zeros((0, 4))
        out_path, _ = save_fdi_overlay(path, boxes, args.out, mirror=args.mirror)
        print(f"🦷 FDI overlay saved: {out_path}")

    print("✅ Inference complete.")
    print("🖼 YOLO visualizations in runs/predict/* and FDI overlays in", args.out)

if __name__ == "__main__":
    main()
