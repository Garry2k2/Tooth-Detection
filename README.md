# OralVis Tooth Detection (Internship Task)

This project is part of the OralVis AI Research Internship.  
The goal is to detect and number teeth in dental panoramic images using the **FDI numbering system (11–48)**.

---

## 🚀 Workflow
1. **Dataset Preparation**  
   - ~500 dental panoramic images with YOLO annotations.  
   - Train/Val/Test split: 80/10/10.  
   - Config file: `data.yaml` with 32 tooth classes.  

2. **Model Training**  
   - Trained YOLOv8s, YOLOv9s, YOLOv11s (100 epochs each).  
   - Best model: **YOLOv8s** (mAP@50 = 0.9930).  

3. **Model Evaluation**  
   - Metrics: Precision, Recall, mAP@50, mAP@50–95.  
   - Confusion matrix generated for all 32 tooth classes.  
   - Training curves plotted (`results.png`).  

4. **Post-Processing**  
   - Separated detections into upper/lower and left/right quadrants.  
   - Sorted teeth within quadrants and assigned **FDI IDs sequentially (11–48)**.  
   - Missing teeth handled automatically.  

---

## 📊 Results
- **Best Model**: YOLOv8s  
- Confusion Matrix: `outputs/confusion_matrix.png`  
- Training Curves: `outputs/results.png`  
- Sample Predictions: `outputs/sample_predictions/`

---

## 📂 Repo Structure
- `train.py` – Training YOLO models  
- `val.py` – Evaluate models + confusion matrix  
- `predict.py` – Run inference on sample images  
- `postprocess.py` – Assign FDI IDs & visualize  
- `requirements.txt` – Dependencies  

---

## ⚙️ Environment
- Python 3.10  
- ultralytics==8.2.103  
- torch==2.2.2  
- opencv-python, matplotlib, pandas  



