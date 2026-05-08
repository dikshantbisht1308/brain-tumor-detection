# Brain Tumor Detection - MRI Triage Pipeline

A computer vision pipeline for detecting and classifying brain tumors from MRI scans. The model uses a fine-tuned YOLO architecture to classify scans into four categories: Glioma, Meningioma, Pituitary, or No Tumor. 

This repository includes a model training script, a batch-processing CLI, and a Streamlit web interface designed for memory-safe bulk image triage.

## Dataset Information

The model was trained on a comprehensive dataset of MRI brain scans [sourced from Kaggle / Roboflow]. 

*   **Classes:** The data is labeled for multi-class object detection, identifying 3 specific tumor types (Glioma, Meningioma, Pituitary) and 1 healthy control class (No Tumor).
*   **Format:** Images are processed in standard RGB format with corresponding YOLO-format `.txt` bounding box annotations.
*   **Test Split:** The final model evaluation was conducted on a strict hold-out test set of 246 images to ensure the model had not memorized the training data.

*Note: Due to file size limits and medical data privacy best practices, the raw dataset is not included in this repository. The `.gitignore` is configured to exclude the `data/raw/` directory.*

## Performance & Metrics

The model was evaluated on the 246-image hold-out test set. To optimize for a medical triage use case, the confidence threshold is intentionally lowered to `0.05` to prioritize recall (minimizing false negatives) over precision.

*   **Sensitivity (Recall):** 97.9% (Missed 4 out of 197 actual tumors)
*   **Specificity:** 95.9% (Flagged 2 out of 49 healthy brains)
*   **mAP50 (Multi-class):** 0.941

## Project Structure

```text
brain-tumor-detection/
│
├── app.py                         # Streamlit web interface 
├── data/
│   └── raw/                       # Dataset directories (train/val/test)
├── notebooks/
│   └── 02_model_evaluation.ipynb  # Metric evaluation and visualization
├── runs/                          # Saved model weights and training logs
└── src/
    ├── train.py                   # Automated YOLO training pipeline
    └── predict.py                 # CLI tool for batch inference
```

## Setup and Usage

### 1. Web Interface (Streamlit)
The web app supports single image uploads and bulk folder uploads. It uses a deferred-rendering loop to process large batches of images without crashing browser memory.

```bash
streamlit run app.py
```

### 2. Command Line Interface (CLI)
The CLI tool is optimized for headless batch processing. It uses generator-based memory management to process large directories of high-resolution images sequentially.

```bash
# Process a single scan
python src/predict.py --source data/raw/test/images/patient_01.jpg

# Process an entire folder and save annotated outputs
python src/predict.py --source data/raw/test/images/ --save

# Override default confidence threshold
python src/predict.py --source data/raw/test/images/ --conf 0.15
```

### 3. Model Training
To retrain the model or test new hyperparameters:

```bash
python src/train.py --epochs 100 --batch 16 --imgsz 640
```
