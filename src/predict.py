import argparse
import os
import glob
from ultralytics import YOLO

def get_model_path():
    """Finds the most recently modified best.pt weights."""
    search_paths = ['runs/**/best.pt', '../runs/**/best.pt', 'notebooks/runs/**/best.pt']
    models = []
    
    for path in search_paths:
        models.extend(glob.glob(path, recursive=True))
        
    if not models:
        raise FileNotFoundError("Model weights (best.pt) not found.")
    
    return max(models, key=os.path.getctime)

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Detection CLI")
    parser.add_argument('--source', type=str, required=True, help="Image or directory path")
    parser.add_argument('--weights', type=str, default=None, help="Path to weights file")
    parser.add_argument('--conf', type=float, default=0.05, help="Confidence threshold")
    parser.add_argument('--save', action='store_true', help="Save annotated images")
    
    args = parser.parse_args()

    model_path = args.weights if args.weights else get_model_path()
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    nt_ids = [k for k, v in model.names.items() if v.lower() == 'no tumor']
    no_tumor_id = nt_ids[0] if nt_ids else -1
    
    if no_tumor_id == -1:
        print("Warning: 'No Tumor' class not found. Treating all detections as anomalies.")

    print(f"Processing source: {args.source} (conf={args.conf})")
    results = model.predict(source=args.source, conf=args.conf, save=args.save, stream=True, verbose=False)

    total = 0
    flagged = 0

    print("\nResults:")
    print("-" * 30)

    for r in results:
        total += 1
        filename = os.path.basename(r.path)
        
        is_sick = any(int(box.cls[0]) != no_tumor_id for box in r.boxes)
                
        if is_sick:
            print(f"[FLAGGED] {filename}")
            flagged += 1
        else:
            print(f"[CLEAN]   {filename}")

    print("-" * 30)
    print(f"Total processed: {total}")
    print(f"Flagged scans:   {flagged}")
    
    if args.save:
        print("Annotated images saved to runs/detect/predict")

if __name__ == "__main__":
    main()