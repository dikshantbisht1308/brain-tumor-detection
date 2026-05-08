import argparse
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Detection Training Pipeline")
    parser.add_argument('--data', type=str, default='data/raw/data.yaml', help="Path to data.yaml dataset config")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch', type=int, default=16, help="Batch size (reduce if CUDA out of memory)")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for training")
    parser.add_argument('--name', type=str, default='tumor_detector', help="Name of the training run folder")
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help="Initial weights (default: yolov8n.pt)")

    args = parser.parse_args()

    # Validate data path before loading the model into GPU memory
    if not os.path.exists(args.data):
        print(f"Error: Dataset configuration file not found at '{args.data}'")
        print("Please ensure the path is correct relative to your current terminal directory.")
        return

    print(f"Initializing YOLO model from weights: {args.weights}")
    model = YOLO(args.weights)

    print(f"Starting training run '{args.name}'...")
    print(f"Parameters: {args.epochs} epochs | batch size {args.batch} | image size {args.imgsz}")
    
    # Execute training
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        project='runs/detect',
        exist_ok=True
    )

    print("\nTraining pipeline finished.")
    print(f"Check runs/detect/{args.name}/weights/best.pt for the saved model.")

if __name__ == "__main__":
    main()