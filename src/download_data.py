import os
from dotenv import load_dotenv
from roboflow import Roboflow

def fetch_brain_tumor_data():
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    
    if not api_key:
        raise ValueError("CRITICAL: ROBOFLOW_API_KEY not found in .env file.")

    # 1. Force an ABSOLUTE path so Roboflow cannot get confused
    project_root = os.getcwd() 
    output_dir = os.path.join(project_root, "data", "raw")
    
    print("Authenticating with Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    project = rf.workspace("ali-rostami").project("labeled-mri-brain-tumor-dataset")
    version = project.version(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Forcing download to absolute path: {output_dir}...")
    
    # 2. Download and explicitly capture the dataset object
    dataset = version.download(model_format="yolov8", location=output_dir)
    
    # 3. Print the EXACT location Roboflow saved it to
    print("\n✅ Download Complete!")
    print(f"📍 EXACT FOLDER PATH: {dataset.location}")

if __name__ == "__main__":
    fetch_brain_tumor_data()