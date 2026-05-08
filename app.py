import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import glob
import os

st.set_page_config(page_title="MRI Tumor Detector", layout="centered")
st.title("Brain Tumor Detection")
st.write("Upload MRI scans for automated triage (Glioma, Meningioma, Pituitary).")

@st.cache_resource
def load_model():
    models = glob.glob('runs/**/best.pt', recursive=True)
    if not models:
        st.error("Model weights not found. Please check runs directory.")
        st.stop()
    
    latest_model = max(models, key=os.path.getctime)
    return YOLO(latest_model)

model = load_model()

# Identify the background class
nt_ids = [k for k, v in model.names.items() if v.lower() == 'no tumor']
no_tumor_id = nt_ids[0] if nt_ids else -1

uploaded_files = st.file_uploader("Upload MRI scan(s)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    st.markdown("---")
    is_bulk = len(uploaded_files) > 1
    
    if is_bulk:
        st.subheader(f"Processing {len(uploaded_files)} scans...")
        progress_bar = st.progress(0)
    
    flagged = 0
    clean = 0

    for i, file in enumerate(uploaded_files):
        if is_bulk:
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        image = Image.open(file)
        img_array = np.array(image)
        
        # Format for YOLO
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        results = model.predict(source=img_array, conf=0.05, verbose=False)
        r = results[0]
        
        # Check if any bounding box belongs to a tumor class
        is_sick = any(int(box.cls[0]) != no_tumor_id for box in r.boxes)
                
        if is_sick:
            flagged += 1
            st.error(f"Flagged for review: {file.name}")
            
            res_plotted = r.plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption=f"Result: {file.name}", use_container_width=True)
            st.markdown("---")
        else:
            clean += 1
            if is_bulk:
                st.success(f"Clean: {file.name}")
            else:
                st.success("No tumors detected.")
                st.image(image, caption="Uploaded Scan", use_container_width=True)
                
    if is_bulk:
        st.markdown("### Batch Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(uploaded_files))
        c2.metric("Flagged", flagged)
        c3.metric("Clean", clean)