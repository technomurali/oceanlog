# Use this command to run this > streamlit run app.py --server.fileWatcherType none

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"  # Disable Streamlit file‑watcher that crashes on torch.classes
# If you prefer hot‑reload, comment the line above and uncomment the patch below
# import torch; torch.classes.__path__ = []

import streamlit as st
import sys
import time
import json
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------
ROOT_DIR = Path("/home/ubuntu/navaoceanlog/UI")
UPLOAD_DIR = ROOT_DIR / "uploaded_video_files"
OUTPUTS_DIR = ROOT_DIR / "video_outputs"
MODEL_INFERENCE_DIR = ROOT_DIR / "model_inference"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Add inference module to PYTHONPATH and import run_all
sys.path.append(str(MODEL_INFERENCE_DIR))
from agent_container import run_all  # noqa: E402

# ------------------------------------------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="Nava Ocean Log", layout="wide")

st.title("Nava Ocean Log")
st.subheader("Powered by AI: Real-time Container Tracking System")
st.markdown("---")

# ------------------------------------------------------------------------------------
# 1. Video Upload Section
# ------------------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a container video (.mp4) – max 500 MB",
    type=["mp4"],
    help="The video will be processed on the EC2 server."
)

# Persist timing information across reruns
if "upload_time" not in st.session_state:
    st.session_state.upload_time = None
if "exec_time" not in st.session_state:
    st.session_state.exec_time = None

if uploaded_file:
    video_name = uploaded_file.name
    uploaded_path = UPLOAD_DIR / video_name

    # Save video to disk and measure upload time
    with st.spinner("Saving video to server…"):
        t0 = time.time()
        with open(uploaded_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.upload_time = time.time() - t0
    st.success(f"Uploaded in {st.session_state.upload_time:.1f} s")

    # --------------------------------------------------------------------------------
    # 2. Processing Trigger
    # --------------------------------------------------------------------------------
    if st.button("Run AI Processing"):
        progress = st.progress(0, text="Starting AI inference… please wait")
        t1 = time.time()
        try:
            run_all(str(uploaded_path), str(OUTPUTS_DIR))
            progress.progress(100, text="Processing complete")
            st.session_state.exec_time = time.time() - t1
            st.success(f"Processing finished in {st.session_state.exec_time:.1f} s")
        except Exception as e:
            st.session_state.exec_time = None
            st.error(f"Processing failed: {e}")

# ------------------------------------------------------------------------------------
# 3. Result Display Section
# ------------------------------------------------------------------------------------
if st.session_state.get("exec_time"):
    stem = uploaded_file.name.rsplit(".", 1)[0]
    out_dir = OUTPUTS_DIR / f"{stem}_container_back"

    # Locate latest container_back image
    container_img_path = None
    if out_dir.exists():
        images = sorted(out_dir.glob("container_back*.jpg"))
        if images:
            container_img_path = images[-1]

    labels_dir = out_dir / "labels"
    json_path = labels_dir / "ocr_results.json"

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.header("Detected Container Image")
        if container_img_path and container_img_path.exists():
            st.image(str(container_img_path), use_container_width=True, caption=container_img_path.name)
        else:
            st.info("Image not available.")

    with col2: 
        st.header("Container Details")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows = [
                {
                    "Label Name": item.get("labelname", ""),
                    "Text Extracted": " ".join(item.get("text", [])),
                }
                for item in data
            ]
            st.table(pd.DataFrame(rows))
        else:
            st.info("OCR results not found.")

    # --------------------------------------------------------------------------------
    # 4. Performance Summary
    # --------------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Performance Summary")
    st.write(f"- **Upload time:** {st.session_state.upload_time:.1f} seconds")
    st.write(f"- **AI processing time:** {st.session_state.exec_time:.1f} seconds")
    st.write(f"- **Total time:** {st.session_state.upload_time + st.session_state.exec_time:.1f} seconds")
