import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="YOLO Vision — Helmet Detection App",
    page_icon="🪖",
    layout="wide"
)

# -----------------------------
# BEAUTIFUL HEADER
# -----------------------------
st.markdown("""
<style>
.big-title {
    font-size:42px !important;
    font-weight:900;
}
.subtitle {
    font-size:18px !important;
    opacity:0.8;
}
.footer {
    text-align:center;
    opacity:0.6;
    font-size:13px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🪖 YOLO Vision — Helmet Detection App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image or video and let AI detect helmets, vests, persons & shoes in real-time</p>', unsafe_allow_html=True)
st.write("---")


# -----------------------------
# LOAD MODEL (CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

with st.spinner("Loading YOLO model..."):
    model = load_model()


# -----------------------------
# SIDEBAR — SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Detection Settings")

conf = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25)
iou = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.45)

mode = st.sidebar.radio("Choose Input Type", ["Image", "Video"])

st.sidebar.write("---")
st.sidebar.info("Tip: Lower confidence → more detections\nHigher confidence → more accurate results")


# -----------------------------
# IMAGE MODE
# -----------------------------
if mode == "Image":
    file = st.file_uploader("📸 Upload an Image", type=["jpg","jpeg","png"])

    if file:
        col1, col2 = st.columns(2)

        with col1:
            img = Image.open(file)
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col2:
            if st.button("🚀 Run Detection", use_container_width=True):

                with st.spinner("Detecting objects..."):
                    results = model.predict(img, conf=conf, iou=iou)
                    annotated = results[0].plot()

                st.success("Detection complete 🎉")
                st.image(annotated, caption="Detected Objects", use_container_width=True)

                names = results[0].names
                boxes = results[0].boxes.cls.tolist()

                if boxes:
                    st.subheader("📊 Detection Summary")
                    for c in set(boxes):
                        st.write(f"• {names[int(c)]}: {boxes.count(c)}")
                else:
                    st.info("No objects detected.")


# -----------------------------
# VIDEO MODE
# -----------------------------
else:
    vid = st.file_uploader("🎬 Upload a Video", type=["mp4","mov","avi","mkv"])

    if vid:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        fps_text = st.empty()

        with st.spinner("Processing video..."):
            while cap.isOpened():
                start = time.time()
                ret, frame = cap.read()

                if not ret:
                    break

                results = model(frame, conf=conf, iou=iou)
                annotated = results[0].plot()

                fps = 1 / (time.time() - start)

                stframe.image(annotated, channels="BGR", use_container_width=True)
                fps_text.markdown(f"**⚡ FPS:** `{fps:.2f}`")

        cap.release()
        st.success("Video processing complete 🎬")


# -----------------------------
# FOOTER
# -----------------------------
st.write("---")
st.markdown(
    '<p class="footer">Made with ❤️ using YOLO & Streamlit</p>',
    unsafe_allow_html=True
)
