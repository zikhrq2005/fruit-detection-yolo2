#pip install streamlit ultralytics Pillow opencv-python
#streamlit run app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Set up the Streamlit page layout

st.set_page_config(page_title="Fruit Detection App", layout="wide")

# SIDEBAR CONTROLS

st.sidebar.title("⚙ Detection Settings")

# Page navigation
page = st.sidebar.selectbox(
    "Choose Page",
    [
        "Home",
        "Single Model Detection",
        "Multi-Model Comparison",
        "Model Metrics"
    ]
)

st.sidebar.markdown("---")

# 1. Model Selection
available_models = [
    "yolov5.pt",    # Subteam A
    "yolov8.pt",    # Subteam A
    "yolov9.pt",    # Subteam B
    "yolov10.pt"    # Subteam B
]

selected_model_name = st.sidebar.selectbox("Choose YOLO Model", available_models)

# 2. Confidence Slider
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.01,
    max_value=1.00,
    value=0.25,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.info("Adjust the slider to filter out low-confidence detections.")

# Load model — uses cache so model is only loaded once

@st.cache_resource
def load_model(model_name):
    try:
        return YOLO(model_name)
    except Exception as e:
        st.error(f"Failed to load {model_name}. Make sure the .pt file is in the same folder as app.py!")
        return None

# PAGE 1 — HOME

if page == "Home":

    st.title("🍎 Fruit Counting and Instance Detection")
    st.subheader("Visual Computing and Image Processing — 25CSAI02I")
    st.markdown("---")

    st.markdown("### About This Project")
    st.write(
        "This app is the final deliverable for our Visual Computing project. "
        "We trained 4 YOLO models across 2 subteams on 2 different fruit datasets "
        "to compare detection performance."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Subteam A")
        st.write("Dataset: Roboflow — fruit-detection-2aajq/fruit-nxoac")
        st.write("Images: 825")
        st.write("Classes: Apple, Tomato, Cauliflower, Adrak, Bitter Gourd, Capsicum")
        st.write("Models: yolov5.pt and yolov8.pt")

    with col2:
        st.markdown("#### Subteam B")
        st.write("Dataset: Roboflow — fruitsdetection/fruits-by-yolo")
        st.write("Images: 1,176")
        st.write("Models: yolov9.pt and yolov10.pt")

    st.markdown("---")
    st.markdown("### How to Use This App")
    st.write("Use the sidebar to navigate between pages:")
    st.write("- Single Model Detection: Upload an image and run one model")
    st.write("- Multi-Model Comparison: Compare all 4 models on the same image")
    st.write("- Model Metrics: View precision, recall, mAP, and F1 scores")


# PAGE 2 — SINGLE MODEL DETECTION

elif page == "Single Model Detection":

    st.title("Single Model Detection")
    st.write("Upload an image and run detection using the model selected in the sidebar.")

    # Load the selected model
    model = load_model(selected_model_name)

    # Create the file uploader
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model is not None:

        # Read the uploaded image using PIL
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detection Results")
            with st.spinner(f"Running {selected_model_name}..."):

                # Run inference using the confidence value from the sidebar slider
                results = model.predict(image, conf=confidence_threshold)

                # Extract and convert the annotated image
                annotated_img_bgr = results[0].plot()
                annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)

                # Display the result
                st.image(annotated_img_rgb, use_container_width=True)

        # Display the raw detection data
        with st.expander("View Detection Details"):
            boxes = results[0].boxes
            if len(boxes) == 0:
                st.write(f"No objects detected with confidence above {confidence_threshold:.0%}.")
            else:
                # Count detections per class
                class_counts = {}
                for box in boxes:
                    class_id   = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    st.write(f"- **{class_name}**: {confidence:.2%} confidence")
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                st.markdown("---")
                st.write("**Total objects detected:**", len(boxes))
                for class_name, count in class_counts.items():
                    st.write(f"- {class_name}: {count}")

# PAGE 3 — MULTI-MODEL COMPARISON

elif page == "Multi-Model Comparison":

    st.title("Multi-Model Comparison")
    st.write("Upload one image and compare all 4 YOLO models side by side.")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        # Read the uploaded image using PIL
        image = Image.open(uploaded_file)

        st.subheader("Original Image")
        st.image(image, width=400)
        st.markdown("---")
        st.subheader("Detection Results — All 4 Models")

        # Row 1: YOLOv5 and YOLOv8
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**yolov5.pt — Subteam A**")
            modelV5 = load_model("yolov5.pt")
            if modelV5 is not None:
                with st.spinner("Running yolov5.pt..."):
                    results = modelV5.predict(image, conf=confidence_threshold)
                    annotated_img_bgr = results[0].plot()
                    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                    st.image(annotated_img_rgb, use_container_width=True)
                    boxes = results[0].boxes
                    st.write("Objects detected:", len(boxes))
                    with st.expander("View detections"):
                        for box in boxes:
                            class_id   = int(box.cls[0])
                            class_name = modelV5.names[class_id]
                            confidence = float(box.conf[0])
                            st.write(f"- **{class_name}**: {confidence:.2%}")
            else:
                st.error("yolov5.pt not found.")

        with col2:
            st.markdown("**yolov8.pt — Subteam A**")
            modelV8 = load_model("yolov8.pt")
            if modelV8 is not None:
                with st.spinner("Running yolov8.pt..."):
                    results = modelV8.predict(image, conf=confidence_threshold)
                    annotated_img_bgr = results[0].plot()
                    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                    st.image(annotated_img_rgb, use_container_width=True)
                    boxes = results[0].boxes
                    st.write("Objects detected:", len(boxes))
                    with st.expander("View detections"):
                        for box in boxes:
                            class_id   = int(box.cls[0])
                            class_name = modelV8.names[class_id]
                            confidence = float(box.conf[0])
                            st.write(f"- **{class_name}**: {confidence:.2%}")
            else:
                st.error("yolov8.pt not found.")

        st.markdown("---")

        # Row 2: YOLOv9 and YOLOv10
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**yolov9.pt — Subteam B**")
            modelV9 = load_model("yolov9.pt")
            if modelV9 is not None:
                with st.spinner("Running yolov9.pt..."):
                    results = modelV9.predict(image, conf=confidence_threshold)
                    annotated_img_bgr = results[0].plot()
                    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                    st.image(annotated_img_rgb, use_container_width=True)
                    boxes = results[0].boxes
                    st.write("Objects detected:", len(boxes))
                    with st.expander("View detections"):
                        for box in boxes:
                            class_id   = int(box.cls[0])
                            class_name = modelV9.names[class_id]
                            confidence = float(box.conf[0])
                            st.write(f"- **{class_name}**: {confidence:.2%}")
            else:
                st.error("yolov9.pt not found.")

        with col4:
            st.markdown("**yolov10.pt — Subteam B**")
            modelV10 = load_model("yolov10.pt")
            if modelV10 is not None:
                with st.spinner("Running yolov10.pt..."):
                    results = modelV10.predict(image, conf=confidence_threshold)
                    annotated_img_bgr = results[0].plot()
                    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                    st.image(annotated_img_rgb, use_container_width=True)
                    boxes = results[0].boxes
                    st.write("Objects detected:", len(boxes))
                    with st.expander("View detections"):
                        for box in boxes:
                            class_id   = int(box.cls[0])
                            class_name = modelV10.names[class_id]
                            confidence = float(box.conf[0])
                            st.write(f"- **{class_name}**: {confidence:.2%}")
            else:
                st.error("yolov10.pt not found.")


# PAGE 4 — MODEL METRICS

elif page == "Model Metrics":

    st.title("Model Performance Metrics")
    st.write("Comparison of all 4 trained YOLO models based on validation results.")
    st.markdown("---")

    # F1 score calculation
    # F1 = 2 * (precision * recall) / (precision + recall)
    def calculate_f1(precision, recall):
        if precision + recall == 0:
            return 0.0
        return round(2 * (precision * recall) / (precision + recall), 3)

    # Subteam A
    v5_map50     = 0.989
    v5_map       = 0.900
    v5_precision = 0.988
    v5_recall    = 0.996
    v5_f1        = calculate_f1(v5_precision, v5_recall)

    v8_map50     = 0.988
    v8_map       = 0.885
    v8_precision = 0.989
    v8_recall    = 0.996
    v8_f1        = calculate_f1(v8_precision, v8_recall)

    # Subteam B
    v9_map50     = 0.740
    v9_map       = 0.614
    v9_precision = 0.707
    v9_recall    = 0.706
    v9_f1        = calculate_f1(v9_precision, v9_recall)

    v10_map50     = 0.709
    v10_map       = 0.567
    v10_precision = 0.739
    v10_recall    = 0.660
    v10_f1        = calculate_f1(v10_precision, v10_recall)

    # Metrics table
    st.subheader("Comparison Table — All 4 Models")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.markdown("**Metric**")
    col2.markdown("**YOLOv5**")
    col3.markdown("**YOLOv8**")
    col4.markdown("**YOLOv9**")
    col5.markdown("**YOLOv10**")

    col1.write("mAP@0.5")
    col2.write(v5_map50)
    col3.write(v8_map50)
    col4.write(v9_map50)
    col5.write(v10_map50)

    col1.write("mAP@0.5:0.95")
    col2.write(v5_map)
    col3.write(v8_map)
    col4.write(v9_map)
    col5.write(v10_map)

    col1.write("Precision")
    col2.write(v5_precision)
    col3.write(v8_precision)
    col4.write(v9_precision)
    col5.write(v10_precision)

    col1.write("Recall")
    col2.write(v5_recall)
    col3.write(v8_recall)
    col4.write(v9_recall)
    col5.write(v10_recall)

    col1.write("F1 Score")
    col2.write(v5_f1)
    col3.write(v8_f1)
    col4.write(v9_f1)
    col5.write(v10_f1)

    st.markdown("---")

    # mAP@0.5 bar chart 
    st.subheader("mAP@0.5 Comparison Chart")
    map50_data = {
        "YOLOv5" : v5_map50,
        "YOLOv8" : v8_map50,
        "YOLOv9" : v9_map50,
        "YOLOv10": v10_map50,
    }
    st.bar_chart(map50_data)

    st.markdown("---")

    # Precision bar chart
    st.subheader("Precision Comparison Chart")
    precision_data = {
        "YOLOv5" : v5_precision,
        "YOLOv8" : v8_precision,
        "YOLOv9" : v9_precision,
        "YOLOv10": v10_precision,
    }
    st.bar_chart(precision_data)

    st.markdown("---")

    # Recall bar chart
    st.subheader("Recall Comparison Chart")
    recall_data = {
        "YOLOv5" : v5_recall,
        "YOLOv8" : v8_recall,
        "YOLOv9" : v9_recall,
        "YOLOv10": v10_recall,
    }
    st.bar_chart(recall_data)

    st.markdown("---")

    # Best model summary
    st.subheader("Summary")
    st.write("Best overall model (mAP@0.5): YOLOv5 — 0.989")
    st.write("Best precision: YOLOv8 — 0.989")
    st.write("Best recall: YOLOv5 and YOLOv8 — 0.996")
    st.write("Subteam A models significantly outperformed Subteam B due to a simpler, more consistent dataset.")
