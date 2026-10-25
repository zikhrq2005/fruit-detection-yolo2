"""Streamlit interface for locally supplied, trained fruit detectors."""
from pathlib import Path
import json
import streamlit as st
from ultralytics import YOLO
from app_helpers import available_models, detection_counts, read_image

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Fruit Detection and Counting", layout="wide")

@st.cache_resource
def load_model(path, modified_ns):
    # Including modification time invalidates the cache when weights are replaced.
    return YOLO(path)


def show_detection(label, path, image, confidence):
    st.subheader(label)
    try:
        model = load_model(str(path), path.stat().st_mtime_ns)
        with st.spinner("Detecting objects..."):
            result = model.predict(image, conf=confidence, verbose=False)[0]
        st.image(result.plot()[:, :, ::-1], use_container_width=True)
        counts = detection_counts(result.boxes, model.names)
        st.metric("Objects detected", sum(counts.values()))
        if counts:
            st.dataframe([{"Class": name, "Count": count}
                          for name, count in counts.items()], hide_index=True)
        else:
            st.info("No objects detected at this confidence threshold.")
    except Exception:
        st.error("Detection failed. Check that these trained weights are compatible "
                 "with the installed Ultralytics version and try a smaller image.")


st.sidebar.title("Detection settings")
page = st.sidebar.selectbox("Page", ["Overview", "Single model", "Compare models",
                                    "Reported validation metrics"])
confidence = st.sidebar.slider("Confidence threshold", 0.01, 1.0, 0.25, 0.01)
models = available_models(ROOT)

if page == "Overview":
    st.title("Fruit Detection and Counting")
    st.write("Group computer vision coursework at The British University in Egypt.")
    st.write("Explore image processing notebooks and run detection with locally supplied "
             "trained YOLO weights.")
    st.markdown("**Subteam A:** YOLOv5 / YOLOv8; Apple, Tomato, Cauliflower, Adrak, "
                "Bitter Gourd, Capsicum.\n\n"
                "**Subteam B:** YOLOv9 / YOLOv10; Apple, Banana, Grapes, Mango, "
                "Orange, Strawberry.")
    st.warning("The subteams used different datasets and class sets. Side-by-side "
               "predictions are illustrative; they do not establish an architecture ranking.")
    if models:
        st.success(f"{len(models)} trained model file(s) available.")
    else:
        st.info("No trained weights are installed. Follow the README to train Subteam A "
                "models and place yolov5.pt or yolov8.pt beside app.py.")
    st.caption("The repository does not include trained weights or a hosted demo.")

elif page == "Reported validation metrics":
    st.title("Reported validation metrics")
    st.warning("Stored coursework results, not a live evaluation. Datasets and class "
               "sets differ between subteams; compare within each dataset only.")
    try:
        rows = json.loads((ROOT / "reported_metrics.json").read_text(encoding="utf-8"))
        for team in ("A", "B"):
            st.subheader(f"Subteam {team}")
            st.dataframe([row for row in rows if row["Subteam"] == team],
                         hide_index=True)
        st.caption("The committed training notebook has no saved run outputs. "
                   "Evaluation logs and model artifacts are needed to verify these figures.")
    except (OSError, ValueError):
        st.error("The reported metrics file is missing or invalid.")

else:
    st.title("Single model detection" if page == "Single model" else "Compare models")
    if not models:
        st.info("Detection needs trained weights. Add the files described in the README "
                "beside app.py, then refresh this page.")
        st.stop()
    if page == "Single model":
        selected = [st.selectbox("Available model", list(models))]
    else:
        st.caption("Class sets differ across subteams. Use a shared labeled evaluation "
                   "dataset for quantitative comparisons.")
        selected = st.multiselect("Available models", list(models), default=list(models)[:2])
    upload = st.file_uploader("Upload a JPG or PNG (up to 10 MB)",
                              type=["jpg", "jpeg", "png"])
    if upload is not None:
        try:
            image = read_image(upload)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        st.image(image, caption="Uploaded image", width=400)
        if st.button("Run detection", disabled=not selected):
            columns = st.columns(min(len(selected), 2))
            for index, label in enumerate(selected):
                with columns[index % len(columns)]:
                    show_detection(label, models[label], image, confidence)
