# Fruit Detection and Counting

Group coursework for **Visual Computing and Image Processing — 25CSAI02I**, The British University in Egypt.

Image processing and YOLO training notebooks accompany a Streamlit app that displays bounding boxes and counts detections by class.

## Project overview

| Subteam | Dataset | Classes | Models |
|---|---|---|---|
| A | [fruit-nxoac](https://universe.roboflow.com/fruit-detection-2aajq/fruit-nxoac), version 4; reported 825 images | Apple, Tomato, Cauliflower, Adrak, Bitter Gourd, Capsicum | YOLOv5, YOLOv8 |
| B | fruitsdetection/fruits-by-yolo; reported 1,176 images | Apple, Banana, Grapes, Mango, Orange, Strawberry | YOLOv9, YOLOv10 |

**The datasets and class sets differ. Their metrics do not establish an overall ranking of model architectures.**

This repository contains group work. Individual contributions have not yet been itemized.

## What's included

| File | Purpose |
|---|---|
| `Preprocessing.ipynb` | Color spaces, resizing, filtering, thresholding, histograms, sharpening |
| `Augmentation.ipynb` | Translation, rotation, flipping, cropping, brightness/contrast, gamma |
| `Training.ipynb` | Subteam A training, prediction examples, and validation |
| `app.py` | Streamlit UI for single-model detection and side-by-side predictions |
| `app_helpers.py` | Model file discovery, image validation, and class counting |
| `reported_metrics.json` | Stored coursework validation scores, separated by subteam |
| `tests/` | Model discovery, counting, and image-input checks |

## Run the app

Use Python 3.11 or 3.12 as the target environment. Create and activate a virtual environment, then run:

```sh
python -m pip install -r requirements.txt
streamlit run app.py
```

The overview and stored metrics pages work without weights. Detection requires locally supplied trained weights beside `app.py`:

- `yolov5.pt` and `yolov8.pt`: generated from Subteam A training.
- `yolov9.pt` and `yolov10.pt`: obtain the corresponding trained artifacts from Subteam B.

Only model files that exist are offered for inference. The app does not substitute general-purpose weights for a missing fruit detector. Use trusted model files from a known training run.

Upload a JPG or PNG, select available models, adjust confidence, and click **Run detection**. Uploads are limited to 10 MB and 20 million pixels; decoded images are normalized to RGB and EXIF orientation is applied.

## Train Subteam A models

Install the training dependencies in the same environment:

```sh
python -m pip install -r requirements-training.txt
```

Configure a suitable PyTorch build for your CPU/CUDA environment. Set your own Roboflow key locally before starting Jupyter.

PowerShell:

```powershell
$env:ROBOFLOW_API_KEY = "your-local-key"
jupyter notebook
```

macOS/Linux:

```sh
export ROBOFLOW_API_KEY="your-local-key"
jupyter notebook
```

Never commit a populated key. Run `Training.ipynb` in order. It downloads dataset version 4, trains with 640-pixel images for up to 50 epochs and early-stopping patience 10, and uses batch sizes 16 (v5) and 8 (v8).

Copy the actual best weights from each training run to the filenames expected by the app. The notebook expects:

```text
runs/detect/fruit_yolov5/weights/best.pt -> yolov5.pt
runs/detect/fruit_yolov8/weights/best.pt -> yolov8.pt
```

Repeated runs can receive suffixed output directories; check the path printed by training.

## Reported validation results

| Subteam | Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|---|
| A | YOLOv5 | 0.989 | 0.900 | 0.988 | 0.996 |
| A | YOLOv8 | 0.988 | 0.885 | 0.989 | 0.996 |
| B | YOLOv9 | 0.740 | 0.614 | 0.707 | 0.706 |
| B | YOLOv10 | 0.709 | 0.567 | 0.739 | 0.660 |

These stored values come from the coursework documentation. The committed training notebook has no saved execution outputs; run logs and model artifacts are still needed to verify the figures. The metrics page reads stored results rather than evaluating uploaded images.

The notebook calls `model.val(data=yamlPath)` without explicitly selecting the test split. These figures should not be described as held-out test scores. Viewing predictions on three test images is a qualitative check, not a test-set evaluation.

## Preprocessing example

![Comparison of image resizing methods](step3_resizing.png)

This figure demonstrates preprocessing, not trained detection performance.

## Checks

```sh
python -m unittest discover -s tests -v
python -m compileall -q app.py app_helpers.py
```

The tests need Pillow but no model weights, GPU, or dataset download. GitHub Actions runs these checks on Python 3.11 and 3.12. It does not train models or certify inference accuracy.

## Limitations and next steps

- Dependency ranges are provided, but a clean full installation and model inference have not been verified. They are not a tested lockfile.
- Trained artifacts and a hosted demo are not included; only Subteam A training is provided.
- Publish training logs, split sizes, per-class results, counting errors, and difficult examples.
- Check duplicate/near-duplicate images across splits and document annotation quality.
- Compare architectures on a common dataset and evaluation protocol.
- Record latency and hardware alongside detection metrics.
