"""Small inference helpers that can be tested without model weights."""
from pathlib import Path
from collections import Counter

MODEL_FILES = {
    "YOLOv5 (Subteam A)": "yolov5.pt",
    "YOLOv8 (Subteam A)": "yolov8.pt",
    "YOLOv9 (Subteam B)": "yolov9.pt",
    "YOLOv10 (Subteam B)": "yolov10.pt",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def available_models(directory):
    """Return only locally supplied model files; never substitute pretrained weights."""
    root = Path(directory)
    return {label: root / filename for label, filename in MODEL_FILES.items()
            if (root / filename).is_file()}


def detection_counts(boxes, names):
    """Count predicted classes without assuming a fixed class ordering."""
    counts = Counter()
    for box in boxes:
        counts[names[int(box.cls[0])]] += 1
    return dict(sorted(counts.items()))


def read_image(upload):
    """Decode a bounded upload, normalize orientation, and return RGB pixels."""
    from PIL import Image, ImageOps, UnidentifiedImageError
    import warnings

    data = upload.getvalue()
    if not data:
        raise ValueError("Choose a non-empty image.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose an image smaller than 10 MB.")
    from io import BytesIO
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Choose an image with at most 20 million pixels.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("This image cannot be decoded safely. Try another JPG or PNG.") from exc
