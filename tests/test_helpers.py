import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from app_helpers import available_models, detection_counts


class InferenceHelperTests(unittest.TestCase):
    def test_only_existing_expected_weight_files_are_offered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "yolov8.pt").touch()
            (root / "unrelated.pt").touch()
            (root / "yolov5.pt").mkdir()
            self.assertEqual(available_models(root),
                             {"YOLOv8 (Subteam A)": root / "yolov8.pt"})

    def test_empty_directory_offers_no_fallback_model(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(available_models(directory), {})

    def test_counts_use_each_models_class_mapping(self):
        boxes = [SimpleNamespace(cls=[2]), SimpleNamespace(cls=[0]),
                 SimpleNamespace(cls=[2])]
        self.assertEqual(detection_counts(boxes, {0: "Orange", 2: "Apple"}),
                         {"Apple": 2, "Orange": 1})

    def test_no_detections(self):
        self.assertEqual(detection_counts([], {}), {})


if __name__ == "__main__":
    unittest.main()
