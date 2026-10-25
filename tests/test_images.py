from io import BytesIO
import unittest
from unittest.mock import patch
from PIL import Image
from app_helpers import read_image


class ImageInputTests(unittest.TestCase):
    def test_grayscale_is_normalized_to_rgb(self):
        upload = BytesIO()
        Image.new("L", (4, 3)).save(upload, format="PNG")
        image = read_image(upload)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (4, 3))

    def test_invalid_image_has_actionable_error(self):
        with self.assertRaises(ValueError):
            read_image(BytesIO(b"not an image"))

    def test_empty_image_is_rejected(self):
        with self.assertRaises(ValueError):
            read_image(BytesIO())

    def test_byte_limit_is_checked(self):
        with patch("app_helpers.MAX_UPLOAD_BYTES", 2):
            with self.assertRaises(ValueError):
                read_image(BytesIO(b"123"))

    def test_pixel_limit_is_checked(self):
        upload = BytesIO()
        Image.new("RGB", (4, 3)).save(upload, format="PNG")
        with patch("app_helpers.MAX_IMAGE_PIXELS", 10):
            with self.assertRaises(ValueError):
                read_image(upload)


if __name__ == "__main__":
    unittest.main()
