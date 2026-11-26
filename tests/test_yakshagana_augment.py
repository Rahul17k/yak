import unittest

import numpy as np
import tensorflow as tf

from yakshagana_augment import apply_blur_distance, blur_distance_views, random_blur_distance


class BlurDistanceAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.image = tf.reshape(
            tf.linspace(0.0, 255.0, 224 * 224 * 3),
            (224, 224, 3),
        )

    def test_fixed_blur_distance_views_preserve_shape_and_valid_pixels(self):
        mild, moderate = blur_distance_views(self.image)
        for result in (mild, moderate):
            self.assertEqual(tuple(result.shape), (224, 224, 3))
            self.assertTrue(np.isfinite(result.numpy()).all())
            self.assertGreaterEqual(float(tf.reduce_min(result)), 0.0)
            self.assertLessEqual(float(tf.reduce_max(result)), 255.0)

    def test_random_augmentation_preserves_shape(self):
        result = random_blur_distance(self.image)
        self.assertEqual(tuple(result.shape), (224, 224, 3))

    def test_full_quality_settings_leave_pixels_unchanged(self):
        result = apply_blur_distance(self.image, 1.0, 1.0, 0.0)
        np.testing.assert_array_equal(result.numpy(), self.image.numpy())


if __name__ == "__main__":
    unittest.main()
