"""兩個修正版模型的結構、遮罩與版本保護檢查；不執行訓練更新。"""
import copy
import unittest

import numpy as np
import tensorflow as tf

from models import get_model
from models.efficientnet import SE_Block
from tools.utils import make_run_metadata, validate_run_metadata


class ModelArchitectureTests(unittest.TestCase):
    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_inception_modules_and_six_class_output(self):
        model = get_model('inception', 6, (160, 160, 3))
        expected = {
            'Mixed_3b': (20, 20, 256), 'Mixed_3c': (20, 20, 320),
            'Mixed_4a': (10, 10, 576), 'Mixed_4b': (10, 10, 576),
            'Mixed_4c': (10, 10, 576), 'Mixed_4d': (10, 10, 576),
            'Mixed_4e': (10, 10, 576), 'Mixed_5a': (5, 5, 1024),
            'Mixed_5b': (5, 5, 1024), 'Mixed_5c': (5, 5, 1024),
        }
        for name, shape in expected.items():
            self.assertEqual(tuple(model.get_layer(name).output.shape[1:]), shape)
        self.assertIsInstance(model.layers[1], tf.keras.layers.SeparableConv2D)
        self.assertEqual(model.layers[1].depth_multiplier, 8)
        probabilities = model(tf.zeros((2, 160, 160, 3)), training=False).numpy()
        self.assertEqual(probabilities.shape, (2, 6))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        self.assertEqual(model.count_params(), 10189342)

    def test_se_uses_unexpanded_channels(self):
        inputs = tf.keras.Input((4, 4, 144))
        model = tf.keras.Model(inputs, SE_Block(inputs, in_filters=24))
        widths = [layer.filters for layer in model.layers
                  if isinstance(layer, tf.keras.layers.Conv2D)]
        self.assertEqual(widths, [6, 144])
        probabilities = model(tf.ones((2, 4, 4, 144)), training=False).numpy()
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))

    def test_efficientnet_branch_masks_and_inference(self):
        model = get_model('efficientnet_b0', 6, (160, 160, 3))
        masks = [layer for layer in model.layers
                 if isinstance(layer, tf.keras.layers.Dropout) and layer.noise_shape]
        self.assertEqual(len(masks), 9)
        self.assertTrue(all(layer.noise_shape == (None, 1, 1, 1) for layer in masks))
        self.assertEqual(model.count_params(), 4057250)
        # 用模型實際建立的遮罩層，確認同張圖片共用遮罩且推論不丟棄。
        layer = max(masks, key=lambda item: item.rate)
        ones = tf.ones((128, 2, 3, 4))
        output = layer(ones, training=True).numpy()
        self.assertTrue(np.all(output == output[:, :1, :1, :1]))
        self.assertTrue(np.any(output == 0))
        self.assertTrue(np.any(output > 0))
        np.testing.assert_array_equal(layer(ones, training=False), ones)
        np.testing.assert_allclose(output[output > 0], 1 / (1 - layer.rate), rtol=1e-6)

    def test_version_check_only_ignores_unrelated_model_changes(self):
        config = {'model': {'name': 'lenet', 'num_classes': 2, 'input_size': [32, 32]}}
        metadata = make_run_metadata(config, {'cat': 0, 'dog': 1})
        unrelated = copy.deepcopy(metadata)
        unrelated['code_sha256']['models/efficientnet.py'] = 'previous version'
        self.assertEqual(validate_run_metadata(config, unrelated), ['cat', 'dog'])
        for filename in ['models/lenet.py', 'models/__init__.py', 'tools/dataloader.py']:
            changed = copy.deepcopy(metadata)
            changed['code_sha256'][filename] = 'previous version'
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                validate_run_metadata(config, changed)
        config['model']['name'] = 'efficientnet_b0'
        old = make_run_metadata(config, {'cat': 0, 'dog': 1})
        old['code_sha256']['models/efficientnet.py'] = 'previous version'
        with self.assertRaises(ValueError):
            validate_run_metadata(config, old)


if __name__ == '__main__':
    unittest.main()
