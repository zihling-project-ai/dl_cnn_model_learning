"""確認套件搬移、匯入安全與 README 的計算範例，不做訓練。"""
import ast
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import tensorflow as tf

class ProjectLayoutTests(unittest.TestCase):
    def test_root_has_only_two_python_entries(self):
        self.assertEqual({p.name for p in ROOT.glob('*.py') if not p.name.startswith('._')},
                         {'main_train.py','main_test.py'})
        self.assertTrue((ROOT/'tools/__init__.py').is_file())
        self.assertTrue((ROOT/'models/__init__.py').is_file())

    def test_all_tools_import_without_training_download_or_cuda_work(self):
        import tensorflow_hub as hub
        modules=['main_train','main_test','tools','tools.data_safety','tools.dataloader',
                 'tools.labeling','tools.labeling_fast','tools.main_fix_filenames','tools.utils',
                 'tools.cuda_check','models']
        with patch.object(hub,'load',side_effect=AssertionError('model download forbidden')) as load, \
             patch.object(tf.keras.Model,'fit',side_effect=AssertionError('training forbidden')) as fit, \
             patch.object(tf,'matmul',side_effect=AssertionError('hardware check must be explicit')) as matmul:
            for name in modules:
                with self.subTest(module=name):importlib.import_module(name)
            load.assert_not_called();fit.assert_not_called();matmul.assert_not_called()

    def test_imagenet_resource_and_fingerprint_paths(self):
        from tools import utils,labeling,labeling_fast
        labels=json.loads((ROOT/'tools/imagenet_labels.json').read_text())
        self.assertEqual(len(labels),1001)
        self.assertEqual(labels[0],"background")
        self.assertIsNone(labeling.model)
        self.assertIsNone(labeling_fast.model)
        fingerprints=utils.code_fingerprints()
        self.assertIn('tools/dataloader.py',fingerprints)
        self.assertIn('models/lenet.py',fingerprints)
        self.assertEqual(len(fingerprints),12)
        for filename in fingerprints:self.assertTrue((ROOT/filename).is_file())

    def test_comment_format_and_function_descriptions(self):
        files=[ROOT/'main_train.py',ROOT/'main_test.py',*sorted((ROOT/'tools').glob('*.py'))]
        for file in files:
            if file.name.startswith('._'):
                continue  # macOS 的附屬資料，不是 Python 程式。
            text=file.read_text()
            self.assertNotRegex(text,r'(?m)^\s*#\s*[-=]{3,}')
            tree=ast.parse(text)
            for node in tree.body:
                if isinstance(node,ast.FunctionDef):
                    doc=ast.get_docstring(node)
                    self.assertTrue(doc,(file,node.name))
                    self.assertNotIn('\n',doc)

    def test_convolution_patch_and_shapes_match_readme(self):
        image=tf.constant([[[[1.],[2.],[0.]],[[0.],[1.],[3.]],[[2.],[2.],[1.]]]])
        kernel=tf.constant([[[[1.]],[[0.]]],[[[0.]],[[1.]]]])
        output=tf.nn.conv2d(image,kernel,strides=1,padding='VALID')
        np.testing.assert_allclose(output.numpy()[0,:,:,0],[[2,5],[2,2]])
        layers=tf.keras.layers
        for padding,stride,expected in [('valid',1,222),('same',1,224),('valid',2,111),('same',2,112)]:
            layer=layers.Conv2D(32,3,strides=stride,padding=padding)
            self.assertEqual(tuple(layer.compute_output_shape((1,224,224,3))),(1,expected,expected,32))
        shape=(1,224,224,3)
        for layer,expected in [
            (layers.Conv2D(32,3,padding='same'),(1,224,224,32)),
            (layers.Conv2D(64,3,strides=2,padding='same'),(1,112,112,64)),
            (layers.DepthwiseConv2D(3,padding='same'),(1,112,112,64)),
            (layers.Conv2D(128,1,padding='same'),(1,112,112,128)),
            (layers.GlobalAveragePooling2D(),(1,128)),
            (layers.Dense(6,activation='softmax'),(1,6))]:
            shape=tuple(layer.compute_output_shape(shape));self.assertEqual(shape,expected)

    def test_se_weight_range_with_unit_input(self):
        from models.efficientnet import SE_Block
        inputs=tf.keras.Input((4,4,8))
        model=tf.keras.Model(inputs,SE_Block(inputs, in_filters=8))
        result=model(tf.ones((1,4,4,8)),training=False).numpy()
        self.assertEqual(result.shape,(1,4,4,8))
        self.assertTrue(np.all(result>=0));self.assertTrue(np.all(result<=1))
        tf.keras.backend.clear_session()

    def test_cli_help_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ['main_train.py','main_test.py','tools/main_fix_filenames.py']:
                run=subprocess.run([sys.executable,'-B',str(ROOT/name),'--help'],cwd=directory,
                    capture_output=True,text=True,timeout=45)
                self.assertEqual(run.returncode,0,run.stderr)
                self.assertIn('usage:',run.stdout)

if __name__=='__main__':unittest.main(verbosity=2)
