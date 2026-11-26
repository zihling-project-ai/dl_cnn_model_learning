"""不做訓練的檢查。只使用暫存圖片、臨時模型與臨時權重。"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import tensorflow as tf
import yaml

from tools.dataloader import load_train_val_dataset, load_test_dataset
from tools.data_safety import map_imagenet_label
from main_test import evaluate_batches, main as evaluate_main
from main_train import main as train_main
from models import get_model
from tools.utils import (validate_config, make_run_metadata, validate_run_metadata,
                   create_run_files, metadata_path)


class TrainingSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for split in ("train", "val", "test"):
            for label in ("cat", "dog"):
                folder = self.root / split / label
                folder.mkdir(parents=True)
                Image.fromarray(np.full((32, 32, 3), 80, dtype=np.uint8)).save(folder / "one.png")
        self.cfg = {
            "model": {"name": "lenet", "num_classes": 2, "input_size": [32, 32], "aux_logits": False},
            "training": {"batch_size": 2, "epochs": 1, "learning_rate": .001, "optimizer": "adam", "seed": 42},
            "augmentation": {"use_aug": False},
            "dataset": {name + "_dir": str(self.root / name) for name in ("train", "val", "test")},
            "save": {"checkpoint_dir": str(self.root / "weights"), "log_dir": str(self.root / "logs")},
        }
        tf.keras.utils.set_random_seed(42)
        self.addCleanup(tf.keras.backend.clear_session)

    def datasets(self):
        return load_train_val_dataset(self.root / "train", self.root / "val", 2, [32, 32],
                                      augmentation={"use_aug": False}, seed=42)

    def test_class_indices_shared_and_augmentation_off(self):
        train, val = self.datasets()
        self.assertEqual(train.class_indices, val.class_indices)
        self.assertEqual(train.class_indices, {"cat": 0, "dog": 1})
        self.assertFalse(val.shuffle)
        self.assertEqual(train.image_data_generator.rotation_range, 0)
        x, y = train[0]
        self.assertEqual(x.shape, (2, 32, 32, 3))
        self.assertTrue(np.allclose(x, 80 / 255))

    def test_validation_missing_class_rejected(self):
        (self.root / "val/dog/one.png").unlink()
        (self.root / "val/dog").rmdir()
        with self.assertRaisesRegex(ValueError, "類別不一致"):
            self.datasets()

    def test_empty_class_rejected(self):
        (self.root / "val/dog/one.png").unlink()
        with self.assertRaisesRegex(ValueError, "沒有可讀取"):
            self.datasets()

    def test_same_train_val_directory_rejected(self):
        with self.assertRaises(ValueError):
            load_train_val_dataset(self.root / "train", self.root / "train", 2, [32, 32])

    def test_test_set_respects_saved_class_order(self):
        ds = load_test_dataset(self.root / "test", [32, 32], 2, class_names=["dog", "cat"])
        self.assertEqual(ds.class_indices, {"dog": 0, "cat": 1})

    def test_seed_repeats_augmentation(self):
        # 使用有紋理的圖片，避免純色讓變換看起來相同。
        image = np.arange(32*32*3, dtype=np.uint16).reshape(32,32,3).astype(np.uint8)
        for label in ("cat", "dog"):
            Image.fromarray(image).save(self.root / "train" / label / "one.png")
        args = (self.root / "train", self.root / "val", 2, [32,32])
        a,_=load_train_val_dataset(*args, augmentation={"use_aug":True,"rotate":10,"flip":True}, seed=17)
        b,_=load_train_val_dataset(*args, augmentation={"use_aug":True,"rotate":10,"flip":True}, seed=17)
        ax,ay=a[0];bx,by=b[0]
        np.testing.assert_array_equal(ay,by)
        np.testing.assert_allclose(ax,bx)

    def test_metadata_roundtrip_and_collision_protection(self):
        cfg = copy.deepcopy(self.cfg)
        weight, history = create_run_files(cfg, {"cat":0,"dog":1})
        metadata=json.loads(metadata_path(weight).read_text())
        self.assertEqual(validate_run_metadata(cfg,metadata),["cat","dog"])
        cfg["model"]["input_size"]=[64,64]
        with self.assertRaisesRegex(ValueError,"input_size"):
            validate_run_metadata(cfg,metadata)
        before=history.read_bytes()
        with self.assertRaises(FileExistsError): create_run_files(self.cfg,{"cat":0,"dog":1})
        self.assertEqual(history.read_bytes(),before)

    def test_metadata_rejects_changed_code_and_invalid_mapping(self):
        metadata=make_run_metadata(self.cfg,{"cat":0,"dog":1})
        metadata["code_sha256"]["lenet.py"]="changed"
        with self.assertRaisesRegex(ValueError,"程式已變更"):
            validate_run_metadata(self.cfg,metadata)
        metadata=make_run_metadata(self.cfg,{"cat":0,"dog":0})
        with self.assertRaisesRegex(ValueError,"類別編號"):
            validate_run_metadata(self.cfg,metadata)

    def test_unknown_legacy_metadata_not_silently_evaluated(self):
        config=self.root/"config.yaml";config.write_text(yaml.safe_dump(self.cfg))
        fake_weight=self.root/"legacy.h5";fake_weight.write_bytes(b"not a real weight")
        with self.assertRaisesRegex(ValueError,"--allow-legacy"):
            evaluate_main(config, fake_weight)

    def test_incorrect_num_classes_stops_before_fit(self):
        self.cfg["model"]["num_classes"]=3
        config=self.root/"config.yaml";config.write_text(yaml.safe_dump(self.cfg))
        with patch.object(tf.keras.Model,"fit",side_effect=AssertionError("training forbidden")) as fit:
            with self.assertRaisesRegex(ValueError,"類別數"):
                train_main(config)
            fit.assert_not_called()
        self.assertFalse((self.root/"weights").exists())

    def test_reporting_keeps_zero_prediction_classes(self):
        ds=load_test_dataset(self.root/"test",[32,32],2)
        class ConstantModel:
            def __call__(self,x,training=False):
                return {"main":np.tile([[1.,0.]],(len(x),1))}
        result=evaluate_batches(ConstantModel(),ds)
        self.assertEqual(result["accuracy"],.5)
        self.assertEqual(result["report"]["dog"]["recall"],0)
        self.assertEqual(result["report"]["dog"]["f1-score"],0)
        self.assertEqual(result["confusion_matrix"],[[1,0],[1,0]])

    def test_invalid_probability_rejected(self):
        ds=load_test_dataset(self.root/"test",[32,32],2)
        class BadModel:
            def __call__(self,x,training=False):return np.full((len(x),2),np.nan)
        with self.assertRaises(ValueError): evaluate_batches(BadModel(),ds)

    def test_label_mapping_excludes_false_matches(self):
        for label in ("sea lion","lionfish","wolf spider","Irish Wolfhound","English foxhound","fox squirrel"):
            self.assertIsNone(map_imagenet_label(label))
        for label,expected in [("lion","lion"),("grey wolf","wolf"),("Arctic fox","fox"),("jaguar","leopard")]:
            self.assertEqual(map_imagenet_label(label),expected)

    def test_invalid_config_stops(self):
        for section,key,value in [("training","learning_rate",float('nan')),
                                  ("training","epochs",0),("training","seed",-1),
                                  ("model","num_classes",1),("model","aux_logits","false")]:
            with self.subTest(key=key):
                cfg=copy.deepcopy(self.cfg);cfg[section][key]=value
                with self.assertRaises(ValueError): validate_config(cfg)

    def test_unsupported_image_extension_is_reported(self):
        (self.root / "train/cat/extra.jpg!d").write_bytes(b"not read by Keras")
        with self.assertWarnsRegex(UserWarning, "不支援的檔案被略過"):
            self.datasets()

    def test_train_test_overlap_rejected(self):
        self.cfg["dataset"]["test_dir"] = self.cfg["dataset"]["train_dir"]
        with self.assertRaisesRegex(ValueError, "互相獨立"):
            validate_config(self.cfg)

    def test_checkpoint_and_early_stopping_choose_same_best_epoch(self):
        # 模擬 callback 收到的指標與權重，不呼叫 fit 或任何優化器更新。
        x=tf.keras.Input((1,));model=tf.keras.Model(x,tf.keras.layers.Dense(2)(x))
        # 從真正的主流程取得 callbacks，但在 fit 執行前攔截，禁止訓練。
        config = self.root / "config.yaml"
        config.write_text(yaml.safe_dump(self.cfg))
        captured = {}
        class StopBeforeTraining(Exception):
            pass
        def stop_before_training(training_model, *args, **kwargs):
            captured["callbacks"] = kwargs["callbacks"]
            raise StopBeforeTraining()
        with patch.object(tf.keras.Model, "fit", new=stop_before_training):
            with self.assertRaises(StopBeforeTraining):
                train_main(config)
        weight = Path(self.cfg["save"]["checkpoint_dir"]) / "lenet_best.h5"
        history = Path(self.cfg["save"]["log_dir"]) / "lenet_history.csv"
        callbacks = captured["callbacks"]
        for cb in callbacks: cb.set_model(model);cb.on_train_begin()
        for epoch,acc in enumerate([.5,.505,.509,.509,.509,.509,.509,.509]):
            model.set_weights([np.full_like(v,epoch) for v in model.get_weights()])
            for cb in callbacks: cb.on_epoch_end(epoch,{"loss":1.,"val_accuracy":acc})
            if model.stop_training:break
        expected=callbacks[1].best_weights
        model.load_weights(str(weight))
        for a,b in zip(expected,model.get_weights()):np.testing.assert_array_equal(a,b)
        for cb in callbacks:cb.on_train_end()
        self.assertTrue(history.is_file())


if __name__=="__main__": unittest.main(verbosity=2)
