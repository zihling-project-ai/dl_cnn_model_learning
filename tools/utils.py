# utils.py：讀取設定、建立優化器與保存實驗紀錄。

import copy
import hashlib
import json
import math
import os
import warnings
from pathlib import Path

import yaml

# utils.py 在 tool 裡，往上一層就是專案根目錄。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 第一組：讀取與檢查設定


def load_config(path=None):
    """讀取 YAML 設定，並整理圖片與輸出的完整路徑。"""

    # 1. 沒有指定檔案時，使用專案根目錄的 config.yaml。
    if path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(path)

    # 相對路徑一律從專案根目錄開始找。
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 整理訓練、驗證、測試照片的路徑。
    for key in ("train_dir", "val_dir", "test_dir"):
        directory = Path(config["dataset"][key])
        if not directory.is_absolute():
            directory = PROJECT_ROOT / directory
        config["dataset"][key] = str(directory)

    # 3. 沒有 save 區塊時，先建立一個空字典。
    if "save" not in config:
        config["save"] = {}

    # 4. 沒有指定輸出位置時，使用預留的 Mac 實驗資料夾。
    default_save_dirs = {
        "checkpoint_dir": "weights/macos_v1",
        "log_dir": "logs/macos_v1",
    }
    for key, default_directory in default_save_dirs.items():
        directory = Path(config["save"].get(key, default_directory))
        if not directory.is_absolute():
            directory = PROJECT_ROOT / directory
        config["save"][key] = str(directory)

    return config


def validate_config(config):
    """檢查設定數值、開關與資料路徑是否合理。"""
    model = config["model"]
    training = config["training"]

    # 1. 圖片尺寸必須有高度、寬度，而且都是至少 32 的整數。 這是基本檢查；個別模型可能還需要更大的尺寸。
    image_size = model["input_size"]
    if len(image_size) != 2:
        raise ValueError("input_size 必須是兩個至少 32 的整數；個別模型仍可能有更高下限。")
    for value in image_size:
        if type(value) is not int or value < 32:
            raise ValueError("input_size 必須是兩個至少 32 的整數；個別模型仍可能有更高下限。")

    # 2. 類別數、每批圖片數、訓練輪數都必須是正整數。
    positive_settings = {
        "num_classes": model["num_classes"],
        "batch_size": training["batch_size"],
        "epochs": training["epochs"],
    }
    for name, value in positive_settings.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} 必須是正整數")
    if model["num_classes"] < 2:
        raise ValueError("分類至少需要兩個類別")

    # 3. 學習率要大於 0，也不能是 NaN 或無限大。
    learning_rate = training["learning_rate"]
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate 必須是有限正數")

    # 4. 亂數種子必須是非負整數。
    seed = training.get("seed", 42)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed 必須是非負整數")

    # 5. 訓練、驗證、測試資料夾不可相同，也不可互相包含。
    directories = []
    for key in ("train_dir", "val_dir", "test_dir"):
        directory = Path(config["dataset"][key]).resolve()
        directories.append(directory)

    for index, directory in enumerate(directories):
        remaining_directories = directories[index + 1:]
        for other_directory in remaining_directories:
            if directory == other_directory:
                raise ValueError("train / val / test 資料夾必須互相獨立，不可相同或包含彼此")
            if directory in other_directory.parents or other_directory in directory.parents:
                raise ValueError("train / val / test 資料夾必須互相獨立，不可相同或包含彼此")

    # 6. 開關必須填 true 或 false，不能用字串 "false" 代替。
    augmentation = config.get("augmentation", {})
    for key in ("use_aug", "flip", "color_jitter"):
        if key in augmentation:
            if type(augmentation[key]) is not bool:
                raise ValueError(f"{key} 必須填 true 或 false")
    if type(model.get("aux_logits", False)) is not bool:
        raise ValueError("aux_logits 必須填 true 或 false")


# 第二組：建立學習工具


def build_optimizer(training):
    """依設定建立優化器，不開始訓練。"""
    # 只有需要建立優化器時才載入 TensorFlow。
    import tensorflow as tf

    name = training.get("optimizer", "adam").lower()
    learning_rate = training["learning_rate"]
    weight_decay = training.get("weight_decay", 0.0)
    momentum = training.get("momentum", 0.0)

    # 先檢查設定，避免填錯名稱或不合理的數值。
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay 必須是有限且非負的數值")
    if name not in {"adam", "sgd", "adamw", "rmsprop"}:
        raise ValueError(f"不支援的 optimizer：{name}")
    if name in {"adam", "rmsprop"} and weight_decay:
        warnings.warn("weight_decay 僅套用於 SGD / AdamW；目前選擇的優化器不套用此值。")

    # 1. Adam。
    if name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        return optimizer

    # 2. RMSprop：保留目前 rho、epsilon 與 momentum 的設定方式。
    if name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            momentum=momentum,
            rho=0.9,
            epsilon=0.001,
        )
        return optimizer

    # 3. SGD：weight_decay 非零時才傳入，舊版不支援就明確報錯。
    if name == "sgd":
        optimizer_settings = {
            "learning_rate": learning_rate,
            "momentum": momentum,
        }
        if weight_decay:
            optimizer_settings["weight_decay"] = weight_decay
        try:
            optimizer = tf.keras.optimizers.SGD(**optimizer_settings)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "無法建立 SGD；若舊版 TensorFlow 不支援 weight_decay，請使用支援版本，"
                "或明確將 weight_decay 設為 0。原始錯誤：" + str(error)
            ) from error
        return optimizer

    # 4. AdamW：不同 TensorFlow 版本可能把它放在不同位置。 getattr 的第三個參數 None 表示：找不到這個名稱時，先回傳 None。
    optimizer_class = getattr(tf.keras.optimizers, "AdamW", None)
    if optimizer_class is None:
        experimental = getattr(tf.keras.optimizers, "experimental", None)
        optimizer_class = getattr(experimental, "AdamW", None)
    if optimizer_class is None:
        raise ValueError("目前 TensorFlow 不支援 AdamW，請使用支援版本或明確改選其他優化器。")

    optimizer = optimizer_class(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return optimizer


# 第三組：保存與核對實驗


def metadata_path(weight_path):
    """取得權重旁的 JSON 紀錄檔路徑。"""
    filename = str(weight_path) + ".json"
    return Path(filename)


def code_fingerprints():
    """計算模型與讀圖程式的 SHA-256，供核對版本使用。"""
    source = PROJECT_ROOT

    files = sorted((source / "models").glob("*.py"))
    files.append(source / "tools" / "dataloader.py")

    fingerprints = {}
    for file_path in files:
        # 略過 Mac 產生的 ._ 隱藏中繼資料檔。
        if file_path.name.startswith("._"):
            continue
        relative_name = str(file_path.relative_to(source))
        file_content = file_path.read_bytes()
        fingerprints[relative_name] = hashlib.sha256(file_content).hexdigest()

    return fingerprints


def make_run_metadata(config, class_indices):
    """整理當次設定、類別、版本與程式雜湊。"""
    import tensorflow as tf

    # 複製設定，避免之後修改原字典時，連帶改掉這份紀錄。
    saved_config = copy.deepcopy(config)
    saved_class_indices = dict(class_indices)
    preprocessing = {
        "rescale": 1./255,
        "color_mode": "rgb",
        "interpolation": "nearest",
    }

    metadata = {
        "format_version": 1,
        "config": saved_config,
        "class_indices": saved_class_indices,
        "preprocessing": preprocessing,
        "tensorflow_version": tf.__version__,
        "code_sha256": code_fingerprints(),
    }
    return metadata


def validate_run_metadata(config, metadata):
    """核對目前條件與權重紀錄，回傳保存的類別順序。"""

    # 1. 確認紀錄格式與圖片前處理方式。
    expected_preprocessing = {
        "rescale": 1./255,
        "color_mode": "rgb",
        "interpolation": "nearest",
    }
    if metadata.get("format_version") != 1:
        raise ValueError("權重的資料處理紀錄格式不相容，請先確認原始設定。")
    if metadata.get("preprocessing") != expected_preprocessing:
        raise ValueError("權重的資料處理紀錄格式不相容，請先確認原始設定。")

    # 2. 模型名稱、類別數與輸入尺寸必須和紀錄相同。
    saved_model = metadata["config"]["model"]
    current_model = config["model"]
    for key in ("name", "num_classes", "input_size"):
        if saved_model[key] != current_model[key]:
            raise ValueError(f"目前 {key} 與權重紀錄不同，請使用當次訓練設定。")

    # GoogLeNet 還需要確認是否使用輔助分類輸出。
    if current_model["name"].lower() == "googlenet":
        saved_aux = saved_model.get("aux_logits", False)
        current_aux = current_model.get("aux_logits", False)
        if saved_aux != current_aux:
            raise ValueError("aux_logits 與權重紀錄不同")

    # 3. 類別編號應從 0 開始，連續到「類別數減 1」。
    class_indices = metadata["class_indices"]
    saved_numbers = sorted(class_indices.values())
    expected_numbers = list(range(current_model["num_classes"]))
    if saved_numbers != expected_numbers:
        raise ValueError("權重的類別編號紀錄不完整")

    # 4. 保存時記錄全部模型，但評分只核對這次使用的模型與共用入口。
    # 例如只修 EfficientNet，不應讓完全未修改的 LeNet 舊權重失效。
    model_files = {
        "lenet": "models/lenet.py",
        "alexnet": "models/alexnet.py",
        "alexnet_lrn": "models/alexnet_lrn.py",
        "vgg16": "models/vgg16.py",
        "vgg16_bn": "models/vgg16_bn.py",
        "googlenet": "models/googlenet.py",
        "inception": "models/inception.py",
        "resnet50": "models/resnet.py",
        "mobilenet_v1": "models/mobilenet_v1.py",
        "efficientnet_b0": "models/efficientnet.py",
    }
    model_name = current_model["name"].lower()
    if model_name not in model_files:
        raise ValueError(f"找不到模型版本對照：{model_name}")
    saved_hashes = metadata.get("code_sha256")
    current_hashes = code_fingerprints()
    error_message = "模型或資料讀取程式已變更，請先對照當次版本，不直接套用成績。"
    # 不接受缺漏或來路不明的檔名；舊紀錄不會被改寫成新版。
    if not isinstance(saved_hashes, dict) or set(saved_hashes) != set(current_hashes):
        raise ValueError(error_message)
    required_files = [model_files[model_name], "models/__init__.py", "tools/dataloader.py"]
    for filename in required_files:
        if saved_hashes[filename] != current_hashes[filename]:
            raise ValueError(error_message)

    # 5. 依編號由小到大，回傳類別名稱，供測試資料讀取使用。
    sorted_class_names = sorted(class_indices, key=class_indices.get)
    class_names = []
    for class_name in sorted_class_names:
        class_names.append(class_name)
    return class_names


def create_run_files(config, class_indices):
    """建立新的實驗紀錄與空白 CSV，既有檔案不覆寫。"""

    # 1. 決定最佳權重、逐輪成績與設定紀錄的檔名。
    model_name = config["model"]["name"]
    checkpoint_dir = Path(config["save"]["checkpoint_dir"])
    log_dir = Path(config["save"]["log_dir"])

    weight_path = checkpoint_dir / f"{model_name}_best.h5"
    history_path = log_dir / f"{model_name}_history.csv"
    metadata_file = metadata_path(weight_path)

    # 2. 任何一個同名檔案已存在，就先停止，不覆寫舊實驗。
    for file_path in (weight_path, metadata_file, history_path):
        if os.path.lexists(file_path):
            raise FileExistsError("權重、設定或訓練紀錄已存在，請指定新的 checkpoint_dir / log_dir。")

    # 3. 先整理好可儲存的紀錄，再建立輸出資料夾。
    metadata = make_run_metadata(config, class_indices)
    content = json.dumps(metadata, ensure_ascii=False, indent=2)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. 用 x 模式建立新檔案；即使檢查後才出現同名檔，也不覆寫。 記住這次建立的檔案，若中途失敗，只清除這次新建的檔案。
    created_files = []
    try:
        with metadata_file.open("x", encoding="utf-8") as f:
            created_files.append(metadata_file)
            f.write(content)
        with history_path.open("x", encoding="utf-8"):
            created_files.append(history_path)
    except BaseException:
        for file_path in created_files:
            file_path.unlink()
        raise

    # 這裡尚未建立 .h5 權重，真正訓練時才由 ModelCheckpoint 儲存。
    return weight_path, history_path
