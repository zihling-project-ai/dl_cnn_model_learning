# dataloader.py：準備訓練、驗證與測試圖片。
from pathlib import Path
import warnings

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_train_val_dataset(train_dir, val_dir, batch_size, img_size, augmentation=None, seed=42):
    
    # 1. 確認訓練與驗證資料夾存在，而且不是同一個資料夾。
    """準備訓練與驗證圖片，使用一致的類別編號。"""

    for directory in [train_dir, val_dir]:
        if not Path(directory).is_dir():
            raise ValueError(f"找不到資料夾：{directory}")

    if Path(train_dir).resolve() == Path(val_dir).resolve():
        raise ValueError("訓練與驗證不可使用同一個資料夾")


    # 2. 讀取分類資料夾名稱，例如 cat、dog、fox。
    train_classes = []
    for folder in Path(train_dir).iterdir():
        if folder.is_dir() and not folder.name.startswith("."):
            train_classes.append(folder.name)
    train_classes.sort()

    val_classes = []
    for folder in Path(val_dir).iterdir():
        if folder.is_dir() and not folder.name.startswith("."):
            val_classes.append(folder.name)
    val_classes.sort()

    if not train_classes:
        raise ValueError(f"沒有分類資料夾：{train_dir}")
    if train_classes != val_classes:
        raise ValueError(f"類別不一致：訓練 {train_classes}，驗證 {val_classes}")


    # 3. 建立訓練用的讀圖工具，依設定決定是否增加照片變化。
    if augmentation is None:
        augmentation = {}

    if augmentation.get("use_aug", True):
        if augmentation.get("color_jitter", False):
            raise ValueError("color_jitter 尚未實作，請先設為 false，避免設定被忽略。")

        train_gen = ImageDataGenerator(
            rescale=1./255,                            # 像素數值縮小到 0～1
            rotation_range=augmentation.get("rotate", 20),
            width_shift_range=0.2,                     # 左右平移
            height_shift_range=0.2,                    # 上下平移
            zoom_range=0.2,                            # 放大縮小
            shear_range=0.2,                           # 形變
            horizontal_flip=augmentation.get("flip", False),
        )
    else:
        train_gen = ImageDataGenerator(rescale=1./255)


    # 4. 驗證照片只調整像素數值，不做隨機影像增強。
    val_gen = ImageDataGenerator(rescale=1./255)


    # 5. 讀取訓練照片，打亂順序。
    train_ds = train_gen.flow_from_directory(
        train_dir,
        classes=train_classes,
        target_size=tuple(img_size),
        batch_size=batch_size,
        class_mode="sparse",
        color_mode="rgb",
        interpolation="nearest",
        shuffle=True,
        seed=seed,
    )


    # 6. 讀取驗證照片，沿用訓練的類別編號，不打亂順序。
    val_ds = val_gen.flow_from_directory(
        val_dir,
        classes=train_classes,
        target_size=tuple(img_size),
        batch_size=batch_size,
        class_mode="sparse",
        color_mode="rgb",
        interpolation="nearest",
        shuffle=False,
        seed=seed,
    )

    # 7. 檢查是否有空分類，以及哪些檔案因副檔名不支援而被略過。
    for dataset, directory in [(train_ds, train_dir), (val_ds, val_dir)]:
        for name, class_id in dataset.class_indices.items():
            if class_id not in dataset.classes:
                raise ValueError(f"有類別沒有可讀取的圖片檔名：{name}，資料夾：{directory}")

        skipped = []
        for name in train_classes:
            for image in (Path(directory) / name).rglob("*"):
                if not image.is_file() or image.name.startswith("."):
                    continue
                extension = image.suffix.lower().lstrip(".")
                if extension not in dataset.white_list_formats:
                    skipped.append(str(image.relative_to(directory)))
        if skipped:
            warnings.warn(f"{directory} 有 {len(skipped)} 個不支援的檔案被略過，例如 {skipped[:5]}")

    return train_ds, val_ds


def load_test_dataset(test_dir, img_size, batch_size, class_names=None):
    
    # 1. 確認測試資料夾與分類名稱。
    """準備測試圖片，不做隨機增強，也不打亂順序。"""

    if not Path(test_dir).is_dir():
        raise ValueError(f"找不到資料夾：{test_dir}")

    test_classes = []
    for folder in Path(test_dir).iterdir():
        if folder.is_dir() and not folder.name.startswith("."):
            test_classes.append(folder.name)
    test_classes.sort()

    if not test_classes:
        raise ValueError(f"沒有分類資料夾：{test_dir}")

    # 有提供訓練時的分類順序，就照那個順序讀取。
    if class_names is not None:
        if set(test_classes) != set(class_names):
            raise ValueError(f"類別不一致：預期 {list(class_names)}，實際 {test_classes}")
        test_classes = list(class_names)


    # 2. 建立測試用的讀圖工具，不做隨機影像增強。
    test_gen = ImageDataGenerator(rescale=1./255)


    # 3. 讀取測試照片，不打亂順序。
    test_ds = test_gen.flow_from_directory(
        test_dir,
        classes=test_classes,
        target_size=tuple(img_size),
        batch_size=batch_size,
        class_mode="sparse",
        color_mode="rgb",
        interpolation="nearest",
        shuffle=False,
        seed=42,
    )


    # 4. 檢查空分類，並提示被略過的檔案。
    for name, class_id in test_ds.class_indices.items():
        if class_id not in test_ds.classes:
            raise ValueError(f"有類別沒有可讀取的圖片檔名：{name}，資料夾：{test_dir}")

    skipped = []
    for name in test_classes:
        for image in (Path(test_dir) / name).rglob("*"):
            if not image.is_file() or image.name.startswith("."):
                continue
            extension = image.suffix.lower().lstrip(".")
            if extension not in test_ds.white_list_formats:
                skipped.append(str(image.relative_to(test_dir)))
    if skipped:
        warnings.warn(f"{test_dir} 有 {len(skipped)} 個不支援的檔案被略過，例如 {skipped[:5]}")

    return test_ds
