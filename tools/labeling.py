# labeling.py：使用現成模型批次標記動物照片。
import os
import json
from pathlib import Path
from .data_safety import copy_labeled_files, map_imagenet_label
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub


# 1. 準備現成模型，需要時才載入。
model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2"
model = None


def get_label_model():
    """第一次需要時載入現成模型，後續重複使用。"""

    global model
    if model is None:
        model = hub.load(model_url)
        print("模型載入完成")
    return model

# 2. 讀取本機的 ImageNet 英文類別清單。
current_dir = os.path.dirname(__file__)
labels_path = os.path.join(current_dir, "imagenet_labels.json")

# 讀取 ImageNet 1000 類標籤（用於模型輸出）
with open(labels_path, "r", encoding="utf-8") as f:
    imagenet_labels = json.load(f)

# 保留 ImageNet 英文類別清單。
labels = imagenet_labels

# 3. 準備單張圖片。
def preprocess_image(path): 
    """將單張照片轉成 RGB、384×384，並把數值除以 255。"""

    img = cv2.imread(path)  # BGR
    if img is None:
        raise ValueError(f"圖片讀取失敗: {path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉 RGB
    img = cv2.resize(img, (384, 384))  # 調整為標記模型的輸入尺寸
    img = img.astype("float32") / 255.0
    return img  # 不加 batch 維度，一張圖 shape = (384,384,3)


# 4. 對照自訂動物類別。
def map_label(pred):
    """使用共用名稱對照表取得自訂動物類別。"""

    return map_imagenet_label(pred)


# 5. 分批辨識並儲存標記清單。
def process_folder(src_dir, output_txt, batch_size=32):
    """分批辨識資料夾中的照片，建立不覆寫的標記清單。"""

    if os.path.lexists(output_txt):
        raise FileExistsError(f"標記檔已存在，不覆寫：{output_txt}")
    model = get_label_model()
    fnames = os.listdir(src_dir)
    results = {}

    batch_images = []
    batch_file_names = []

    for fname in tqdm(fnames):
        path = os.path.join(src_dir, fname)

        try:
            img = preprocess_image(path)  # shape (384,384,3)
            batch_images.append(img)
            batch_file_names.append(fname)

            # batch 滿了就一次推論
            if len(batch_images) == batch_size:
                batch_tensor = tf.convert_to_tensor(np.stack(batch_images, axis=0))  # 修正
                preds = model(batch_tensor).numpy()

                for i, p in enumerate(preds):
                    idx = np.argmax(p)
                    label = imagenet_labels[idx]
                    mapped = map_label(label)
                    results[batch_file_names[i]] = mapped

                batch_images = []
                batch_file_names = []

        except Exception as e:
            print("Error:", e)

    # 最後剩下的圖片
    if len(batch_images) > 0:
        batch_tensor = np.stack(batch_images, axis=0)
        preds = model(batch_tensor).numpy()

        for i, p in enumerate(preds):
            idx = np.argmax(p)
            label = imagenet_labels[idx]
            mapped = map_label(label)
            results[batch_file_names[i]] = mapped

    # 寫入檔案
    with open(output_txt, "x", encoding="utf-8") as f:
        for fname, label in results.items():
            if label is not None:
                f.write(f"{fname},{label}\n")

    print(f"=== {output_txt} 標記完成 ===")

# 6. 依清單複製照片，原圖保留。
def move_files(pred_txt, source_dir, target_dir, apply=False):
    """依標記清單安全複製照片，保留原圖，預設只預覽。"""
    return copy_labeled_files(pred_txt, source_dir, target_dir, apply=apply)


# 7. 串接完整標記流程。
def auto_label(apply=False):
    """串接訓練與驗證照片的標記流程，明確允許後才執行。"""

    project_root = Path(__file__).resolve().parents[1]
    train_wild = str(project_root / "data_afhq/train/wild")
    val_wild = str(project_root / "data_afhq/val/wild")

    out_train = str(project_root / "wild_predictions_train.txt")
    out_val = str(project_root / "wild_predictions_val.txt")

    if not apply:
        print("僅預覽：確認後使用 auto_label(apply=True) 才產生標記檔與複製圖片。")
        return
    for folder in (train_wild, val_wild):
        if Path(folder).is_symlink() or not Path(folder).is_dir():
            raise ValueError(f"找不到一般來源資料夾：{folder}")
    for output in (out_train, out_val):
        if os.path.lexists(output):
            raise FileExistsError(f"標記檔已存在，不覆寫：{output}")

    # 1. 產生標記清單。
    process_folder(train_wild, out_train)
    process_folder(val_wild, out_val)

    # 2. 依清單安全複製照片。
    move_files(out_train, train_wild, project_root / "data_afhq/train", apply=True)
    move_files(out_val, val_wild, project_root / "data_afhq/val", apply=True)
