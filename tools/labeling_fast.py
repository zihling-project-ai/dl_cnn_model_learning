# labeling_fast.py：使用多執行緒讀圖與 XLA 批次標記照片。
import os
import json
from pathlib import Path
from .data_safety import copy_labeled_files, map_imagenet_label
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from concurrent.futures import ThreadPoolExecutor


# 1. 準備模型：第一次需要時才載入，並設定 XLA 編譯

model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2"
model = None  # 尚未載入；定義函式時不會下載模型。


def get_label_model():
    """第一次需要時載入現成模型，後續重複使用。"""

    global model  # 使用上面那個 model，讓後續批次可以重複使用。
    if model is None:  # 第一次呼叫才執行這一段。
        tf.config.optimizer.set_jit(True)
        model = hub.load(model_url)  # 載入模型；本機沒有快取時可能需要下載。
        print("模型載入完成，XLA 加速已啟用")
    return model  # 之後呼叫時，直接取用已載入的模型。

# 要求這段辨識運算使用 XLA 編譯；實際速度需測量，呼叫前先載入模型。
@tf.function(jit_compile=True)
def fast_predict(batch_tensor):
    """使用 XLA 編譯辨識一批照片，呼叫前必須先載入模型。"""

    return model(batch_tensor)


# 2. 讀取 ImageNet 的英文類別名稱

current_dir = os.path.dirname(__file__)
labels_path = os.path.join(current_dir, "imagenet_labels.json")

with open(labels_path, "r", encoding="utf-8") as f:
    imagenet_labels = json.load(f)

# ImageNet 類別名稱列表
labels = imagenet_labels


# 3. 準備單張照片：讀圖、轉成 RGB、縮放成 384×384、數值除以 255

def preprocess_image(path):
    """將單張照片轉成 RGB、384×384，並把數值除以 255。"""

    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"讀不到圖片: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))
    return img.astype("float32") / 255.0


# 4. 將英文名稱對應成狐狸、狼、獅子或豹類（對照表在 data_safety.py）

def map_label(pred):
    """使用共用名稱對照表取得自訂動物類別。"""

    return map_imagenet_label(pred)


# 5. 使用多個執行緒同時讀取照片

def load_image_threaded(path_list):
    """使用最多八個執行緒，同時準備一批圖片。"""

    with ThreadPoolExecutor(max_workers=8) as executor:
        imgs = list(executor.map(preprocess_image, path_list))
    return imgs


# 6. 分批辨識資料夾中的照片，產生標記清單

def process_folder(src_dir, output_txt, batch_size=32):
    """分批辨識資料夾中的照片，建立不覆寫的標記清單。"""

    if os.path.lexists(output_txt):
        raise FileExistsError(f"標記檔已存在，不覆寫：{output_txt}")

    get_label_model()  # 到這裡才載入模型，並設定 XLA。
    fnames = os.listdir(src_dir)
    results = {}

    # 先用一張全零的測試圖片暖機，不會更新模型權重。
    dummy = tf.zeros((1, 384, 384, 3), dtype=tf.float32)
    _ = fast_predict(dummy)

    for i in tqdm(range(0, len(fnames), batch_size)):
        batch_files = fnames[i : i + batch_size]
        batch_paths = [os.path.join(src_dir, f) for f in batch_files]

        # 同時讀取這一批照片
        imgs = load_image_threaded(batch_paths)     # list of np.array

        # 把照片陣列合成一批，再轉成 Tensor
        batch_tensor = tf.convert_to_tensor(np.stack(imgs, axis=0))

        # 辨識這一批照片
        preds = fast_predict(batch_tensor).numpy()

        for fname, p in zip(batch_files, preds):
            idx = np.argmax(p)
            label = imagenet_labels[idx]
            mapped = map_label(label)
            if mapped:
                results[fname] = mapped


    # 儲存「檔名,類別」清單，已有同名標記檔就拒絕覆寫
    
    with open(output_txt, "x", encoding="utf-8") as f:
        for fname, label in results.items():
            f.write(f"{fname},{label}\n")

    print(f"=== {output_txt} 標記完成 ===")


# 7. 依標記清單複製照片，預設只預覽；原始照片保留

def move_files(pred_txt, source_dir, target_dir, apply=False):
    """依標記清單安全複製照片，保留原圖，預設只預覽。"""
    return copy_labeled_files(pred_txt, source_dir, target_dir, apply=apply)


# 8. 串接整個標記流程，預設只顯示提示，apply=True 才執行

def auto_label(apply=False):
    """串接訓練與驗證照片的標記流程，明確允許後才執行。"""

    project_root = Path(__file__).resolve().parents[1]

    train_wild = str(project_root / "data_afhq/train/wild")
    val_wild   = str(project_root / "data_afhq/val/wild")

    out_train = str(project_root / "wild_predictions_train.txt")
    out_val   = str(project_root / "wild_predictions_val.txt")

    if not apply:
        print("僅預覽：確認後使用 auto_label(apply=True) 才產生標記檔與複製圖片。")
        return
    for folder in (train_wild, val_wild):
        if Path(folder).is_symlink() or not Path(folder).is_dir():
            raise ValueError(f"找不到一般來源資料夾：{folder}")
    for output in (out_train, out_val):
        if os.path.lexists(output):
            raise FileExistsError(f"標記檔已存在，不覆寫：{output}")

    print("\n=== Step 1: Auto Labeling (TRAIN) ===")
    process_folder(train_wild, out_train)

    print("\n=== Step 2: Auto Labeling (VAL) ===")
    process_folder(val_wild, out_val)

    print("\n=== Step 3: Moving Files ===")
    move_files(out_train, train_wild, project_root / "data_afhq/train", apply=True)
    move_files(out_val,   val_wild,   project_root / "data_afhq/val", apply=True)
