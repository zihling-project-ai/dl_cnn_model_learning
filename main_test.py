# main_test.py：測試入口，核對設定並計算分類成績。
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from tools.dataloader import load_test_dataset
from models import get_model
from tools.utils import load_config, validate_config, metadata_path, validate_run_metadata


def evaluate_batches(model, dataset, include_predictions=False, on_batch_end=None):
    """分批辨識並計分，不訓練或更新模型權重。"""

    # 1. 測試照片不能打亂順序，而且必須有資料。
    if dataset.shuffle:
        raise ValueError("測試資料必須使用 shuffle=False")
    if len(dataset) == 0:
        raise ValueError("測試資料集沒有圖片")

    num_classes = len(dataset.class_indices)
    all_probabilities = []
    all_labels = []

    # 2. 每次拿一批照片，連同這一批的正確答案一起處理。
    for batch_index in range(len(dataset)):
        images, correct_labels = dataset[batch_index]
        predictions = model(images, training=False)

        # GoogLeNet 可能有三份輸出，測試時只看主要的 main。
        if isinstance(predictions, dict):
            predictions = predictions["main"]

        batch_probabilities = np.asarray(predictions)

        # 每張照片應有 num_classes 個分數，且不能出現 NaN 或 Inf。
        expected_shape = (len(correct_labels), num_classes)
        if batch_probabilities.shape != expected_shape:
            raise ValueError("模型輸出的圖片數或類別數不正確")
        if not np.isfinite(batch_probabilities).all():
            raise ValueError("模型輸出包含 NaN 或 Inf")

        # Softmax 分數應為非負數，每張照片的分數加總約等於 1。
        if np.any(batch_probabilities < 0):
            raise ValueError("模型輸出包含負的機率")
        probability_sums = batch_probabilities.sum(axis=1)
        if not np.allclose(probability_sums, 1, atol=1e-4):
            raise ValueError("每張照片的分類機率加總不等於 1")

        all_probabilities.append(batch_probabilities)
        all_labels.append(np.asarray(correct_labels, dtype=int))
        if on_batch_end is not None:
            on_batch_end(batch_index + 1, len(dataset))

    # 3. 把各批結果接起來，選分數最高的類別作為答案。
    all_probabilities = np.concatenate(all_probabilities)
    y_true = np.concatenate(all_labels)
    y_pred = np.argmax(all_probabilities, axis=1)
    num_images = len(y_true)

    # 4. 計算平均分類損失與整體準確率。 loss 只計主要分類輸出，不加上 GoogLeNet 的輔助損失或額外正則化損失。
    image_losses = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, all_probabilities,
    )
    average_loss = float(np.mean(image_losses.numpy()))
    accuracy = float(np.mean(y_pred == y_true))

    # 5. 計算 Top3：正確答案有沒有出現在分數最高的三個類別中。 如果總共只有兩類，就最多取兩個答案。
    top_count = min(3, num_classes)
    sorted_predictions = np.argsort(all_probabilities, axis=1)
    top_predictions = sorted_predictions[:, -top_count:]

    correct_top3 = 0
    for correct_label, candidate_labels in zip(y_true, top_predictions):
        if correct_label in candidate_labels:
            correct_top3 += 1
    top3_accuracy = correct_top3 / num_images

    # 6. 依類別編號排列名稱，避免報告上的名稱與答案對不上。
    class_names = [None] * num_classes
    for class_name, class_id in dataset.class_indices.items():
        class_names[class_id] = class_name
    class_ids = list(range(num_classes))

    # 7. 產生每一類的 Precision、Recall、F1，以及混淆矩陣。
    report = classification_report(
        y_true, y_pred,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true, y_pred,
        labels=class_ids,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=class_ids)

    result = {
        "samples": num_images,
        "class_indices": dataset.class_indices,
        "loss": average_loss,
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "report": report,
        "report_text": report_text,
        "confusion_matrix": matrix.tolist(),
    }
    # 介面需要逐張預測來顯示誤判照片；原命令列預設不回傳此清單。
    if include_predictions:
        if len(dataset.filenames) != num_images:
            raise ValueError("圖片檔名與預測數量不一致")
        result["predictions"] = []
        for index, filename in enumerate(dataset.filenames):
            result["predictions"].append({
                "filename": filename,
                "true_id": int(y_true[index]),
                "pred_id": int(y_pred[index]),
                "confidence": float(all_probabilities[index, y_pred[index]]),
            })
    return result


def main(config_path=None, weights=None, allow_legacy=False):
    
    # 1. 讀取並檢查設定。
    """核對設定並測試權重，印出易讀的分類報告。"""

    cfg = load_config(config_path)
    validate_config(cfg)

    model_name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]
    input_size = tuple(cfg["model"]["input_size"])
    aux_logits = cfg["model"].get("aux_logits", False)

    # 2. 找到權重，核對旁邊的設定與類別紀錄。
    if weights is not None:
        weight_path = Path(weights)
    else:
        checkpoint_dir = Path(cfg["save"]["checkpoint_dir"])
        weight_path = checkpoint_dir / f"{model_name}_best.h5"

    if not weight_path.is_file():
        raise FileNotFoundError(f"找不到權重：{weight_path}")

    metadata_file = metadata_path(weight_path)
    class_names = None
    settings_verified = False

    if metadata_file.is_file():
        with metadata_file.open(encoding="utf-8") as f:
            saved_settings = json.load(f)
        class_names = validate_run_metadata(cfg, saved_settings)
        settings_verified = True
    else:
        if not allow_legacy:
            raise ValueError(
                "舊權重缺少當次設定與類別紀錄；確認設定後可加 --allow-legacy 探索，"
                "不代表重現歷史成績。"
            )
        warnings.warn("探索舊權重：輸入尺寸、前處理與類別順序尚未核實，結果不代表歷史成績。")

    # 3. 建立模型，再載入以前學到的權重。
    model = get_model(
        name=model_name,
        num_classes=num_classes,
        input_shape=input_size + (3,),
        aux_logits=aux_logits,
    )
    dummy = tf.zeros((1,) + input_size + (3,))
    model(dummy, training=False)
    model.load_weights(str(weight_path))  # 不跳過不相容的層。

    # 4. 讀取測試照片，不打亂順序。
    test_ds = load_test_dataset(
        cfg["dataset"]["test_dir"],
        img_size=input_size,
        batch_size=cfg["training"]["batch_size"],
        class_names=class_names,
    )
    if len(test_ds.class_indices) != num_classes:
        raise ValueError("測試資料類別數與模型不同")

    # 5. 分批辨識並計分，不呼叫 fit。
    result = evaluate_batches(model, test_ds)
    result["settings_verified"] = settings_verified

    # 6. 印出容易閱讀的成績表；仍回傳 result，方便其他程式使用。
    print("\n=== 測試結果 ===")
    print(f"模型：{model_name}")
    print(f"權重：{weight_path}")
    if settings_verified:
        print("設定紀錄：核對通過")
    else:
        print("設定紀錄：舊權重探索模式，尚未核實歷史設定")
    print(f"測試圖片：{result['samples']} 張")
    print(f"Loss（主要分類輸出）：{result['loss']:.4f}")
    print(f"Accuracy：{result['accuracy']:.2%}")
    print(f"Top3 Accuracy：{result['top3_accuracy']:.2%}")

    print("\n=== 各類別分類報告 ===")
    print(result["report_text"])

    print("=== 混淆矩陣：列是實際類別，欄是預測類別 ===")
    names_in_order = [None] * num_classes
    for class_name, class_id in result["class_indices"].items():
        names_in_order[class_id] = class_name
    print("類別順序：" + ", ".join(names_in_order))
    print(np.asarray(result["confusion_matrix"]))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="驗證分類權重與 Precision / Recall / F1")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--allow-legacy", action="store_true", help="明確允許探索缺少設定紀錄的舊權重")
    args = parser.parse_args()
    main(args.config, args.weights, args.allow_legacy)
