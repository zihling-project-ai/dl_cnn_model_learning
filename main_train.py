# main_train.py：訓練入口，依七個步驟準備與執行訓練。
import argparse
from pathlib import Path

import tensorflow as tf

from tools.dataloader import load_train_val_dataset
from models import get_model
from tools.utils import load_config, build_optimizer, validate_config, create_run_files


def main(config_path=None, extra_callbacks=None):
    # 1. 讀取設定並檢查，固定這次實驗的亂數種子。
    """依設定執行訓練；只有 fit 會開始更新模型參數。"""

    cfg = load_config(config_path)
    validate_config(cfg)

    model_name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]
    input_size = tuple(cfg["model"]["input_size"])
    aux_logits = cfg["model"].get("aux_logits", False)
    seed = cfg["training"].get("seed", 42)
    tf.keras.utils.set_random_seed(seed)

    # 2. 建立模型；這一步還沒有開始訓練。
    model = get_model(
        name=model_name,
        num_classes=num_classes,
        input_shape=input_size + (3,),
        aux_logits=aux_logits,
    )

    # 3. 準備訓練與驗證照片，依設定決定影像增強方式。
    train_ds, val_ds = load_train_val_dataset(
        cfg["dataset"]["train_dir"],
        cfg["dataset"]["val_dir"],
        batch_size=cfg["training"]["batch_size"],
        img_size=cfg["model"]["input_size"],
        augmentation=cfg.get("augmentation", {}),
        seed=seed,
    )
    if len(train_ds.class_indices) != num_classes:
        raise ValueError("num_classes 與訓練資料夾的類別數不同")

    # 4. 設定優化器、損失函數與評分方式。 優化器仍依 config.yaml 選擇，不改回固定使用 RMSprop。
    optimizer = build_optimizer(cfg["training"])
    use_aux = model_name.lower() == "googlenet" and aux_logits
    top_count = min(3, num_classes)

    if use_aux:
        print("=== GoogLeNet：使用兩個輔助分類輸出 ===")
        model.compile(
            optimizer=optimizer,
            loss={
                "aux1": "sparse_categorical_crossentropy",
                "aux2": "sparse_categorical_crossentropy",
                "main": "sparse_categorical_crossentropy",
            },
            loss_weights={"aux1": 0.3, "aux2": 0.3, "main": 1.0},
            metrics={
                "main": [
                    "accuracy",
                    tf.keras.metrics.SparseTopKCategoricalAccuracy(
                        k=top_count, name="top3_acc",
                    ),
                ]
            },
        )
        monitor_val = "val_main_accuracy"
    else:
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.SparseTopKCategoricalAccuracy(
                    k=top_count, name="top3_acc",
                ),
            ],
        )
        monitor_val = "val_accuracy"

    # 5. 準備存檔、早停與紀錄工具。 同名實驗已存在就停止；這裡也會保存設定與類別對照。
    weight_path, history_path = create_run_files(cfg, train_ds.class_indices)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(weight_path),
        monitor=monitor_val,
        mode="max",
        save_best_only=True,
        save_weights_only=True,
    )

    # 保留目前 min_delta=0 的設定，和最佳權重使用相同改善門檻。
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_val,
        mode="max",
        patience=5,
        min_delta=0.0,
        restore_best_weights=True,
    )

    # create_run_files 已先建立新的空白 CSV，因此這裡接續寫入。
    log_cb = tf.keras.callbacks.CSVLogger(str(history_path), append=True)
    nan_cb = tf.keras.callbacks.TerminateOnNaN()
    # Lab 需要完整比較指定輪數，明確關閉早停；舊命令列設定仍預設開啟。
    callbacks = [checkpoint_cb]
    if cfg["training"].get("early_stopping", True):
        callbacks.append(earlystop_cb)
    callbacks.extend([log_cb, nan_cb])
    # 介面可另外接收進度與取消訊號，命令列使用方式不變。
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # 6. 開始訓練；只有執行到 fit 才會更新模型參數。
    print("\n=== 開始訓練 ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["training"]["epochs"],
        callbacks=callbacks,
        workers=1,
    )

    # 7. 載回最佳權重，印出模型結構與每輪訓練結果。
    if not weight_path.is_file():
        raise RuntimeError("沒有產生最佳權重，請檢查訓練紀錄中的 loss 與驗證指標。")
    model.load_weights(str(weight_path))

    print("\n=== 訓練結束，已載回最佳權重 ===")
    model.summary()
    print(f"最佳權重：{weight_path}")
    print(f"逐輪紀錄：{history_path}")

    # 使用 Python 直接印表格，不額外依賴 pandas 或 tabulate。
    metric_names = list(history.history.keys())
    column_widths = {}
    for metric_name in metric_names:
        column_widths[metric_name] = max(12, len(metric_name))

    header = f"{'Epoch':>5}"
    for metric_name in metric_names:
        width = column_widths[metric_name]
        header += f" | {metric_name:>{width}}"
    print("\n" + header)
    print("-" * len(header))

    if metric_names:
        num_epochs = len(history.history[metric_names[0]])
        for epoch in range(num_epochs):
            row = f"{epoch + 1:>5}"
            for metric_name in metric_names:
                width = column_widths[metric_name]
                value = history.history[metric_name][epoch]
                row += f" | {value:>{width}.4f}"
            print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="依設定訓練影像分類模型")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    # 批次檢查可明確標記略過，避免重跑已完成的實驗。
    skip_file = args.config.parent / "skip_training.json" if args.config else None
    if skip_file and skip_file.is_file():
        print(f"此實驗已標記略過，沒有開始訓練。原因紀錄：{skip_file}")
    else:
        main(args.config)
