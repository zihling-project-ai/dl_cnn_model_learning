# data_safety.py：安全複製圖片與對照分類名稱。

import csv
import os
import shutil
from pathlib import Path

LABELS = {"cat", "dog", "fox", "leopard", "lion", "wolf"}


def copy_labeled_files(pred_txt, source_dir, target_dir, apply=False):
    """依標記清單規劃並安全複製照片，預設只預覽。"""

    source = Path(source_dir).absolute()
    target = Path(target_dir).absolute()
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"來源必須是一般資料夾：{source}")
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError(f"目標必須是一般資料夾：{target}")
    source = source.resolve()
    target = target.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("來源與目標資料夾不可重疊")

    plan = []
    destinations = set()
    with open(pred_txt, newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.reader(f), 1):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f"標記檔第 {line_no} 行必須是 檔名,類別")
            fname, label = (value.strip() for value in row)
            if not fname or fname in {".", ".."} or "/" in fname or "\\" in fname:
                raise ValueError(f"不安全的檔名：{fname!r}")
            if label not in LABELS:
                raise ValueError(f"不支援的類別：{label!r}")
            src = source / fname
            dst = target / label / fname
            if src.is_symlink() or not src.is_file():
                raise ValueError(f"來源圖片不存在或是符號連結：{src}")
            if dst.parent.is_symlink() or (dst.parent.exists() and not dst.parent.is_dir()):
                raise ValueError(f"不安全的分類資料夾：{dst.parent}")
            if dst in destinations or os.path.lexists(dst):
                raise FileExistsError(f"目的檔案重複或已存在，停止且不覆寫：{dst}")
            destinations.add(dst)
            plan.append((src, dst))

    # 整批檢查完成後才允許寫入；任何已存在的目的檔都會讓整批停止。
    if not apply:
        for src, dst in plan:
            print(f"[預覽] 複製 {src} → {dst}")
        return plan
    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as reader:
            with dst.open("xb") as writer:  # 排他建立，避免檢查後出現撞名而覆寫
                try:
                    shutil.copyfileobj(reader, writer)
                except BaseException:
                    dst.unlink(missing_ok=True)
                    raise
        print(f"已複製：{dst}")
    return plan


# 明確列出允許的大類映射，避免 lionfish、wolf spider、foxhound 被誤標。
IMAGENET_ANIMAL_LABELS = {
    "red fox": "fox", "kit fox": "fox", "arctic fox": "fox", "grey fox": "fox",
    "grey wolf": "wolf", "alaskan tundra wolf": "wolf", "red wolf": "wolf",
    "lion": "lion", "leopard": "leopard", "snow leopard": "leopard", "jaguar": "leopard",
}


def map_imagenet_label(prediction):
    # 延續原專案把雪豹、美洲豹歸入「豹類」的分組方式。
    """將英文動物名稱對應成專案類別，不符合時回傳 None。"""

    return IMAGENET_ANIMAL_LABELS.get(prediction.strip().lower())
