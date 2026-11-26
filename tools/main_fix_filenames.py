# main_fix_filenames.py：先預覽再安全整理圖片檔名。
import argparse
import os
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data_afhq" / "test"


def force_rename_all(root, apply=False):
    """規劃 clean_流水號 檔名，預設只預覽，遇到撞名就停止。"""

    # 1. 檢查資料夾。
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"來源必須是一般資料夾：{root}")

    # 2. 先規劃整批新檔名，這一段不會改動圖片。
    rename_plan = []
    counter = 1

    for path, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()

        # 符號連結像是指向其他位置的捷徑，先擋下，避免改到別處。
        for directory_name in dirs:
            directory_path = Path(path) / directory_name
            if directory_path.is_symlink():
                raise ValueError("資料夾內含符號連結，請先人工確認")

        for old_name in sorted(files):
            full_old_path = Path(path) / old_name

            # 保留副檔名，例如 .jpg、.png，並統一轉成小寫。
            _, ext = os.path.splitext(old_name)
            ext = ext.lower()

            # 隱藏檔和非圖片檔案不處理。
            if old_name.startswith(".") or ext not in IMAGE_SUFFIXES:
                continue
            if full_old_path.is_symlink() or not full_old_path.is_file():
                raise ValueError(f"圖片必須是一般檔案：{full_old_path}")

            # 組合新檔名；05d 表示補成五位數，例如 00001。
            new_name = f"clean_{counter:05d}{ext}"
            full_new_path = Path(path) / new_name
            counter += 1

            # 名稱已經相同，就不需要改。
            if full_old_path == full_new_path:
                continue

            # 發現同名檔案，整批先停止，不覆寫任何圖片。
            if os.path.lexists(full_new_path):
                raise FileExistsError(f"目的檔已存在，停止且不覆寫：{full_new_path}")

            rename_plan.append((full_old_path, full_new_path))

    # 3. 預設只列出計畫，讓你先確認。
    if not apply:
        for full_old_path, full_new_path in rename_plan:
            print(f"[預覽] {full_old_path} → {full_new_path}")
        print(f"僅預覽，沒有變更：{len(rename_plan)} 個檔案")
        return rename_plan

    # 4. 確認 apply=True 後才執行。
    for full_old_path, full_new_path in rename_plan:
        # 先替同一份檔案建立新名字；若新名字已存在，os.link 會拒絕。 這裡保留硬連結做法，避免 os.rename 在部分系統上覆寫舊檔。
        os.link(full_old_path, full_new_path, follow_symlinks=False)
        try:
            full_old_path.unlink()  # 新名字建立成功後，才移除舊名字。
        except BaseException:
            full_new_path.unlink()  # 舊名字移除失敗時，撤回剛建立的新名字。
            raise
        print(f"[已完成] {full_old_path} → {full_new_path}")

    print(f"已完成：{len(rename_plan)} 個檔案")
    return rename_plan


if __name__ == "__main__":
    # 不加 --apply 就只預覽；--root 可以指定要整理的資料夾。
    parser = argparse.ArgumentParser(description="安全整理圖片檔名，預設只預覽")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true", help="確認預覽後，明確執行改名")
    args = parser.parse_args()
    force_rename_all(args.root, apply=args.apply)
