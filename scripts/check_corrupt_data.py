"""check_corrupt_data.py

使用方式：

```bash
python scripts/check_corrupt_data.py /path/to/dataset
# 若不指定，預設為目前工作目錄
```

腳本會遞迴掃描指定資料夾下的所有影像檔（副檔名符合 EXTENSIONS ），
無法開啟/驗證的檔案將直接刪除，並列出總數統計。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from PIL import Image, UnidentifiedImageError


# 可支援的影像副檔名（統一小寫比較）
EXTENSIONS: List[str] = [
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tiff",
    ".webp",
]


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in EXTENSIONS


def find_images(root: Path) -> List[Path]:
    """遞迴尋找所有影像檔案。"""
    return [p for p in root.rglob("*") if p.is_file() and is_image_file(p)]


def check_and_remove(images: List[Path]) -> int:
    """檢查影像檔，若損毀則刪除。回傳刪除數量。"""
    removed = 0
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                img.verify()  # 只驗證檔頭
        except (IOError, SyntaxError, UnidentifiedImageError) as e:
            print(f"[警告] 損毀影像已刪除: {img_path} - {e}")
            try:
                img_path.unlink()
            except Exception as rm_err:
                print(f"    -> 刪除失敗: {rm_err}")
            removed += 1
    return removed


def main() -> None:
    default_dir = Path.cwd()
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir

    if not root.exists():
        print(f"[錯誤] 目錄不存在: {root}")
        sys.exit(1)

    all_images = find_images(root)
    print(f"在 {root} 共有 {len(all_images)} 個影像檔，開始檢查...")

    removed_count = check_and_remove(all_images)

    print("-" * 40)
    print(f"完成！共檢查 {len(all_images)} 個影像，刪除 {removed_count} 個損毀檔案。")


if __name__ == "__main__":
    main()