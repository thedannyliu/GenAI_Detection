import os
import sys
from typing import Dict


def count_files_recursive(directory: str) -> int:
    """回傳指定目錄及其所有子目錄中的檔案總數。"""
    total = 0
    try:
        for _, _, files in os.walk(directory):
            total += len(files)
    except FileNotFoundError:
        print(f"[警告] 找不到目錄: {directory}")
    return total


def gather_counts(root_dir: str) -> Dict[str, int]:
    """取得 root 目錄本身以及其第一層子目錄的檔案數量。"""
    # root 目錄的「遞迴」檔案總數
    counts = {root_dir: count_files_recursive(root_dir)}

    # 計算第一層子目錄（各自遞迴）的檔案總數
    try:
        for entry in os.scandir(root_dir):
            if entry.is_dir():
                counts[entry.path] = count_files_recursive(entry.path)
    except FileNotFoundError:
        print(f"[錯誤] 目錄不存在: {root_dir}")
    return counts


def main():
    # 預設目錄，可透過命令列參數覆蓋
    default_path = "/raid/dannyliu/GAI_Detection/GenAI_Detection/AIGCDetectionBenchMark/AIGCDetectionBenchMark/test"
    root_dir = sys.argv[1] if len(sys.argv) > 1 else default_path

    counts = gather_counts(root_dir)
    # 依路徑排序輸出
    for path in sorted(counts.keys()):
        print(f"{path}: {counts[path]} 個檔案")


if __name__ == "__main__":
    main() 