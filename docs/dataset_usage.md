# AIGC Detection Benchmark 資料集使用說明

> 更新日期：2024-XX-XX

本文件說明 **AIGC Detection Benchmark** 相關資料集的檔案結構、常見情境使用方式，以及輔助腳本的操作範例，協助您快速上手並正確存取資料。

---

## 1. 資料夾結構

```
AIGCDetectionBenchMark/
├── AIGCDetectionBenchMark/        # 官方競賽資料
│   ├── train/...
│   └── test/...
├── progan_train/                  # ProGAN 產生之訓練影像
├── progan_val/                    # ProGAN 產生之驗證影像
└── progan_test/                   # ProGAN 產生之測試影像
```

各層資料夾下再依 **類別 (class name)** 分資料夾，例如 `airplane`、`bicycle`、`bird` ... 等。每個類別資料夾內皆為對應的影像檔 (`*.png`, `*.jpg`)。

> 註：若有其他來源（如 Stable Diffusion、StyleGAN2 等），亦會以相同方式被放置於對應資料夾中。

---

## 2. 檔案數量快速統計

專案內提供 `scripts/count_files.py`，可一次列出
1. *指定根目錄*（遞迴至所有子資料夾）的 **總檔案** 數
2. *同層每個子資料夾*（遞迴）的 **個別檔案** 數

### 2.1 使用方法

```bash
# (1) 使用預設路徑（可於腳本內修改 default_path）
python scripts/count_files.py

# (2) 指定自訂資料夾
python scripts/count_files.py /your/target/path
```

### 2.2 範例輸出

```
/raid/.../progan_train: 720119 個檔案
/raid/.../progan_train/airplane: 36006 個檔案
/raid/.../progan_train/bicycle: 36006 個檔案
...
```

---

## 3. 影像讀取範例 (PyTorch)

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

root_dir = "/raid/dannyliu/GAI_Detection/GenAI_Detection/AIGCDetectionBenchMark/progan_train"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

train_ds = datasets.ImageFolder(root=root_dir, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8)

for images, labels in train_loader:
    # 進行訓練...
    pass
```

---

## 4. 影像完整性檢查

若您擔心資料集中存在損毀檔案，可使用 `scripts/check_corrupt_data.py`：

```bash
python scripts/check_corrupt_data.py
```

腳本會列出無法順利開啟或驗證的圖片，並統計問題檔案數量。

---

## 5. 建議工作流程

1. **下載/同步** 資料集後，先執行 `check_corrupt_data.py`，確保所有影像可正常開啟。
2. 透過 `count_files.py` 快速檢查各類別影像數是否符合預期。
3. 依需求修改 `default_path` 或直接以參數指定路徑。
4. 在深度學習流程 (如 PyTorch) 中，以 `torchvision.datasets.ImageFolder` 搭配 `DataLoader` 載入。
5. 訓練/評估完成後，可將結果或模型保存在專案的 `outputs/` 或 `checkpoints/` 目錄 (需自行建立)。

---

## 6. 常見問題 (FAQ)

| 問題                                   | 解答 |
|----------------------------------------|------|
| 執行腳本時顯示 "找不到目錄"？           | 請確認路徑是否拼寫正確，並具有讀取權限。 |
| 需要只列出**檔案類型**數量 (如僅 *.png)？ | 目前腳本為統計所有類型檔案，可自行於 `os.walk` 中加入副檔名過濾條件。 |
| 如何同時列出多個資料夾統計？            | 可改寫簡易迴圈或一次傳多個路徑給腳本。 |

---

歡迎依實際需求調整與擴充上述腳本，如有疑問請隨時提出！ 