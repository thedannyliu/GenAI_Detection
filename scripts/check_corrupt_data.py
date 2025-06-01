from PIL import Image
import glob
import os

image_files = glob.glob("/raid/dannyliu/dataset/GAI_Dataset/genimage/imagenet_ai_0419_sdv4/train/ai/*.png") # 調整路徑和模式
image_files += glob.glob("/raid/dannyliu/dataset/GAI_Dataset/genimage/imagenet_ai_0419_sdv4/train/nature/*.png")
# ... 為其他類別和 split 添加更多

corrupted_images = []
for f_path in image_files:
    try:
        img = Image.open(f_path)
        img.verify() # 驗證圖片數據是否完整
        # img.load() # 更徹底的檢查，會解碼整個圖片，但更耗時
    except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
        print(f"Corrupted image: {f_path} - {e}")
        corrupted_images.append(f_path)
        # os.remove(f_path) # 如果確定要刪除
    
print(f"Found {len(corrupted_images)} corrupted images.")