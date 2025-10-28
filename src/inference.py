# src/inference.py
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm


# ======================================================
# Dataset（推論用）
# ======================================================
class OfficeHomeInferenceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        self.filenames = df.iloc[:, 1].tolist()  # 第二欄 filename
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.filenames[idx]


# ======================================================
# 建構模型（與 Setting C 相同）
# ======================================================
def build_model(num_classes=65, device="cpu"):
    model = models.resnet50(weights=None)
    in_dim = model.fc.in_features
    model.fc = nn.Linear(in_dim, num_classes)
    model.to(device)
    return model


# ======================================================
# 主程式：輸入參數 → 推論 → 輸出 CSV
# ======================================================
def main():
    if len(sys.argv) != 4:
        print("Usage: python src/inference.py <csv_path> <img_dir> <out_csv>")
        sys.exit(1)

    csv_path, img_dir, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    ckpt_path = "ckpt/settingC_best.pth"  # 由 hw1_download_ckpt.sh 下載後會放在這裡

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Transform（與 val 相同） ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # === Dataset & Dataloader ===
    dataset = OfficeHomeInferenceDataset(csv_path, img_dir, transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False,
                        num_workers=2, pin_memory=(device.type == "cuda"))

    # === 載入模型 ===
    model = build_model(num_classes=65, device=device)
    print(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # === 推論 ===
    results = []
    running_id = 0
    with torch.no_grad():
        for images, filenames in tqdm(loader, desc="Inferencing"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            for fn, pred in zip(filenames, preds):
                results.append((running_id, fn, int(pred)))
                running_id += 1

    # === 輸出 CSV ===
    df_out = pd.DataFrame(results, columns=["id", "filename", "label"])
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Saved predictions to {out_csv}")
    print(df_out.head())


if __name__ == "__main__":
    main()
