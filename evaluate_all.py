"""Chấm đủ 4 chỉ số (Sα, Fωβ, MAE, mean Eξ) trên CAMO, COD10K, NC4K.
Cách chạy:
    python evaluate_all.py                 # chấm cả 3 tập
    python evaluate_all.py CAMO            # chỉ chấm 1 tập
    python evaluate_all.py CAMO 20         # chỉ chấm 20 ảnh đầu (để thử nhanh)
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from py_sod_metrics import Smeasure, WeightedFmeasure, MAE, Emeasure

from src.dataset import COD10KDataset
from src.model import DepthSAM

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

CHECKPOINT = os.environ.get("CHECKPOINT", "weights/depthsam_best_structure.pth")
DATA_ROOT = os.environ.get("DATA_ROOT", "dataset")   # giống train.py: ./dataset/TestDataset/...
DATASETS = sys.argv[1:2] or ["CAMO", "COD10K", "NC4K"]
MAX_IMAGES = int(sys.argv[2]) if len(sys.argv) > 2 else None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthSAM(encoder_type="vitl", checkpoint_path=None).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

for name in DATASETS:
    ds = COD10KDataset(os.path.join(DATA_ROOT, "TestDataset", name, "Imgs"),
                       os.path.join(DATA_ROOT, "TestDataset", name, "GT"))
    n = len(ds) if MAX_IMAGES is None else min(MAX_IMAGES, len(ds))
    sm, wfm, mae, em = Smeasure(), WeightedFmeasure(), MAE(), Emeasure()
    t0 = time.time()

    with torch.no_grad():
        for i in range(n):
            image, _ = ds[i]
            gt = np.array(Image.open(os.path.join(ds.mask_dir, os.path.splitext(ds.images[i])[0] + ".png")).convert("L"))
            pred = model(image.unsqueeze(0).to(device))
            pred = F.interpolate(pred, size=gt.shape, mode="bilinear", align_corners=False)
            pred = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
            for m in (sm, wfm, mae, em):
                m.step(pred=pred, gt=gt)
            if (i + 1) % 20 == 0 or i + 1 == n:
                spent = time.time() - t0
                print(f"{name}: {i + 1}/{n} ảnh | còn khoảng {spent / (i + 1) * (n - i - 1) / 60:.0f} phút")

    print("=" * 60)
    print(f"{name} ({n} ảnh): Sα = {sm.get_results()['sm']:.3f} | Fωβ = {wfm.get_results()['wfm']:.3f} | "
          f"MAE = {mae.get_results()['mae']:.3f} | mean Eξ = {em.get_results()['em']['curve'].mean():.3f} | "
          f"{(time.time() - t0) / n:.1f} giây/ảnh")
    print("=" * 60)
