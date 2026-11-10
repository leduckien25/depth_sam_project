import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class COD10KDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=378, augment=False):
        """
        Args:
            image_dir (str): Đường dẫn tới thư mục chứa ảnh RGB gốc
            mask_dir (str): Đường dẫn tới thư mục chứa ảnh mặt nạ (Ground Truth)
            image_size (int): Kích thước resize ảnh, phải chia hết cho 14 (378 = 27 x 14)
            augment (bool): True khi train: lật ngang + cắt ngẫu nhiên (bài báo mục 4.1)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        # Chỉ lấy ảnh có mask tương ứng
        self.images = [f for f in sorted(os.listdir(image_dir))
                       if os.path.exists(os.path.join(mask_dir, os.path.splitext(f)[0] + ".png"))]
        
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        base_name, _ = os.path.splitext(img_name)
        mask_name = base_name + ".png"
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            # Lật ngang ngẫu nhiên (50%)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            # Cắt ngẫu nhiên một vùng 80-100% kích thước ảnh
            w, h = image.size
            s = random.uniform(0.8, 1.0)
            cw, ch = int(w * s), int(h * s)
            x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
            image = image.crop((x0, y0, x0 + cw, y0 + ch))
            mask = mask.crop((x0, y0, x0 + cw, y0 + ch))

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        
        return image, mask