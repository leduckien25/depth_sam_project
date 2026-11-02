import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class COD10KDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=384):
        """
        Args:
            image_dir (str): Đường dẫn tới thư mục chứa ảnh RGB gốc
            mask_dir (str): Đường dẫn tới thư mục chứa ảnh mặt nạ (Ground Truth)
            image_size (int): Kích thước chuẩn để resize ảnh (thường dùng 384 hoặc 512 cho DAv2)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        
        self.images = sorted(os.listdir(image_dir))
        
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
        
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        
        return image, mask