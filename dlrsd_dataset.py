import os
from PIL import Image
from torch.utils.data import Dataset

class DLRSDDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.image_paths = []
        
        split_dir = os.path.join(root, split)
        for img_name in os.listdir(split_dir):
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                self.image_paths.append(os.path.join(split_dir, img_name))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # 返回图像和伪标签
