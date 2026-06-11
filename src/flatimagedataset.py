# Dataset should be defined in the main script
import os

from torch.utils.data import Dataset
from PIL import Image

class FlatImageDataset(Dataset):
    """Loads images from a flat directory for CLIP embedding extraction."""
    def __init__(self, root_dir, preprocess):
        self.root_dir   = root_dir
        self.preprocess = preprocess
        self.image_paths = []
        self.image_names = []
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                self.image_paths.append(os.path.join(root_dir, filename))
                self.image_names.append(filename)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image    = Image.open(img_path).convert('RGB')
        return self.preprocess(image), self.image_names[idx]