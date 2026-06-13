import os
from collections import defaultdict
from torch.utils.data import Dataset, Subset
from PIL import Image

import torchvision.transforms as T

class compIdentityDataset(Dataset):
    """Loads competition images grouped by identity for ArcFace training."""
    def __init__(self, root, transform, min_images=2):
        self.transform = transform
        label_to_paths  = defaultdict(list)
        for identity in os.listdir(root):
            folder = os.path.join(root, identity)
            if not os.path.isdir(folder):
                continue
            for img_file in os.listdir(folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    label_to_paths[identity].append(
                        os.path.join(folder, img_file))
        # Keep only identities with enough images
        kept            = {k: v for k, v in label_to_paths.items()
                           if len(v) >= min_images}
        self.label_map  = {name: idx for idx, name in enumerate(sorted(kept))}
        self.samples    = [(path, self.label_map[identity])
                           for identity, paths in kept.items()
                           for path in paths]
        self.num_classes = len(kept)
        print(f'compIdentityDataset: {self.num_classes} identities, {len(self.samples)} images')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# augmentation pipeline
class SubsetWithTransform(Subset):
    def __init__(self, subset, transform):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label
    
    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

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



