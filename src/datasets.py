import os
from collections import defaultdict
from torch.utils.data import Dataset, Subset
import PIL.Image as PILImage  # FIXED: Explicitly use this to avoid collisions

import torchvision.transforms as T

class TrainDataset(Dataset):
    """Loads Train dataset grouped by identity for ArcFace training."""
    def __init__(self, root, preprocess, min_images=2):
        self.preprocess = preprocess
        label_to_paths  = defaultdict(list)

        # scan folders structure
        for identity in os.listdir(root):
            folder = os.path.join(root, identity)
            if not os.path.isdir(folder):
                continue
            # create labels list
            for img_file in os.listdir(folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    label_to_paths[identity].append(
                        os.path.join(folder, img_file))

        # filter out the ids with less than min_images files
        kept            = {k: v for k, v in label_to_paths.items()
                           if len(v) >= min_images}
        # identity to integer in alphabetic order
        self.label_map  = {name: idx for idx, name in enumerate(sorted(kept))}
        self.samples    = [(path, self.label_map[identity])
                           for identity, paths in kept.items()
                           for path in paths]
        self.num_classes = len(kept)
        print(f'Train dataset: {self.num_classes} identities, {len(self.samples)} images')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # to avoid collisions with torchvision.transform
        img = PILImage.open(path).convert('RGB')
        return self.preprocess(img), label



class SubsetWithTransform(Subset):
    """Allows to apply different transforms to the subset, e.g. data augmentation"""
    def __init__(self, subset, transform):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform
    # overrides getitem to apply augm
    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        img = PILImage.open(path).convert('RGB')
        return self.transform(img), label
    
    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

class FlatImageDataset(Dataset):
    """Loads images from a flat directory."""
    def __init__(self, root_dir, preprocess=None):
        self.root_dir   = root_dir
        self.preprocess = preprocess
        self.filenames  = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.filenames[idx])
        img  = PILImage.open(path).convert('RGB')
        if self.preprocess:
            img = self.preprocess(img)
        # returns filename because there is no ground truth for inference
        return img, self.filenames[idx]



