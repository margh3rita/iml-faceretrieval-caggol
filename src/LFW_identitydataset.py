import os
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image

class LFWIdentityDataset(Dataset):
    """Loads LFW faces grouped by identity for ArcFace training."""
    def __init__(self, root, preprocess, min_images=2):
        self.preprocess = preprocess
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
        print(f'LFWDataset: {self.num_classes} identities, {len(self.samples)} images')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.preprocess(img), label