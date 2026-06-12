from PIL import Image
from torch.utils.data import Subset
import torchvision.transforms as T


# 'potere' dell'augmentation, dipende dalla dimensione del dataset,
# eventualmente si possono cambiare i parametri
# train_transform = T.Compose([
#     T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.9, 1.1),
#                         interpolation=T.InterpolationMode.BICUBIC),
#     T.RandomHorizontalFlip(p=0.5),
#     T.RandomRotation(degrees=15),
#     T.RandomApply([T.ColorJitter(
#         brightness=0.3, contrast=0.3, saturation=0.2, hue=0.08
#     )], p=0.7),
#     T.RandomGrayscale(p=0.15),
#     T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
#     T.ToTensor(),
#     T.Normalize(mean=(0.48145466, 0.4578275,  0.40821073),
#                 std =(0.26862954, 0.26130258, 0.27577711)),
#     T.RandomErasing(p=0.2, scale=(0.01, 0.08), ratio=(0.3, 3.3), value=0),
# ])

# val_transform = T.Compose([
#     T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
#     T.CenterCrop(224),
#     T.ToTensor(),
#     T.Normalize(mean=(0.48145466, 0.4578275,  0.40821073),
#                 std =(0.26862954, 0.26130258, 0.27577711)),
# ])

# ── Wrapper val subset: applica val_transform senza copiare il dataset ────────
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
