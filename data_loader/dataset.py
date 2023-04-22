from PIL import Image
from dataclasses import dataclass
from typing import List, Optional
from torch.utils.data import Dataset
from torchvision.transforms import Compose


@dataclass(kw_only=True)
class AffDataset(Dataset):
    """ Dataset class for the AffWild data """
    image_paths: List[str] = None
    labels: List[str | int | float] = None
    transforms: Compose = None
    transforms_type: Optional[str] = None

    def __len__(self):
        """ Return the length of the dataset """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """ Get an item from the dataset """
        self.validate_transforms_type()

        label = self.labels[idx]
        image_path = self.image_paths[idx]

        # Open image and convert it to RGB
        with Image.open(image_path).convert('RGB') as image:
            if self.transforms is not None:
                image = self.transforms(image)

        return {'image': image, 'label': label}

    def validate_transforms_type(self) -> None:
        """Validate the value of transforms_type"""
        if self.transforms_type is not None and self.transforms_type not in ('train', 'valid'):
            raise ValueError(f"Invalid transforms_type '{self.transforms_type}', must be 'train' or 'valid'.")
