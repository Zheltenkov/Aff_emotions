from dataclasses import dataclass

import torch
from torchvision import transforms
from typing import Dict, List, Optional
from data_loader.dataset import AffDataset
from torch.utils.data import DataLoader, WeightedRandomSampler


@dataclass(kw_only=True)
class LoadData:
    img_size: int
    batch_size: int
    class_weights: Optional[List[int | float]]
    train_data: Dict[str, List[str | int | float]]
    valid_data: Dict[str, List[str | int | float]]

    def get_transform(self, transform_type: str) -> transforms.Compose:
        # Initialize data transformers
        if transform_type == 'train':
            data_transform = transforms.Compose([transforms.Resize(self.img_size),
                                                 transforms.RandomVerticalFlip(),
                                                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                                                 transforms.RandomAffine(degrees=40, translate=None, shear=15),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
        elif transform_type == 'valid':
            data_transform = transforms.Compose([transforms.Resize(self.img_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
        else:
            raise ValueError(f'Choose type: train or valid')

        return data_transform

    def get_sampler(self, train_dataset) -> Optional[torch.Tensor]:
        if self.class_weights is not None:
            sampler = WeightedRandomSampler(weights=self.class_weights,
                                            num_samples=len(train_dataset), replacement=True)
        else:
            sampler = None
        return sampler

    def get_dataloader(self) -> Dict[str, DataLoader]:
        # Initialize train and validation datasets
        train_dataset = AffDataset(image_paths=self.train_data['images'], labels=self.train_data['labels'],
                                   transforms=self.get_transform(transform_type='train'))

        valid_dataset = AffDataset(image_paths=self.valid_data['images'], labels=self.valid_data['labels'],
                                   transforms=self.get_transform(transform_type='valid'))

        # Initialize sampler for DataLoader
        sampler = self.get_sampler(train_dataset)

        # Initialize train and validation dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=4, drop_last=True, sampler=sampler)

        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=4, drop_last=True)

        dataloaders = {'train': train_loader, 'valid': valid_loader}

        return dataloaders
