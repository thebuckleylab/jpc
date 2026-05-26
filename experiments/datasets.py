import os
from pathlib import Path

import jax.random as jr

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader


DATA_DIR = "datasets"
TINYIMAGENET_DIR = f"{DATA_DIR}/tiny-imagenet-200"
IMAGENET_DIR = f"~/projects/jpc/experiments/{DATA_DIR}/ImageNet"


def make_gaussian_dataset(key, mean, std, shape):
    x = mean + std * jr.normal(key, shape)
    y = x
    return (x, y)


def get_dataloaders(dataset_id, batch_size, flatten=True, generator=None):
    train_data = get_dataset(
        id=dataset_id,
        train=True,
        normalise=True,
        flatten=flatten
    )
    test_data = get_dataset(
        id=dataset_id,
        train=False,
        normalise=True,
        flatten=flatten
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator
    )
    return train_loader, test_loader


def get_dataset(id, train, normalise, flatten=True):
    if id == "MNIST":
        dataset = MNIST(train=train, normalise=normalise, flatten=flatten)
    elif id == "Fashion-MNIST":
        dataset = FashionMNIST(train=train, normalise=normalise, flatten=flatten)
    elif id == "CIFAR10":
        dataset = CIFAR10(train=train, normalise=normalise, flatten=flatten)
    else:
        raise ValueError(
            "Invalid dataset ID. Options are `MNIST`, `Fashion-MNIST` and `CIFAR10`"
        )
    return dataset


def get_imagenet_loaders(batch_size):
    train_data, val_data = ImageNet(split="train"), ImageNet(split="val")
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=32,
        persistent_workers=True
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=32,
        persistent_workers=True
    )
    return train_loader, val_loader


def get_tinyimagenet_loaders(batch_size, generator=None):
    train_data = TinyImageNet(split="train")
    val_data = TinyImageNet(split="val")
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=8,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=8,
        persistent_workers=True,
    )
    return train_loader, val_loader


class MNIST(datasets.MNIST):
    def __init__(self, train, normalise=True, flatten=True, save_dir=DATA_DIR):
        self.flatten = flatten
        if normalise:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.1307), std=(0.3081)
                    )
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(save_dir, download=True, train=train, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        if self.flatten:
            img = torch.flatten(img)
        label = one_hot(label, n_classes=10)
        return img, label


class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, train, normalise=True, flatten=True, save_dir=DATA_DIR):
        self.flatten = flatten
        if normalise:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.5), std=(0.5)
                    )
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(save_dir, download=True, train=train, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        if self.flatten:
            img = torch.flatten(img)
        label = one_hot(label)
        return img, label


class CIFAR10(datasets.CIFAR10):
    def __init__(self, train, normalise=True, flatten=True, save_dir=f"{DATA_DIR}/CIFAR10"):
        self.flatten = flatten
        if normalise:
            if train:
                transform = transforms.Compose(
                    [
                        transforms.Resize((32,32)), 
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(), 
                        transforms.RandomRotation(10),
                        transforms.ToTensor(), 
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.Resize((32,32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(save_dir, download=True, train=train, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        if self.flatten:
            img = torch.flatten(img)
        label = one_hot(label)
        return img, label


class ImageNet(datasets.ImageNet):
    def __init__(self, split):
        if split == "train":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])
        elif split == "val":
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        super().__init__(root=IMAGENET_DIR, split=split, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        label = one_hot(label, n_classes=1000)
        return img, label


class TinyImageNet(Dataset):
    def __init__(self, split, root=TINYIMAGENET_DIR):
        self.root = root
        self.split = split
        self.loader = default_loader
        self.classes = self._load_classes()
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.transform = self._make_transform(split)
        self.samples = self._build_samples()

    def _load_classes(self):
        wnids_path = os.path.join(self.root, "wnids.txt")
        if os.path.isfile(wnids_path):
            with open(wnids_path, "r", encoding="utf-8") as handle:
                return [line.strip() for line in handle if line.strip()]

        train_dir = Path(self.root) / "train"
        return sorted(path.name for path in train_dir.iterdir() if path.is_dir())

    def _make_transform(self, split):
        if split == "train":
            return transforms.Compose(
                [
                    transforms.RandomCrop(64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        if split == "val":
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        raise ValueError(f"Unsupported TinyImageNet split: {split}")

    def _build_samples(self):
        if self.split == "train":
            samples = []
            train_dir = Path(self.root) / "train"
            for class_name in self.classes:
                image_dir = train_dir / class_name / "images"
                if not image_dir.is_dir():
                    continue
                for image_path in sorted(image_dir.iterdir()):
                    if image_path.is_file():
                        samples.append((str(image_path), self.class_to_idx[class_name]))
            return samples

        val_dir = Path(self.root) / "val"
        images_dir = val_dir / "images"
        annotations_path = val_dir / "val_annotations.txt"
        if not annotations_path.is_file():
            raise FileNotFoundError(
                f"TinyImageNet validation annotations not found: '{annotations_path}'."
            )

        samples = []
        with open(annotations_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, class_name = parts[0], parts[1]
                if class_name not in self.class_to_idx:
                    continue
                samples.append(
                    (str(images_dir / image_name), self.class_to_idx[class_name])
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = self.loader(img_path)
        img = self.transform(img)
        label = one_hot(label, n_classes=len(self.classes))
        return img, label


def one_hot(labels, n_classes=10):
    arr = torch.eye(n_classes)
    return arr[labels]
