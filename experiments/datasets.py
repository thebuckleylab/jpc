import jax.random as jr
import jax.numpy as jnp

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_DIR = "datasets"
IMAGENET_DIR = f"~/projects/jpc/experiments/{DATA_DIR}/ImageNet"


def make_gaussian_dataset(key, mean, std, shape):
    x = mean + std * jr.normal(key, shape)
    y = x
    return (x, y)


def get_dataloaders(dataset_id, batch_size, flatten=True):
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
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
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
            "Invalid dataset ID. Options are 'MNIST', 'Fashion-MNIST' and 'CIFAR10'"
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
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=10),
                    #transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
                    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2470, 0.2435, 0.2616)
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


def one_hot(labels, n_classes=10):
    arr = torch.eye(n_classes)
    return arr[labels]
