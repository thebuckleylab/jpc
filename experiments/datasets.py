import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_DIR = "datasets"


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
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.247, 0.243, 0.261)
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


def one_hot(labels, n_classes=10):
    arr = torch.eye(n_classes)
    return arr[labels]
