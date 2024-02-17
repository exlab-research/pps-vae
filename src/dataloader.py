import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision import datasets, transforms
from datasets import load_dataset

from PIL import Image

import csv

import pathlib


def _get_kwargs(device):
    return {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}


def fmnist_loader(data_dir, batch_size, shuffle=True, device="cuda", tx_train=transforms.ToTensor(), tx_test=transforms.ToTensor()):

    train = DataLoader(datasets.FashionMNIST(data_dir, train=True, download=True, transform=tx_train),
                       batch_size=batch_size, shuffle=shuffle, **_get_kwargs(device))
    test = DataLoader(datasets.FashionMNIST(data_dir, train=False, download=True, transform=tx_test),
                      batch_size=batch_size, shuffle=False, **_get_kwargs(device))
    return train, test


def cifar10_loader(data_dir, batch_size, shuffle=True, device='cuda', tx_train=transforms.ToTensor(), tx_test=transforms.ToTensor()):
    # Note that this loads the data _without_ the recommended Normalize transform
    # that transform gets applied at the start of the classifer fwd pass
    # tx = transforms.Compose([transforms.ToTensor(),
    #                          transforms.Normalize([0.4914, 0.4822, 0.4465],
    #                                               [0.2023, 0.1994, 0.2010])])

    train = DataLoader(datasets.cifar.CIFAR10(data_dir, train=True, download=False, transform=tx_train),
                       batch_size=batch_size, shuffle=shuffle, **_get_kwargs(device))
    test = DataLoader(datasets.cifar.CIFAR10(data_dir, train=False, download=False, transform=tx_test),
                      batch_size=batch_size, shuffle=False, **_get_kwargs(device))
    return train, test


def celeba_loader(data_dir, batch_size, shuffle=True, device="cuda", img_dim=64):
    tx_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(size=(img_dim, img_dim))])
    tx_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(size=(img_dim, img_dim))])

    train = DataLoader(datasets.CelebA(data_dir, split='train', download=False, transform=tx_train),
                       batch_size=batch_size, shuffle=shuffle, **_get_kwargs(device))
    test = DataLoader(datasets.CelebA(data_dir, split='test', download=False, transform=tx_test),
                      batch_size=batch_size, shuffle=False, **_get_kwargs(device))
    return train, test


def tiny_imagenet_loader(data_dir, batch_size, shuffle=True, device='cuda', img_dim=64):
    if img_dim == 64:
        tx_train = transforms.ToTensor()
        tx_test = transforms.ToTensor()
    else:
        tx_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(size=(img_dim, img_dim))])
        tx_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(size=(img_dim, img_dim))])

    def transform(examples):
        examples["pixel_values"] = [
            tx_train(image.convert("RGB")) for image in examples["image"]]
        return examples

    def collate_fn(examples):

        images = []
        labels = []
        for example in examples:
            # print(example.keys())
            images.append((example["pixel_values"]))
            labels.append(example["label"])

        pixel_values = torch.stack(images)
        labels = torch.tensor(labels)
        return (pixel_values, labels)

    tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
    tiny_imagenet_train.set_format(type="torch", columns=["image", "label"])
    tiny_imagenet_train = tiny_imagenet_train.with_transform(transform)
    train = DataLoader(tiny_imagenet_train, batch_size=batch_size,
                       collate_fn=collate_fn, shuffle=shuffle, **_get_kwargs(device))

    tiny_imagenet_val = load_dataset('Maysee/tiny-imagenet', split='valid')
    tiny_imagenet_val.set_format(type="torch", columns=["image", "label"])
    tiny_imagenet_val = tiny_imagenet_val.with_transform(transform)
    test = DataLoader(tiny_imagenet_val, batch_size=batch_size,
                      collate_fn=collate_fn, shuffle=True, **_get_kwargs(device))
    return train, test


def clevr_loader(data_dir, batch_size, shuffle=True, device="cuda", img_dim=64):
    tx_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(size=(img_dim, img_dim))])
    tx_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(size=(img_dim, img_dim))])

    train = DataLoader(datasets.CLEVRClassification(data_dir, split='train', download=True, transform=tx_train),
                       batch_size=batch_size, shuffle=shuffle, **_get_kwargs(device))
    test = DataLoader(datasets.CLEVRClassification(data_dir, split='val', download=True, transform=tx_test),
                      batch_size=batch_size, shuffle=False, **_get_kwargs(device))
    return train, test


def fer2013_loader(data_dir, batch_size, shuffle=True, device="cuda", img_dim=64):
    tx_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(size=(img_dim, img_dim))])
    tx_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(size=(img_dim, img_dim))])

    train = DataLoader(FER2013(data_dir, split='train', transform=tx_train),
                       batch_size=batch_size, shuffle=shuffle, **_get_kwargs(device))
    test = DataLoader(FER2013(data_dir, split='test', transform=tx_test),
                      batch_size=batch_size, shuffle=False, **_get_kwargs(device))
    return train, test


class FER2013(VisionDataset):

    _RESOURCES = {
        "train": "train.csv",
        "test": "test.csv",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
    ) -> None:
        self._split = split
        super().__init__(root, transform=transform, target_transform=target_transform)

        base_folder = pathlib.Path(self.root) / "fer2013"
        file_name = self._RESOURCES[self._split]
        data_file = base_folder / file_name

        with open(data_file, "r", newline="") as file:
            self._samples = [
                (
                    torch.tensor(
                        [int(idx) for idx in row["pixels"].split()], dtype=torch.uint8).reshape(48, 48),
                    int(row["emotion"]) if "emotion" in row else None,
                )
                for row in csv.DictReader(file)
            ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        image_tensor, target = self._samples[idx]
        image = Image.fromarray(image_tensor.numpy())

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def extra_repr(self):
        return f"split={self._split}"
