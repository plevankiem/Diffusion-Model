from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional, Tuple, Union

from torch.utils.data import Dataset
from torchvision import datasets


DatasetSplit = Literal["train", "val", "test"]


@dataclass
class DatasetConfig:
    """Configuration minimale pour instancier un dataset.

    Cette classe est volontairement indépendante du système de config global.
    Vous pourrez la relier plus tard à `DataConfig` dans `config.py`.
    """

    name: str  # ex: "cifar10", "image_folder"
    root: Union[str, Path]
    split: DatasetSplit = "train"
    download: bool = True

    # Paramètres optionnels communs
    image_size: Optional[int] = None


def _build_cifar10(
    cfg: DatasetConfig,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Dataset:
    """Construit un dataset CIFAR10 à partir d'une DatasetConfig."""
    if cfg.split not in ("train", "test"):
        raise ValueError(
            f"Split '{cfg.split}' non valide pour CIFAR10. "
            "Utilisez 'train' ou 'test'."
        )

    is_train = cfg.split == "train"
    return datasets.CIFAR10(
        root=str(Path(cfg.root) / "raw"),
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=cfg.download,
    )

def _build_celeba(
    cfg: DatasetConfig,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Dataset:
    """Construit un dataset CelebA à partir d'une DatasetConfig."""
    return datasets.CelebA(
        root=str(Path(cfg.root) / "raw"),
        split=cfg.split,
        transform=transform,
        target_transform=target_transform,
        download=cfg.download,
    )

_DATASET_BUILDERS: Dict[str, Callable[..., Dataset]] = {
    "cifar10": _build_cifar10,
    "celeba": _build_celeba,
}


def get_supported_datasets() -> Iterable[str]:
    """Retourne la liste des noms de datasets supportés."""
    return sorted(_DATASET_BUILDERS.keys())


def create_dataset(
    cfg: DatasetConfig,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Dataset:
    """Fabrique un dataset à partir d'une DatasetConfig et d'un transform.

    Exemple d'usage:

        cfg = DatasetConfig(name="cifar10", root="data", split="train")
        transform = build_transform_for_split("cifar10", "train")
        dataset = create_dataset(cfg, transform=transform)
    """
    name = cfg.name.lower()
    if name not in _DATASET_BUILDERS:
        supported = ", ".join(get_supported_datasets())
        raise ValueError(
            f"Dataset '{cfg.name}' non supporté. Datasets disponibles: {supported}"
        )

    builder = _DATASET_BUILDERS[name]
    return builder(cfg=cfg, transform=transform, target_transform=target_transform)


def create_train_val_test_datasets(
    name: str,
    root: Union[str, Path],
    transforms_by_split: Dict[DatasetSplit, Callable],
    download: bool = True,
    image_size: Optional[int] = None,
) -> Dict[DatasetSplit, Dataset]:
    """Crée un dictionnaire {\"train\", \"val\", \"test\"} de datasets.

    - `transforms_by_split` doit contenir au minimum une clé \"train\".
    - Si \"val\" ou \"test\" ne sont pas fournis, ils seront ignorés.
    """
    datasets_dict: Dict[DatasetSplit, Dataset] = {}

    for split in ("train", "val", "test"):
        if split not in transforms_by_split:
            continue

        cfg = DatasetConfig(
            name=name,
            root=root,
            split=split,  # type: ignore[arg-type]
            download=download,
            image_size=image_size,
        )
        ds = create_dataset(cfg, transform=transforms_by_split[split])
        datasets_dict[split] = ds

    if "train" not in datasets_dict:
        raise ValueError(
            "Aucun dataset 'train' créé. Vérifiez `transforms_by_split['train']`."
        )
    return datasets_dict

if __name__ == "__main__":
    name = "cifar10"
    root = "data"
    download = True
    image_size = 32
    transform_by_split = {
        "train": None,
        "test": None,
    }
    datasets = create_train_val_test_datasets(name, root, transform_by_split, download, image_size)
    print(datasets["train"])