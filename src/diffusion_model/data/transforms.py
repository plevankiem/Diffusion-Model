from __future__ import annotations

from typing import Dict, Literal, Optional

from torchvision import transforms


DatasetName = Literal["cifar10"]


_DATASET_MEAN_STD: Dict[str, Dict[str, tuple]] = {
    """
    "dataset_name": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    """
}


def get_normalization(dataset_name: str):
    """Retourne la transformation de normalisation pour un dataset donné."""
    name = dataset_name.lower()
    if name not in _DATASET_MEAN_STD:
        return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    stats = _DATASET_MEAN_STD[name]
    return transforms.Normalize(mean=stats["mean"], std=stats["std"])


def _base_image_transform(
    image_size: Optional[int],
    normalize,
) -> transforms.Compose:
    """Construit un pipeline de base pour des images."""
    t = []

    if image_size is not None:
        t.append(transforms.Resize(image_size))

    t.append(transforms.ToTensor())
    t.append(normalize)

    return transforms.Compose(t)


def build_transform_for_split(
    dataset_name: DatasetName,
    image_size: Optional[int] = None,
):
    """Construit un transform adapté à un dataset et un split (train/val/test).

    - Pour `train`, on applique des augmentations légères si `augment_train` est True.
    - Pour `val` et `test`, on applique uniquement des transformations déterministes.
    """
    name = dataset_name.lower()
    normalize = get_normalization(name)

    return _base_image_transform(
        image_size=image_size,
        normalize=normalize,
    )