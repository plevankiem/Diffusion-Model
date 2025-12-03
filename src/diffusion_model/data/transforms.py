from __future__ import annotations

from typing import Dict, Literal, Optional

from torchvision import transforms


DatasetName = Literal["cifar10", "imagenet", "celeba"]


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
    make_square: bool = False,
) -> transforms.Compose:
    """Construit un pipeline de base pour des images.
    
    Args:
        image_size: Taille cible des images (None pour ne pas redimensionner)
        normalize: Transformation de normalisation
        make_square: Si True, redimensionne/recadre pour obtenir des images carrées
    """
    t = []

    if image_size is not None:
        if make_square:
            # Pour des images carrées: redimensionner en gardant le ratio, puis recadrer au centre
            # On redimensionne d'abord en gardant le ratio pour que le plus petit côté soit image_size
            t.append(transforms.Resize(image_size, antialias=True))
            # Puis on recadre au centre pour obtenir une image carrée
            t.append(transforms.CenterCrop(image_size))
        else:
            t.append(transforms.Resize(image_size, antialias=True))

    t.append(transforms.ToTensor())
    t.append(normalize)

    return transforms.Compose(t)


def build_transform_for_split(
    dataset_name: DatasetName,
    image_size: Optional[int] = None,
    make_square: bool = False,
):
    """Construit un transform adapté à un dataset et un split (train/val/test).

    - Pour `train`, on applique des augmentations légères si `augment_train` est True.
    - Pour `val` et `test`, on applique uniquement des transformations déterministes.
    
    Args:
        dataset_name: Nom du dataset
        image_size: Taille cible des images (None pour ne pas redimensionner)
        make_square: Si True, redimensionne/recadre pour obtenir des images carrées
    """
    name = dataset_name.lower()
    normalize = get_normalization(name)

    return _base_image_transform(
        image_size=image_size,
        normalize=normalize,
        make_square=make_square,
    )