from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional, Tuple, Union

from PIL import Image
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

    # Paramètres optionnels communs
    image_size: Optional[int] = None
    dataset_path: Optional[Union[str, Path]] = None  # Si rempli, correspond à root/"raw"/{dataset_name}


def _build_cifar10(
    cfg: DatasetConfig,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Dataset:
    """Construit un dataset CIFAR10 à partir des fichiers existants dans raw/cifar10/.
    
    Structure attendue:
        root/raw/cifar10/cifar-10-batches-py/
            data_batch_1, data_batch_2, ..., data_batch_5 (pour train)
            test_batch (pour test)
    """
    if cfg.split not in ("train", "test"):
        raise ValueError(
            f"Split '{cfg.split}' non valide pour CIFAR10. "
            "Utilisez 'train' ou 'test'."
        )

    is_train = cfg.split == "train"
    # Les données sont dans root/raw/cifar10/ ou dans dataset_path si fourni
    # torchvision CIFAR10 s'attend à trouver cifar-10-batches-py/ dans le dossier root
    if cfg.dataset_path is not None:
        dataset_root = Path(cfg.dataset_path)
    else:
        dataset_root = Path(cfg.root) / "raw" / "cifar10"
    
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Le dossier CIFAR10 n'existe pas à: {dataset_root}\n"
            "Assurez-vous que les données CIFAR10 sont présentes dans raw/cifar10/"
        )
    
    batches_dir = dataset_root / "cifar-10-batches-py"
    if not batches_dir.exists():
        raise FileNotFoundError(
            f"Le dossier cifar-10-batches-py n'existe pas dans {dataset_root}\n"
            "Structure attendue: root/raw/cifar10/cifar-10-batches-py/"
        )
    
    return datasets.CIFAR10(
        root=str(dataset_root),
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=False,  # Pas de téléchargement, les données sont déjà là
    )

class ImageFolderDataset(Dataset):
    """Dataset qui charge des images depuis un dossier, retournant (image, 0).
    
    Utile pour des datasets non supervisés où les labels ne sont pas nécessaires.
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        
        if not self.root.exists():
            raise FileNotFoundError(f"Le dossier {self.root} n'existe pas")
        
        # Récupérer toutes les images
        self.image_paths = sorted([
            p for p in self.root.iterdir() 
            if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        ])
        
        if len(self.image_paths) == 0:
            raise ValueError(f"Aucune image trouvée dans {self.root}")
    
    def __getitem__(self, index: int) -> Tuple:
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        # Retourner (image, 0) pour compatibilité avec les datasets supervisés
        label = 0
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return (img, label)
    
    def __len__(self) -> int:
        return len(self.image_paths)


def _build_celeba(
    cfg: DatasetConfig,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Dataset:
    """Construit un dataset CelebA à partir des fichiers existants dans raw/celeba/.
    
    Structure attendue:
        root/raw/celeba/train/  (images .jpg)
        root/raw/celeba/test/   (images .jpg)
    """
    if cfg.split not in ("train", "test"):
        raise ValueError(
            f"Split '{cfg.split}' non valide pour CelebA. "
            "Utilisez 'train' ou 'test'."
        )
    
    # Les données sont dans root/raw/celeba/{split}/ ou dans dataset_path/{split}/ si fourni
    if cfg.dataset_path is not None:
        split_dir = Path(cfg.dataset_path) / cfg.split
    else:
        split_dir = Path(cfg.root) / "raw" / "celeba" / cfg.split
    
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Le dossier CelebA {cfg.split} n'existe pas à: {split_dir}\n"
            "Structure attendue: root/raw/celeba/train/ et root/raw/celeba/test/"
        )
    
    return ImageFolderDataset(
        root=split_dir,
        transform=transform,
        target_transform=target_transform,
    )


_DATASET_BUILDERS: Dict[str, Callable[..., Dataset]] = {
    "cifar10": _build_cifar10,
    "celeba": _build_celeba,
}


def get_supported_datasets() -> Iterable[str]:
    """Retourne la liste des noms de datasets supportés."""
    return sorted(_DATASET_BUILDERS.keys())


def get_default_image_size(dataset_name: str, root: Optional[Union[str, Path]] = None) -> int:
    """Retourne la taille d'image par défaut (originale) pour un dataset.
    
    Args:
        dataset_name: Nom du dataset (ex: "cifar10", "celeba")
        root: Optionnel, racine des données pour détecter dynamiquement la taille
        
    Returns:
        Taille d'image par défaut (dimension pour images carrées, ou hauteur pour images rectangulaires)
    """
    name = dataset_name.lower()
    
    # Tailles standards connues
    default_sizes = {
        "cifar10": 32,  # CIFAR10 est 32x32
        "celeba": 64,  # CelebA original images sont 178x218 (width x height), on retourne la hauteur
    }
    
    if name in default_sizes:
        return default_sizes[name]
    
    # Pour CelebA, on essaie de détecter la taille depuis les fichiers si root est fourni
    if name == "celeba" and root is not None:
        try:
            celeba_root = Path(root) / "raw" / "celeba" / "train"
            if celeba_root.exists():
                # Trouver une image et charger sa taille
                image_files = sorted([
                    p for p in celeba_root.iterdir()
                    if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
                ])
                if image_files:
                    img = Image.open(image_files[0])
                    width, height = img.size
                    # Retourner la hauteur (dimension verticale) comme taille par défaut
                    return height
        except Exception:
            # En cas d'erreur, utiliser la valeur par défaut
            pass
    
    # Valeur par défaut si le dataset n'est pas reconnu
    return default_sizes.get(name, 32)


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
    image_size: Optional[int] = None,
    dataset_path: Optional[Union[str, Path]] = None,
) -> Dict[DatasetSplit, Dataset]:
    """Crée un dictionnaire {\"train\", \"val\", \"test\"} de datasets.

    - `transforms_by_split` doit contenir au minimum une clé \"train\".
    - Si \"val\" ou \"test\" ne sont pas fournis, ils seront ignorés.
    - Les données sont chargées depuis root/raw/{dataset_name}/, pas de téléchargement.
    - Si `dataset_path` est fourni, il sera utilisé directement (équivalent à root/raw/{dataset_name}).
    """
    datasets_dict: Dict[DatasetSplit, Dataset] = {}

    for split in ("train", "val", "test"):
        if split not in transforms_by_split:
            continue

        cfg = DatasetConfig(
            name=name,
            root=root,
            split=split,  # type: ignore[arg-type]
            image_size=image_size,
            dataset_path=dataset_path,
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
    image_size = 32
    transform_by_split = {
        "train": None,
        "test": None,
    }
    datasets = create_train_val_test_datasets(name, root, transform_by_split, image_size)
    print(datasets["train"])