"""MVTec AD data loading.

Expected layout:
    <root>/<category>/train/good/*.png                       normal images only
    <root>/<category>/test/<good | defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/<stem>_mask.png
"""
from pathlib import Path

# src/anomaly/data.py -> parents[2] is the project root
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "mvtec_ad"


def _category_dir(category, root):
    cat_dir = Path(root) / category
    if not cat_dir.is_dir():
        raise FileNotFoundError(f"category folder not found: {cat_dir}")
    return cat_dir


def list_train(category, root=DEFAULT_ROOT):
    """Sorted paths of all defect-free training images."""
    return sorted((_category_dir(category, root) / "train" / "good").glob("*.png"))


def list_test(category, root=DEFAULT_ROOT):
    """List of (image_path, label, mask_path) for every test image.

    label: 0 = good, 1 = defective.
    mask_path: None for good images (their mask is all zeros).
    """
    cat_dir = _category_dir(category, root)
    samples = []
    for defect_dir in sorted(d for d in (cat_dir / "test").iterdir() if d.is_dir()):
        defect = defect_dir.name
        for img_path in sorted(defect_dir.glob("*.png")):
            if defect == "good":
                samples.append((img_path, 0, None))
                continue
            # test/cut/003.png  <->  ground_truth/cut/003_mask.png
            mask_path = cat_dir / "ground_truth" / defect / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"no mask for {img_path}: expected {mask_path}")
            samples.append((img_path, 1, mask_path))
    return samples
