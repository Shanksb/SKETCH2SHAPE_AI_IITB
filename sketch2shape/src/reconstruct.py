import sys
import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.append("third_party/TripoSR")
from tsr.system import TSR
from tsr.utils import resize_foreground

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None


def _load_model():
    """Load weights once and reuse them."""
    global _model
    if _model is None:
        m = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        m.renderer.set_chunk_size(8192)  # lower to 4096 if out of memory
        m.to(DEVICE)
        _model = m
    return _model


def _orient(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate TripoSR's native frame so the object stands upright (Y-up).
    Verify on the chair; if it is still off, change the angle/axis here."""
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return mesh


def reconstruct(img: Image.Image, mc_resolution: int = 256,
                foreground_ratio: float = 0.85) -> trimesh.Trimesh:
    """RGBA image (background removed) -> coloured trimesh."""
    if img.mode != "RGBA":
        raise ValueError("reconstruct() expects an RGBA image with the background removed")

    model = _load_model()

    img = resize_foreground(img, foreground_ratio)
    arr = np.array(img).astype(np.float32) / 255.0
    rgb = arr[..., :3] * arr[..., 3:4] + 0.5 * (1.0 - arr[..., 3:4])  # grey background
    image = Image.fromarray((rgb * 255).astype(np.uint8))

    with torch.no_grad():
        codes = model([image], device=DEVICE)
        meshes = model.extract_mesh(codes, True, resolution=mc_resolution)
    torch.cuda.empty_cache()
    return _orient(meshes[0])