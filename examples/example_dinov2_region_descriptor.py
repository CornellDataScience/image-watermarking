"""
Example usage for DinoV2WholeImageRegionDescriptor.

This script creates a simple 2x2 segmentation map, runs DINOv2 once on the full
image, pools patch embeddings per region, and prints descriptor diagnostics.

Examples
--------
    python examples/example_dinov2_region_descriptor.py
    python examples/example_dinov2_region_descriptor.py --image data/dog.jpg --device cpu
    python examples/example_dinov2_region_descriptor.py --target-regions 0 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from descriptors.dinov2_region_descriptor import DinoV2WholeImageRegionDescriptor


def load_sample_image(path: str | None) -> Image.Image:
    """Load an RGB sample image, falling back to data/dog.jpg or a synthetic image."""

    if path is not None:
        return Image.open(path).convert("RGB")

    default_path = Path("data/dog.jpg")
    if default_path.exists():
        return Image.open(default_path).convert("RGB")

    h, w = 256, 256
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    image = np.stack(
        [
            np.repeat(x, h, axis=0),
            np.repeat(y, w, axis=1),
            ((np.repeat(x, h, axis=0).astype(np.uint16) + np.repeat(y, w, axis=1)) // 2).astype(np.uint8),
        ],
        axis=2,
    )
    return Image.fromarray(image, mode="RGB")


def create_quadrant_segmentation(height: int, width: int) -> np.ndarray:
    """Create four segmentation regions with IDs 0, 1, 2, and 3."""

    seg = np.zeros((height, width), dtype=np.int64)
    mid_y = height // 2
    mid_x = width // 2
    seg[:mid_y, :mid_x] = 0
    seg[:mid_y, mid_x:] = 1
    seg[mid_y:, :mid_x] = 2
    seg[mid_y:, mid_x:] = 3
    return seg


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom.item()) == 0:
        return 0.0
    return float(F.cosine_similarity(a[None], b[None]).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINOv2 whole-image region descriptors.")
    parser.add_argument("--image", default=None, help="Path to an RGB image. Defaults to data/dog.jpg if present.")
    parser.add_argument("--model", default="facebook/dinov2-base", help="Hugging Face DINOv2 checkpoint.")
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision or commit hash. Use a commit hash for strict reproducibility.",
    )
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Execution device.")
    parser.add_argument("--input-size", type=int, default=224, help="Square model input size, divisible by 14.")
    parser.add_argument(
        "--pooling-mode",
        default="mean",
        choices=["mean", "area_weighted_mean", "max"],
        help="Region pooling mode.",
    )
    parser.add_argument(
        "--patch-assignment-mode",
        default="soft_area_weights",
        choices=["soft_area_weights", "majority_region"],
        help="Patch-to-region assignment policy.",
    )
    parser.add_argument(
        "--scalar-mode",
        default="fixed_random_projection",
        choices=[
            "fixed_random_projection",
            "cosine_to_anchor",
            "first_principal_component_projection_given_embeddings",
        ],
        help="Optional scalar reduction mode used in this example.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Seed for fixed random projection.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 autocast on CUDA.")
    parser.add_argument(
        "--target-regions",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of region IDs to describe.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load only cached Hugging Face artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_sample_image(args.image)
    width, height = image.size
    segmentation = create_quadrant_segmentation(height, width)

    descriptor = DinoV2WholeImageRegionDescriptor(
        model_name=args.model,
        model_revision=args.revision,
        device=args.device,
        input_size=args.input_size,
        pooling_mode=args.pooling_mode,
        patch_assignment_mode=args.patch_assignment_mode,
        scalar_mode=args.scalar_mode,
        random_seed=args.seed,
        use_fp16=args.fp16,
        local_files_only=args.local_files_only,
    )

    result = descriptor.describe_image(
        image=image,
        segmentation=segmentation,
        return_patch_embeddings=True,
        return_scalars=True,
        target_region_ids=args.target_regions,
    )

    assert result.patch_embeddings is not None
    print(f"patch embedding shape: {tuple(result.patch_embeddings.shape)}")
    print(f"valid regions: {len(result.region_embeddings)}")

    if result.region_embeddings:
        first_region = next(iter(result.region_embeddings))
        first_embedding = result.region_embeddings[first_region]
        print(f"region embedding shape: {tuple(first_embedding.shape)}")

    if result.scalar_descriptors is not None:
        sample_scalars = list(result.scalar_descriptors.items())[:4]
        print(f"sample scalar descriptors: {sample_scalars}")

    region_ids = list(result.region_embeddings.keys())
    if len(region_ids) >= 2:
        a, b = region_ids[0], region_ids[1]
        sim = cosine(result.region_embeddings[a], result.region_embeddings[b])
        print(f"cosine similarity region {a} vs {b}: {sim:.4f}")
    if len(region_ids) >= 3:
        a, b = region_ids[0], region_ids[2]
        sim = cosine(result.region_embeddings[a], result.region_embeddings[b])
        print(f"cosine similarity region {a} vs {b}: {sim:.4f}")

    subset = descriptor.describe_image(
        image=image,
        segmentation=segmentation,
        return_patch_embeddings=False,
        return_scalars=False,
        target_region_ids=[0, 2],
    )
    print(f"subset request [0, 2] returned regions: {list(subset.region_embeddings.keys())}")


if __name__ == "__main__":
    main()
