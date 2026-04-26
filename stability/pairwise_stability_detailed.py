import numpy as np


class PairResult:
    def __init__(self, i, j, flip_rate, mean_margin, signs):
        """
        Result for a single pair (i, j).

        i, j : int           region indices, i < j
        flip_rate : float     fraction of transforms where sign flipped
        mean_margin : float   average |d[i] - d[j]| across original + all transforms
        signs : list[int]     sign per test: signs[0] = original, rest = transforms
        """
        self.i = i
        self.j = j
        self.flip_rate = flip_rate
        self.mean_margin = mean_margin
        self.signs = signs


class StabilityResult:
    def __init__(self, pair_results, segment_func_name, num_regions, num_transforms):
        self.pair_results = pair_results
        self.segment_func_name = segment_func_name
        self.num_regions = num_regions
        self.num_transforms = num_transforms


def compute_pairwise_signs(descriptors):
    """
    Compute sign(d[i] - d[j]) for all pairs i < j.

    Returns dict: (i, j) -> +1, -1, or 0 (if equal).
    """
    signs = {}
    n = len(descriptors)
    for i in range(n):
        for j in range(i + 1, n):
            diff = descriptors[i] - descriptors[j]
            if diff > 0:
                signs[(i, j)] = 1
            elif diff < 0:
                signs[(i, j)] = -1
            else:
                signs[(i, j)] = 0
    return signs


def compute_pairwise_margins(descriptors):
    """
    Compute |d[i] - d[j]| for all pairs i < j.
    """
    margins = {}
    n = len(descriptors)
    for i in range(n):
        for j in range(i + 1, n):
            margins[(i, j)] = abs(descriptors[i] - descriptors[j])
    return margins


def run_stability_test(image, segment_func, descriptor_func, transformations, segment_func_name="unknown"):
    """
    Run the full pairwise stability pipeline.

    Parameters
    ----------
    image : np.ndarray (H, W, C) BGR
    segment_func : callable(image) -> SegmentationResult
    descriptor_func : callable(image, SegmentationResult) -> list[float]
    transformations : list[tuple[str, callable]]
    segment_func_name : str

    Returns
    -------
    StabilityResult
    """
    # Original
    seg_orig = segment_func(image)
    desc_orig = descriptor_func(image, seg_orig)
    num_regions = seg_orig.num_regions

    signs_orig = compute_pairwise_signs(desc_orig)
    margins_orig = compute_pairwise_margins(desc_orig)

    # Track per-pair: list of signs and margins across transforms
    all_signs = {pair: [signs_orig[pair]] for pair in signs_orig}
    all_margins = {pair: [margins_orig[pair]] for pair in margins_orig}
    flip_counts = {pair: 0 for pair in signs_orig}

    for name, transform_fn in transformations:
        img_t = transform_fn(image)
        seg_t = segment_func(img_t)
        desc_t = descriptor_func(img_t, seg_t)
        num_regions_t = seg_t.num_regions

        signs_t = compute_pairwise_signs(desc_t)
        margins_t = compute_pairwise_margins(desc_t)

        for pair in signs_orig:
            i, j = pair
            # If either region doesn't exist after transform, count as flip
            if i >= num_regions_t or j >= num_regions_t:
                flip_counts[pair] += 1
                all_signs[pair].append(0)
                all_margins[pair].append(0.0)
            else:
                all_signs[pair].append(signs_t.get(pair, 0))
                all_margins[pair].append(margins_t.get(pair, 0.0))
                if signs_t.get(pair, 0) != signs_orig[pair]:
                    flip_counts[pair] += 1

    # Build PairResults
    num_transforms = len(transformations)
    pair_results = []
    for pair in signs_orig:
        i, j = pair
        flip_rate = flip_counts[pair] / num_transforms if num_transforms > 0 else 0.0
        mean_margin = np.mean(all_margins[pair])
        pair_results.append(PairResult(i, j, flip_rate, mean_margin, all_signs[pair]))

    return StabilityResult(pair_results, segment_func_name, num_regions, num_transforms)


def select_stable_pairs(stability_result, max_flip_rate=0.0, min_margin=5.0):
    """
    Filter and rank pairs by reliability.

    Returns list[PairResult] sorted by mean_margin descending.
    """
    selected = [
        pr for pr in stability_result.pair_results
        if pr.flip_rate <= max_flip_rate and pr.mean_margin >= min_margin
    ]
    selected.sort(key=lambda pr: pr.mean_margin, reverse=True)
    return selected


def print_stability_report(stability_result, top_n=20):
    """Print a human-readable summary."""
    sr = stability_result
    total_pairs = len(sr.pair_results)

    print(f"\nStability Report: {sr.segment_func_name}")
    print(f"  Regions: {sr.num_regions}")
    print(f"  Transforms: {sr.num_transforms}")
    print(f"  Total pairs: {total_pairs}")

    if total_pairs == 0:
        print("  No pairs to analyze.")
        return

    # Count perfectly stable pairs
    perfect = [pr for pr in sr.pair_results if pr.flip_rate == 0.0]
    print(f"  Perfectly stable pairs (0% flip): {len(perfect)} / {total_pairs}")

    # Sort by stability: low flip rate first, then high margin
    sorted_pairs = sorted(sr.pair_results, key=lambda pr: (pr.flip_rate, -pr.mean_margin))

    print(f"\n  Top {min(top_n, total_pairs)} most stable pairs:")
    print(f"  {'Pair':>10}  {'Flip Rate':>10}  {'Mean Margin':>12}  Signs")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*20}")

    for pr in sorted_pairs[:top_n]:
        signs_str = " ".join(f"{s:+d}" for s in pr.signs)
        print(f"  ({pr.i:3d},{pr.j:3d})  {pr.flip_rate:10.2f}  {pr.mean_margin:12.2f}  [{signs_str}]")

    # Show worst pairs
    worst_n = min(5, total_pairs)
    print(f"\n  Bottom {worst_n} least stable pairs:")
    for pr in sorted_pairs[-worst_n:]:
        signs_str = " ".join(f"{s:+d}" for s in pr.signs)
        print(f"  ({pr.i:3d},{pr.j:3d})  {pr.flip_rate:10.2f}  {pr.mean_margin:12.2f}  [{signs_str}]")
