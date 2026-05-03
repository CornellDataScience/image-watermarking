from dataclasses import dataclass
from typing import List, Optional
import numpy as np
 
@dataclass
class PairResult:
    image_id: str
    pair_id: tuple[int, int]
    before_margin: float
    after_margin: Optional[float]
    status: str
    embedding_cosine_similarity: Optional[float] = None
    
    @property
    def segmentation_survived(self) -> bool:
        return self.status in ['stable', 'descriptor_flip']
    
    @property
    def sign_flipped(self) -> Optional[bool]:
        if self.status == 'segmentation_failure':
            return None
        return self.status == 'descriptor_flip'
    
    def is_usable(self) -> bool:
        return self.status == 'stable'
 
@dataclass
class EvaluationMetrics:
    mean_flip_rate: float
    mean_margin_retention: float
    segmentation_survival_rate: float
    usable_pair_yield: float
    total_pairs: int
    survived_pairs: int
    flipped_pairs: int
    stable_pairs: int
    num_images: int
    survived_above_threshold_count: int
    
    def __repr__(self) -> str:
        return (
            f"EvaluationMetrics(\n"
            f"  mean_flip_rate={self.mean_flip_rate:.4f},\n"
            f"  mean_margin_retention={self.mean_margin_retention:.4f},\n"
            f"  segmentation_survival_rate={self.segmentation_survival_rate:.4f},\n"
            f"  usable_pair_yield={self.usable_pair_yield:.2f},\n"
            f"  total_pairs={self.total_pairs},\n"
            f"  survived_pairs={self.survived_pairs},\n"
            f"  flipped_pairs={self.flipped_pairs},\n"
            f"  stable_pairs={self.stable_pairs},\n"
            f"  num_images={self.num_images},\n"
            f"  survived_above_threshold_count={self.survived_above_threshold_count}\n"
            f")"
        )
 
def compute_metrics(
    pair_results: List[PairResult],
    min_margin_threshold: float = 0.0
) -> EvaluationMetrics:
    if not pair_results:
        raise ValueError("Cannot compute metrics on empty pair_results list")
    
    total_pairs = len(pair_results)
    num_images = len(set(pr.image_id for pr in pair_results))
    
    survived = [pr for pr in pair_results if pr.segmentation_survived]
    survived_pairs = len(survived)
    
    survived_above_threshold = [
        pr for pr in survived 
        if pr.before_margin >= min_margin_threshold
    ]
    
    if survived_above_threshold:
        flipped_pairs = sum(1 for pr in survived_above_threshold if pr.sign_flipped)
        mean_flip_rate = flipped_pairs / len(survived_above_threshold)
        survived_above_threshold_count = len(survived_above_threshold)
    else:
        flipped_pairs = 0
        mean_flip_rate = 0.0
        survived_above_threshold_count = 0
    
    margin_ratios = []
    for pr in survived:
        if pr.before_margin > 0 and pr.after_margin is not None:
            ratio = pr.after_margin / pr.before_margin
            margin_ratios.append(ratio)
    
    mean_margin_retention = np.mean(margin_ratios) if margin_ratios else 0.0
    
    segmentation_survival_rate = survived_pairs / total_pairs
    
    usable_pairs = [pr for pr in pair_results if pr.is_usable()]
    usable_pair_yield = len(usable_pairs) / num_images
    
    stable_pairs = len(usable_pairs)
    
    return EvaluationMetrics(
        mean_flip_rate=mean_flip_rate,
        mean_margin_retention=mean_margin_retention,
        segmentation_survival_rate=segmentation_survival_rate,
        usable_pair_yield=usable_pair_yield,
        total_pairs=total_pairs,
        survived_pairs=survived_pairs,
        flipped_pairs=flipped_pairs,
        stable_pairs=stable_pairs,
        num_images=num_images,
        survived_above_threshold_count=survived_above_threshold_count
    )
 
 
def compute_metrics_by_stratum(
    pair_results: List[PairResult],
    stratum_key_fn,
    min_margin_threshold: float = 0.0
) -> dict[str, EvaluationMetrics]:
    strata = {}
    for pr in pair_results:
        key = stratum_key_fn(pr)
        if key not in strata:
            strata[key] = []
        strata[key].append(pr)
    
    return {
        stratum: compute_metrics(results, min_margin_threshold)
        for stratum, results in strata.items()
    }
 
 
def print_metrics_summary(metrics: EvaluationMetrics, combo_name: str = ""):
    header = f"=== Metrics for {combo_name} ===" if combo_name else "=== Metrics ==="
    print(header)
    print(f"Images evaluated: {metrics.num_images}")
    print(f"Total pairs: {metrics.total_pairs}")
    print(f"")
    print(f"Segmentation survival rate: {metrics.segmentation_survival_rate:.2%}")
    print(f"  → {metrics.survived_pairs}/{metrics.total_pairs} pairs had both regions survive")
    print(f"")
    print(f"Mean flip rate: {metrics.mean_flip_rate:.2%}")
    print(f"  → {metrics.flipped_pairs}/{metrics.survived_above_threshold_count} pairs flipped sign")
    print(f"")
    print(f"Mean margin retention: {metrics.mean_margin_retention:.3f}")
    print(f"  → Margin ratio averaged over surviving pairs")
    print(f"")
    print(f"Usable pair yield: {metrics.usable_pair_yield:.1f} pairs/image")
    print(f"  → {metrics.stable_pairs} stable pairs across {metrics.num_images} images")
    print(f"  → This is the practical bit budget per image")
    print("=" * len(header))
 
