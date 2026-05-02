# watermark/pair_pool.py
#
# Builds the candidate pair pool from region descriptors, filters by margin,
# splits by natural sign, and shuffles with the secret key.
#
# Logic extracted from stability/pairwise_stability.py (evaluation) and adapted
# for embedding: instead of measuring all pairs, we select the best ones and
# assign them to RS-encoded bit positions.
#
# Sort → shuffle order matters:
#   Sort first so highest-margin (most stable) pairs are at the front.
#   Shuffle with key second for security (makes assignment unpredictable without key).
#   assign_pairs_to_bits pops from the front, so the most stable pairs are used first.

import random
import itertools
from dataclasses import dataclass
from collections import deque

MIN_MARGIN_DEFAULT = 0.05


@dataclass
class PairPool:
    pool_1: list    # list[tuple[int,int]] — pairs where d[i] > d[j]; encode bit 1
    pool_0: list    # list[tuple[int,int]] — pairs where d[i] < d[j]; encode bit 0
    total_capacity: int  # max embeddable bits = min(len(pool_1), len(pool_0)) // k


def build_pair_pool(descriptors, key, min_margin=MIN_MARGIN_DEFAULT, k=7) -> PairPool:
    """
    Build and shuffle the pair pool from region descriptor values.

    Parameters
    ----------
    descriptors : list[float]
        One scalar per region (e.g. DWT-LL energy), length = num_regions (~200).
    key : int | str | bytes
        Random seed for shuffling.  Consumed entirely here; decoder does not need it.
    min_margin : float
        Minimum |d[i] - d[j]| for a pair to be included.  Pairs below this
        are too close to be reliably ordered after AI editing.
    k : int
        Witnesses per bit.  Used only to compute total_capacity.

    Returns
    -------
    PairPool
    """
    n = len(descriptors)

    # Step 1+2: Generate all (i,j) pairs with i<j, compute margin, filter.
    candidates = []
    for i, j in itertools.combinations(range(n), 2):
        margin = abs(descriptors[i] - descriptors[j])
        if margin >= min_margin:
            candidates.append((margin, i, j))

    # Step 3+4: Sort descending by margin — highest-margin pairs at front.
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Step 5: Split by natural sign.
    pool_1 = [(i, j) for _m, i, j in candidates if descriptors[i] > descriptors[j]]
    pool_0 = [(i, j) for _m, i, j in candidates if descriptors[i] < descriptors[j]]

    # Step 6: Shuffle each pool independently with the key.
    rng = random.Random(key)
    rng.shuffle(pool_1)
    rng.shuffle(pool_0)

    total_capacity = min(len(pool_1), len(pool_0)) // k

    return PairPool(pool_1=pool_1, pool_0=pool_0, total_capacity=total_capacity)


def assign_pairs_to_bits(pool, encoded_bits, k) -> list:
    """
    Assign K pairs from the pool to each encoded bit position.

    Pops from the front of each pool so highest-margin pairs (placed at front
    by build_pair_pool's sort) are assigned first.

    Parameters
    ----------
    pool         : PairPool
    encoded_bits : list[int]  — RS-encoded bit array (0s and 1s), e.g. length 112
    k            : int        — witnesses per bit

    Returns
    -------
    list[list[tuple[int, int]]]
        pairs_for_bits[j] = list of K (region_i, region_j) tuples for bit j.
        This is exactly what goes into Sidecar.pairs.
    """
    dq1 = deque(pool.pool_1)
    dq0 = deque(pool.pool_0)

    pairs_for_bits = []
    for bit in encoded_bits:
        if bit == 1:
            if len(dq1) < k:
                raise ValueError(
                    f"pool_1 exhausted ({len(dq1)} pairs left, need {k}). "
                    "Use a shorter message, smaller K, or verify total_capacity before calling."
                )
            pairs_for_bits.append([dq1.popleft() for _ in range(k)])
        else:
            if len(dq0) < k:
                raise ValueError(
                    f"pool_0 exhausted ({len(dq0)} pairs left, need {k}). "
                    "Use a shorter message, smaller K, or verify total_capacity before calling."
                )
            pairs_for_bits.append([dq0.popleft() for _ in range(k)])

    return pairs_for_bits
