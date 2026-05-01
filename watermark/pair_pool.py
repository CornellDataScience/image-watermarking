# watermark/pair_pool.py
#
# Responsible for: building the candidate pair pool from region descriptors,
# filtering by margin, splitting by natural sign, and shuffling with the
# secret key so that pair-to-bit assignment is unpredictable without the key.
#
# This is a standalone extraction of the logic that already exists in
# stability/pairwise_stability.py for evaluation purposes.  The difference:
#   - pairwise_stability.py  → evaluates ALL pairs, measures flip/survival
#   - pair_pool.py           → selects the BEST pairs and assigns them to bits
#
# ─────────────────────────────────────────────────────────────────────────────
# DATA FLOW
# ─────────────────────────────────────────────────────────────────────────────
#
#   descriptors: list[float]   (length = num_regions, e.g. 200 LL-energy values)
#       │
#       ▼
#   build_pair_pool(descriptors, min_margin, key)
#       │
#       ├── generates all (i, j) pairs with i < j  →  ~19,900 pairs for n=200
#       ├── computes margin = |d[i] - d[j]| for each
#       ├── discards pairs where margin < min_margin (default 0.05)
#       ├── sorts survivors by margin descending
#       │       (highest-margin pairs are most stable → assigned first)
#       ├── splits into two pools:
#       │       pool[1]: pairs where d[i] > d[j]  (naturally encodes bit 1)
#       │       pool[0]: pairs where d[i] < d[j]  (naturally encodes bit 0)
#       └── shuffles each pool independently using `key` as the random seed
#               (without the key, an attacker with the sidecar cannot
#                reconstruct which pool positions map to which bit positions)
#       │
#       ▼
#   PairPool(pool_1=list[tuple[int,int]], pool_0=list[tuple[int,int]])
#
# ─────────────────────────────────────────────────────────────────────────────
# KEY TYPES
# ─────────────────────────────────────────────────────────────────────────────
#
#   min_margin : float
#       Minimum absolute difference |d[i] - d[j]| for a pair to be eligible.
#       Empirically set to 0.05 (from RESULTS_EXTENDED.md).
#       Pairs below this are too close to be reliably ordered after editing.
#
#   key : int | bytes | str
#       The secret seed.  Used only to shuffle pool[0] and pool[1] before
#       pair assignment.  Consumed entirely at encode time; decoder does NOT
#       need the key (it reads the fully-materialized pair table from the sidecar).
#
#   PairPool : dataclass
#       pool_1 : list[tuple[int, int]]
#           Shuffled list of (i, j) pairs where d[i] > d[j] naturally.
#           The encoder pops from the front of this list for every bit position
#           that encodes a 1.  Front = highest margin = most stable.
#       pool_0 : list[tuple[int, int]]
#           Same structure for bit positions that encode a 0.
#       total_capacity : int
#           min(len(pool_1), len(pool_0)) * K — rough upper bound on how many
#           bits can be embedded.  Caller should check this is ≥ n_encoded_bits.
#
# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass


MIN_MARGIN_DEFAULT = 0.05


@dataclass
class PairPool:
    # Pairs where d[i] > d[j]  →  naturally encode bit 1
    pool_1: list  # list[tuple[int, int]]

    # Pairs where d[i] < d[j]  →  naturally encode bit 0
    pool_0: list  # list[tuple[int, int]]

    # How many bits COULD be embedded given pool sizes and K
    # = min(len(pool_1), len(pool_0)) // K
    # Encoder checks: n_encoded_bits <= total_capacity
    total_capacity: int


def build_pair_pool(
    descriptors,   # list[float], length = num_regions (200 for SLIC n=200)
    key,           # int | bytes | str — random seed for shuffling
    min_margin=MIN_MARGIN_DEFAULT,  # float — discard pairs below this margin
    k=7,           # int — witnesses per bit (K parameter from the design doc)
                   #        use 7 for easy edits, 11 for hard (MagicBrush)
) -> PairPool:
    # Step 1: Generate all (i, j) pairs with i < j.
    #   For n=200 regions: C(200, 2) = 19,900 pairs.
    #   Implementation: nested loop or itertools.combinations(range(n), 2).

    # Step 2: Compute margin for each pair.
    #   margin(i, j) = abs(descriptors[i] - descriptors[j])
    #   This is the same margin used in pairwise_stability.py evaluation.

    # Step 3: Filter pairs where margin < min_margin.
    #   Empirically ~7,000–13,000 pairs survive per image (RESULTS_EXTENDED.md).
    #   These are the pairs stable enough to carry a watermark bit.

    # Step 4: Sort survivors by margin descending.
    #   Highest-margin pairs go to the front of the pool.
    #   Because the encoder pops from the front for each bit position,
    #   bit positions with lower index get the most stable (highest-margin) pairs.
    #   This means RS redundancy bits are backed by the best pairs.

    # Step 5: Split into pool_1 and pool_0.
    #   pool_1: pairs where descriptors[i] > descriptors[j]  →  sign = +1
    #   pool_0: pairs where descriptors[i] < descriptors[j]  →  sign = -1
    #   (Equal descriptors should never reach this point due to margin filter,
    #    but add an assertion just in case.)

    # Step 6: Shuffle each pool independently using the key.
    #   Use random.Random(key).shuffle(pool_1) and similarly for pool_0.
    #   This preserves the existing sort order statistically but makes
    #   the specific assignment unpredictable without the key.
    #   NOTE: the sort-then-shuffle order matters.  Sort first so that
    #   margin ordering is the dominant signal; shuffle second so the
    #   key provides security.  Do NOT shuffle before sorting.

    # Step 7: Compute total_capacity.
    #   total_capacity = min(len(pool_1), len(pool_0)) // k
    #   If total_capacity < n_encoded_bits the encoder must abort or
    #   fall back to a shorter message / smaller K.

    # Return PairPool(pool_1=..., pool_0=..., total_capacity=...)
    pass


def assign_pairs_to_bits(
    pool,            # PairPool — the output of build_pair_pool
    encoded_bits,    # list[int] — RS-encoded bit array, e.g. length 960
    k,               # int — witnesses per bit position
) -> list:           # list[list[tuple[int, int]]]
                     #   length = len(encoded_bits)
                     #   pairs_for_bit[j] = list of K (region_i, region_j) tuples
    # For each bit position j in encoded_bits:
    #   if encoded_bits[j] == 1: pop k pairs from pool.pool_1
    #   if encoded_bits[j] == 0: pop k pairs from pool.pool_0
    #   Record the K pairs as pairs_for_bit[j].
    #
    # Pop from the FRONT (pop(0) or use a deque) so that the highest-margin
    # pairs (placed at front by the sort in build_pair_pool) are assigned first.
    #
    # If either pool runs out of pairs mid-assignment, raise an error.
    # The caller should have verified total_capacity beforehand.
    #
    # Returns: list of length len(encoded_bits),
    #   each element is a list of K (i, j) tuples.
    #   This is exactly what goes into Sidecar.pairs.
    pass
