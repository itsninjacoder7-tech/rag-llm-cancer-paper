"""
adaptive_k.py
=============
Adaptive top-k selection strategies for precision oncology RAG.

Three strategies are implemented:
  A — Score-Gap   : Largest gap in descending similarity score distribution.
  B — Threshold   : All documents with score >= theta.
  C — Complexity  : Query-complexity-weighted score-gap (primary contribution).

All strategies respect k_min and k_max bounds.
No LLM calls; no model fine-tuning; zero inference cost.

Usage:
    from context_retriever.adaptive_k import get_adaptive_k

    k = get_adaptive_k(
        method="C",
        scores=[0.92, 0.88, 0.71, 0.40, 0.38, 0.35],
        query_features={"complexity_score": 0.6},
        k_min=3,
        k_max=25,
    )
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_K_MIN: int = 3
DEFAULT_K_MAX: int = 25
DEFAULT_THETA: float = 0.65          # Threshold for Strategy B
GAP_EPSILON: float = 1e-6            # Minimum gap size to be considered meaningful


# ---------------------------------------------------------------------------
# Strategy A — Score-Gap (Taguchi-style, adapted for oncology hybrid scores)
# ---------------------------------------------------------------------------

def score_gap_adaptive_k(
    scores: List[float],
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
) -> int:
    """
    Select k by finding the largest gap in the sorted (descending) similarity
    score array.

    The intuition: a large drop between position i and i+1 means that
    document i+1 and beyond are substantially less relevant.  We cut there.

    Parameters
    ----------
    scores : list of float
        Similarity scores, already sorted descending (index 0 = most relevant).
        Accepts both dense (cosine, 0-1) and hybrid (RRF, any positive range).
    k_min : int
        Minimum number of documents to return (safety floor).
    k_max : int
        Maximum number of documents to consider (ceiling).

    Returns
    -------
    int
        Chosen k in [k_min, k_max].

    Notes
    -----
    - If scores is empty or shorter than k_min, returns k_min.
    - If all gaps are near-zero (uniform distribution), returns k_max.
    - Operates on the first k_max scores only.
    """
    scores = _validate_scores(scores, k_min, k_max)
    if scores is None:
        return k_min

    n = len(scores)
    if n <= k_min:
        return n

    # Only look for gaps within the window [k_min, k_max]
    # Gap at position i means: we include i documents (indices 0..i-1)
    # and cut before index i.  Search range: positions k_min..k_max-1.
    candidate_range = range(k_min, min(n, k_max))
    if not candidate_range:
        return k_min

    gaps = np.array([scores[i - 1] - scores[i] for i in candidate_range])
    max_gap_idx = int(np.argmax(gaps))
    best_k = k_min + max_gap_idx          # position in original scores array

    # If the maximum gap is negligible, fall back to k_max (no clear cutoff)
    if gaps[max_gap_idx] < GAP_EPSILON:
        logger.debug("score_gap: no meaningful gap found; returning k_max=%d", k_max)
        return min(k_max, n)

    logger.debug(
        "score_gap: gap=%.4f at position %d → k=%d",
        gaps[max_gap_idx], best_k, best_k,
    )
    return int(np.clip(best_k, k_min, k_max))


# ---------------------------------------------------------------------------
# Strategy B — Threshold-based
# ---------------------------------------------------------------------------

def threshold_adaptive_k(
    scores: List[float],
    theta: float = DEFAULT_THETA,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
) -> int:
    """
    Return all documents with similarity score >= theta, bounded by [k_min, k_max].

    Parameters
    ----------
    scores : list of float
        Similarity scores, sorted descending.
    theta : float
        Minimum similarity score to include a document.
        Default 0.65 is empirically derived for cosine similarity on
        biomedical sentence embeddings.  For RRF/hybrid scores, theta
        should be recalibrated (use 0.3 as a starting point).
    k_min : int
        Minimum k (floor).
    k_max : int
        Maximum k (ceiling).

    Returns
    -------
    int
        Chosen k in [k_min, k_max].

    Notes
    -----
    - At least k_min documents are always returned regardless of theta.
    - When zero documents exceed theta, returns k_min (graceful degradation).
    """
    scores = _validate_scores(scores, k_min, k_max)
    if scores is None:
        return k_min

    above = int(np.sum(np.array(scores) >= theta))
    k = int(np.clip(above, k_min, k_max))
    logger.debug("threshold: theta=%.2f → %d docs above threshold → k=%d", theta, above, k)
    return k


# ---------------------------------------------------------------------------
# Strategy C — Query-Complexity-Weighted (primary novel contribution)
# ---------------------------------------------------------------------------

def complexity_weighted_adaptive_k(
    query_features: dict,
    scores: List[float],
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
) -> int:
    """
    Determine adaptive k by combining:
      1. Query complexity (from query_features["complexity_score"]) to set a
         complexity-derived k range [k_lo, k_hi].
      2. Score-gap analysis within [k_lo, k_hi] to find the precise cutoff.

    This is the PRIMARY contribution: a domain-aware adaptive-k that uses
    oncology-specific query features to narrow the search space before
    applying the score-gap heuristic.

    Parameters
    ----------
    query_features : dict
        Output of ``query_complexity.extract_features()``.  Required keys:
            "complexity_score"  : float in [0, 1]
        Optional keys (used for logging/ablation):
            "n_biomarkers"      : int
            "cancer_type_rarity": float in [0, 1]
            "has_alteration_type": bool
            "query_length"      : int
    scores : list of float
        Similarity scores, sorted descending, up to k_max elements.
    k_min : int
        Global minimum k (floor, overrides complexity range).
    k_max : int
        Global maximum k (ceiling, overrides complexity range).

    Returns
    -------
    int
        Chosen k in [k_min, k_max].

    Complexity → Range mapping (tunable):
    ----------------------------------------
    complexity_score in [0.0, 0.33)  → low    → k range [k_min, 10]
    complexity_score in [0.33, 0.66) → medium → k range [8, 20]
    complexity_score in [0.66, 1.0]  → high   → k range [15, k_max]

    Within the complexity-derived range, Score-Gap (Strategy A) finds the
    exact cutoff.
    """
    complexity = float(query_features.get("complexity_score", 0.5))
    complexity = float(np.clip(complexity, 0.0, 1.0))

    # Map complexity to a k sub-range
    if complexity < 0.33:
        c_lo, c_hi = k_min, min(10, k_max)
        tier = "low"
    elif complexity < 0.66:
        c_lo, c_hi = max(k_min, 8), min(20, k_max)
        tier = "medium"
    else:
        c_lo, c_hi = max(k_min, 15), k_max
        tier = "high"

    # Ensure sub-range is valid
    c_lo = int(np.clip(c_lo, k_min, k_max))
    c_hi = int(np.clip(c_hi, c_lo, k_max))

    logger.debug(
        "complexity: score=%.3f tier=%s → range=[%d, %d]",
        complexity, tier, c_lo, c_hi,
    )

    # Apply score-gap within the complexity-derived range
    k = score_gap_adaptive_k(scores, k_min=c_lo, k_max=c_hi)
    return k


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def get_adaptive_k(
    method: str,
    scores: List[float],
    query_features: Optional[dict] = None,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    theta: float = DEFAULT_THETA,
) -> int:
    """
    Dispatch to the chosen adaptive-k strategy.

    Parameters
    ----------
    method : str
        One of "A", "B", or "C".
    scores : list of float
        Similarity scores, sorted descending.
    query_features : dict, optional
        Required for method "C".  Ignored for A and B.
    k_min : int
        Global minimum k.
    k_max : int
        Global maximum k.
    theta : float
        Threshold for Strategy B.

    Returns
    -------
    int
        Chosen k in [k_min, k_max].

    Raises
    ------
    ValueError
        If method is not one of "A", "B", "C".
    ValueError
        If method is "C" and query_features is None.
    """
    method = method.upper().strip()

    if method == "A":
        return score_gap_adaptive_k(scores, k_min=k_min, k_max=k_max)

    elif method == "B":
        return threshold_adaptive_k(scores, theta=theta, k_min=k_min, k_max=k_max)

    elif method == "C":
        if query_features is None:
            raise ValueError(
                "query_features must be provided for adaptive-k method 'C'. "
                "Call query_complexity.extract_features(query_text) first."
            )
        return complexity_weighted_adaptive_k(
            query_features=query_features,
            scores=scores,
            k_min=k_min,
            k_max=k_max,
        )

    else:
        raise ValueError(
            f"Unknown adaptive-k method: '{method}'. "
            "Choose one of: 'A' (score-gap), 'B' (threshold), 'C' (complexity-weighted)."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_scores(
    scores: List[float],
    k_min: int,
    k_max: int,
) -> Optional[np.ndarray]:
    """
    Validate and prepare the scores array.

    Returns a numpy array truncated to k_max, or None if the input is
    degenerate (empty list, all NaN, all identical).
    """
    if not scores:
        logger.warning("adaptive_k: received empty scores list; returning k_min=%d", k_min)
        return None

    arr = np.array(scores, dtype=float)

    if np.all(np.isnan(arr)):
        logger.warning("adaptive_k: all scores are NaN; returning k_min=%d", k_min)
        return None

    # Replace NaN with the column minimum so they sort to the bottom
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr[nan_mask] = np.nanmin(arr) - 1.0

    # Truncate to k_max
    arr = arr[:k_max]

    # Ensure descending order (caller should already provide sorted scores,
    # but we re-sort defensively)
    if not np.all(arr[:-1] >= arr[1:]):
        logger.debug("adaptive_k: scores not sorted; re-sorting descending")
        arr = np.sort(arr)[::-1]

    return arr
