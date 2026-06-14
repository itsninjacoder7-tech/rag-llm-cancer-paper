"""
query_complexity.py
===================
Extracts complexity features from precision oncology query strings.

These features feed Strategy C (complexity_weighted_adaptive_k) to determine
how many documents should be retrieved for a given query.

The core insight: in the MOAlmanac knowledge base, queries involving:
  - Multiple biomarkers   → more documents needed (higher k)
  - Rare cancer types     → fewer matching entries, but need broader search
  - Specific alteration types (e.g., amplification vs. mutation) → more targeted

All feature extraction is rule-based and regex-based — zero external API calls.

Usage:
    from context_retriever.query_complexity import extract_features

    features = extract_features("BRAF V600E mutation in metastatic melanoma")
    # → {"n_biomarkers": 1, "cancer_type_rarity": 0.1, ..., "complexity_score": 0.31}
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Biomarker vocabulary
# ---------------------------------------------------------------------------

# Common single-gene biomarkers found in MOAlmanac
_COMMON_GENES: List[str] = [
    "BRAF", "KRAS", "NRAS", "EGFR", "ALK", "ROS1", "MET", "HER2", "ERBB2",
    "PIK3CA", "PTEN", "TP53", "RB1", "CDKN2A", "FGFR1", "FGFR2", "FGFR3",
    "IDH1", "IDH2", "BRCA1", "BRCA2", "APC", "RET", "PDGFRA", "KIT",
    "FLT3", "NPM1", "DNMT3A", "TET2", "SF3B1", "U2AF1", "SRSF2",
    "NTRK1", "NTRK2", "NTRK3", "NF1", "NF2", "VHL", "MLH1", "MSH2",
    "MSH6", "PMS2", "ATM", "CHEK2", "PALB2", "RAD51C", "RAD51D",
    "POLE", "POLD1", "TMB", "MSI", "PD-L1", "CD274",
    "MDM2", "MDM4", "CDK4", "CDK6", "CCND1", "CCND2", "CCND3",
    "MYC", "MYCN", "BCL2", "BCL6", "IGH", "ABL1", "BCR",
    "JAK2", "STAT3", "STAT5", "PTCH1", "SMO", "GLI1", "GLI2",
    "AKT1", "AKT2", "AKT3", "MTOR", "TSC1", "TSC2", "STK11",
    "SMAD4", "FBXW7", "ARID1A", "ARID2", "SMARCA4", "SMARCB1",
]

# Alteration type keywords
_ALTERATION_TYPES: List[str] = [
    "mutation", "amplification", "deletion", "fusion", "rearrangement",
    "overexpression", "loss", "gain", "copy number", "translocation",
    "insertion", "frameshift", "splice", "nonsense", "missense",
    "exon skipping", "kinase domain", "activation", "inactivation",
    "methylation", "expression",
]

# Specific hotspot patterns (e.g., V600E, G12D, L858R)
_HOTSPOT_PATTERN: re.Pattern = re.compile(
    r"\b[A-Z]\d{1,4}[A-Z*](?:fs\*\d*)?\b"
)

# ---------------------------------------------------------------------------
# Cancer type rarity table
# ---------------------------------------------------------------------------
# Derived from approximate MOAlmanac entry frequency.
# 1.0 = most common (many entries), 0.0 = very rare (few entries).
# Lower rarity → more context needed → higher k.

_CANCER_RARITY: Dict[str, float] = {
    # Common (many MOAlmanac entries) → lower rarity score
    "melanoma": 0.10,
    "lung": 0.10,
    "nsclc": 0.10,
    "non-small cell lung": 0.10,
    "breast": 0.12,
    "colorectal": 0.15,
    "colon": 0.15,
    "leukemia": 0.15,
    "aml": 0.15,
    "cll": 0.18,
    "lymphoma": 0.18,
    "glioblastoma": 0.20,
    "glioma": 0.20,
    "prostate": 0.22,
    "ovarian": 0.22,
    "pancreatic": 0.25,
    "gastric": 0.28,
    "bladder": 0.30,
    "renal": 0.30,
    "thyroid": 0.30,
    "hepatocellular": 0.35,
    "liver": 0.35,
    "endometrial": 0.38,
    "uterine": 0.38,
    # Moderately rare
    "sarcoma": 0.55,
    "mesothelioma": 0.60,
    "myeloma": 0.50,
    "esophageal": 0.55,
    "cervical": 0.58,
    "head and neck": 0.45,
    "squamous": 0.40,
    "cholangiocarcinoma": 0.65,
    "bile duct": 0.65,
    "gallbladder": 0.70,
    "small cell": 0.45,
    "sclc": 0.45,
    # Rare / ultra-rare
    "appendiceal": 0.80,
    "adrenocortical": 0.80,
    "pheochromocytoma": 0.82,
    "paraganglioma": 0.82,
    "thymoma": 0.85,
    "uveal melanoma": 0.75,
    "merkel cell": 0.88,
    "penile": 0.90,
    "vaginal": 0.90,
    "vulvar": 0.90,
}

_DEFAULT_RARITY: float = 0.60   # unknown cancer type → treat as moderately rare

# ---------------------------------------------------------------------------
# Multi-biomarker indicator patterns
# ---------------------------------------------------------------------------

_AND_PATTERNS: List[re.Pattern] = [
    re.compile(r"\band\b", re.IGNORECASE),
    re.compile(r"\bwith\b", re.IGNORECASE),
    re.compile(r"\bco-?mutation\b", re.IGNORECASE),
    re.compile(r"\bco-?occurring\b", re.IGNORECASE),
    re.compile(r"\bplus\b", re.IGNORECASE),
    re.compile(r"[+&]"),
]

# ---------------------------------------------------------------------------
# Disease status keywords
# ---------------------------------------------------------------------------

_DISEASE_STATUS_WORDS: List[str] = [
    "metastatic", "advanced", "recurrent", "relapsed", "refractory",
    "early-stage", "locally advanced", "unresectable", "stage iv",
    "stage iii", "stage ii", "stage i", "first-line", "second-line",
    "resistant", "sensitive", "naive", "pretreated",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(query_text: str) -> Dict[str, object]:
    """
    Extract complexity features from a precision oncology query.

    Parameters
    ----------
    query_text : str
        Raw query string, e.g.:
        "BRAF V600E mutation in metastatic melanoma with CDKN2A loss"

    Returns
    -------
    dict with keys:
        n_biomarkers       : int   — number of distinct biomarker mentions
        n_hotspots         : int   — number of specific hotspot patterns (e.g., V600E)
        n_alteration_types : int   — number of alteration-type keywords
        cancer_type_rarity : float — rarity score of identified cancer type (0–1)
        has_disease_status : bool  — query contains disease stage/status info
        query_length       : int   — character length of query
        complexity_score   : float — composite score in [0, 1] (higher = harder)
    """
    if not isinstance(query_text, str) or not query_text.strip():
        logger.warning("extract_features: received empty or non-string query")
        return _default_features()

    query_upper = query_text.upper()

    # --- Feature 1: Number of biomarker gene mentions ---
    n_biomarkers = sum(
        1 for gene in _COMMON_GENES
        if re.search(r"\b" + gene + r"\b", query_upper)
    )
    # Also count any multi-biomarker conjunction patterns
    multi_bio_signals = sum(
        1 for pat in _AND_PATTERNS if pat.search(query_text)
    )
    # Combine: if conjunctions present and at least 1 gene, assume 2+
    if multi_bio_signals > 0 and n_biomarkers == 1:
        n_biomarkers = 2
    n_biomarkers = max(n_biomarkers, 0)

    # --- Feature 2: Hotspot specificity ---
    hotspots = _HOTSPOT_PATTERN.findall(query_upper)
    n_hotspots = len(hotspots)

    # --- Feature 3: Alteration type keywords ---
    query_lower = query_text.lower()
    n_alteration_types = sum(
        1 for alt in _ALTERATION_TYPES if alt in query_lower
    )

    # --- Feature 4: Cancer type rarity ---
    cancer_type_rarity = _detect_cancer_rarity(query_lower)

    # --- Feature 5: Disease status ---
    has_disease_status = any(ds in query_lower for ds in _DISEASE_STATUS_WORDS)

    # --- Feature 6: Query length ---
    query_length = len(query_text.strip())

    # --- Feature 7: Composite complexity score ---
    complexity_score = _compute_complexity_score(
        n_biomarkers=n_biomarkers,
        n_hotspots=n_hotspots,
        n_alteration_types=n_alteration_types,
        cancer_type_rarity=cancer_type_rarity,
        has_disease_status=has_disease_status,
        query_length=query_length,
    )

    features = {
        "n_biomarkers": n_biomarkers,
        "n_hotspots": n_hotspots,
        "n_alteration_types": n_alteration_types,
        "cancer_type_rarity": cancer_type_rarity,
        "has_disease_status": has_disease_status,
        "query_length": query_length,
        "complexity_score": round(complexity_score, 4),
    }

    logger.debug("query_complexity: %s", features)
    return features


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_cancer_rarity(query_lower: str) -> float:
    """
    Return the rarity score for the cancer type mentioned in the query.
    Lower score = more common cancer (e.g., melanoma = 0.10).
    """
    best_rarity = _DEFAULT_RARITY
    best_match_len = 0
    for cancer, rarity in _CANCER_RARITY.items():
        if cancer in query_lower:
            if len(cancer) > best_match_len:
                best_match_len = len(cancer)
                best_rarity = rarity
    return best_rarity


def _compute_complexity_score(
    n_biomarkers: int,
    n_hotspots: int,
    n_alteration_types: int,
    cancer_type_rarity: float,
    has_disease_status: bool,
    query_length: int,
) -> float:
    """
    Produce a composite complexity score in [0, 1].

    Component weights (sum = 1.0):
      biomarker_count    : 0.35  (most important — drives need for more context)
      cancer_rarity      : 0.25  (rare cancer = fewer clear matches = need more k)
      alteration_type    : 0.15  (specific alteration narrows search; low k needed)
      hotspot            : 0.10  (specific hotspot = low k; absence = high k)
      disease_status     : 0.10  (additional dimension = more context needed)
      query_length       : 0.05  (proxy for overall specificity)

    Scoring logic (counter-intuitive for some features):
    - More biomarkers → higher complexity (need broader retrieval)
    - Rarer cancer   → higher complexity (fewer exact matches)
    - More alteration types → LOWER complexity (more specific = fewer docs needed)
    - Specific hotspot present → LOWER complexity (very targeted)
    - Disease status present → slightly higher complexity
    - Longer query → slightly higher complexity
    """
    # Biomarker contribution (0–1): saturates at 4+
    bio_score = min(n_biomarkers / 4.0, 1.0)

    # Cancer rarity contribution (already 0–1, higher = more complex)
    rarity_score = cancer_type_rarity

    # Alteration type: more specific = lower complexity needed
    # 0 types = 1.0 (unknown), 1 type = 0.5 (known), 2+ = 0.2 (very specific)
    if n_alteration_types == 0:
        alt_score = 1.0
    elif n_alteration_types == 1:
        alt_score = 0.5
    else:
        alt_score = 0.2

    # Hotspot specificity: hotspot present = simpler retrieval
    hotspot_score = 0.0 if n_hotspots >= 1 else 1.0

    # Disease status: adds context dimension
    status_score = 1.0 if has_disease_status else 0.0

    # Query length: normalize to [0, 1] over range [20, 300]
    length_score = min(max((query_length - 20) / 280.0, 0.0), 1.0)

    complexity = (
        0.35 * bio_score
        + 0.25 * rarity_score
        + 0.15 * alt_score
        + 0.10 * hotspot_score
        + 0.10 * status_score
        + 0.05 * length_score
    )

    return float(min(max(complexity, 0.0), 1.0))


def _default_features() -> Dict[str, object]:
    """Return safe default features for empty/invalid queries."""
    return {
        "n_biomarkers": 0,
        "n_hotspots": 0,
        "n_alteration_types": 0,
        "cancer_type_rarity": _DEFAULT_RARITY,
        "has_disease_status": False,
        "query_length": 0,
        "complexity_score": 0.5,
    }
