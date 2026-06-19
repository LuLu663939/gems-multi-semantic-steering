"""GEMS: Geometric Constraints Enable Multi-Semantic Superposition in LLMs.

Core library for multi-directional activation steering via geometric constraints.
"""

from gems.hooks import GEMSHook, ActAddHook
from gems.extraction import extract_vecs, compute_diff_vectors, compute_raw_vectors
from gems.utils import (
    get_out_proj,
    NaNSafeLogitsProcessor,
    DEFAULT_PEAK_LAYER,
    DEFAULT_SIGMA,
    DEFAULT_COSINE_DECAY_START,
    DEFAULT_COSINE_DECAY_SPAN,
    DEFAULT_INTERVENTION_LAYERS,
    DEFAULT_INTENSITIES,
    DEFAULT_MAX_GEN_TOKENS,
    DEFAULT_TEMPERATURE,
)

__all__ = [
    "GEMSHook",
    "ActAddHook",
    "extract_vecs",
    "compute_diff_vectors",
    "compute_raw_vectors",
    "get_out_proj",
    "NaNSafeLogitsProcessor",
    "DEFAULT_PEAK_LAYER",
    "DEFAULT_SIGMA",
    "DEFAULT_COSINE_DECAY_START",
    "DEFAULT_COSINE_DECAY_SPAN",
    "DEFAULT_INTERVENTION_LAYERS",
    "DEFAULT_INTENSITIES",
    "DEFAULT_MAX_GEN_TOKENS",
    "DEFAULT_TEMPERATURE",
]
